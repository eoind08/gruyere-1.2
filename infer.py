import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_MODEL = "Gruyere-1.2-HF"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 256
TEMPERATURE = 1
TOP_K = 40
DISPLAY_TOP_N = 20

print(f"Loading model on {DEVICE}...")

hf_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
hf_model = AutoModelForCausalLM.from_pretrained(HF_MODEL, trust_remote_code=True, torch_dtype=torch.float32)
hf_model.eval().to(DEVICE)

print("Model loaded!")

def get_token_distribution(prompt):
    tokens = hf_tokenizer.encode(prompt, add_special_tokens=False)
    max_context = getattr(hf_model.config, "max_position_embeddings", 1024)
    x = torch.tensor([tokens[-max_context:]], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        logits = hf_model(x).logits[:, -1, :]

    if TEMPERATURE > 0:
        logits = logits / TEMPERATURE
        if TOP_K is not None and TOP_K > 0:
            top_k_values, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
            cutoff = top_k_values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
        return torch.softmax(logits, dim=-1)[0]

    probabilities = torch.zeros_like(logits)
    probabilities.scatter_(1, logits.argmax(dim=-1, keepdim=True), 1.0)
    return probabilities[0]

def display_distribution(prompt):
    probabilities = get_token_distribution(prompt)
    above_one_percent = (probabilities > 0.01).sum().item()
    top_probs, top_ids = torch.topk(probabilities, min(DISPLAY_TOP_N, probabilities.numel()))

    print("\n" + "-" * 70)
    print(f"Prompt: {prompt!r}")
    print("-" * 70)
    print(f"Temperature: {TEMPERATURE} | Top-k: {TOP_K}")
    print(f"Tokens with probability > 1%: {above_one_percent}")
    print("\nTop tokens:")

    for rank, (prob, token_id) in enumerate(zip(top_probs.tolist(), top_ids.tolist()), 1):
        token = hf_tokenizer.decode([token_id], skip_special_tokens=False)
        token = token.replace("\n", "\\n").replace("\t", "\\t")
        print(f"{rank:2d}. {token!r:<20} {prob * 100:8.3f}% (token {token_id})")

def generate(prompt):
    tokens = hf_tokenizer.encode(prompt, add_special_tokens=False)
    max_context = getattr(hf_model.config, "max_position_embeddings", 1024)
    x = torch.tensor([tokens[-max_context:]], dtype=torch.long, device=DEVICE)
    generated_tokens = []
    eos_token_id = hf_tokenizer.eos_token_id

    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            logits = hf_model(x[:, -max_context:]).logits[:, -1, :]

            if TEMPERATURE > 0:
                logits = logits / TEMPERATURE
                if TOP_K is not None and TOP_K > 0:
                    top_k_values, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
                    cutoff = top_k_values[:, -1].unsqueeze(-1)
                    logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
                probabilities = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            token_id = next_token.item()
            if eos_token_id is not None and token_id == eos_token_id:
                break

            generated_tokens.append(token_id)
            x = torch.cat((x, next_token), dim=1)

    return hf_tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("\n" + "=" * 70)
print("GRUYÈRE-1.2 INTERACTIVE")
print("=" * 70)
print(f"Temperature = {TEMPERATURE} | Top-k = {TOP_K} | Max tokens = {MAX_NEW_TOKENS}")

while True:
    prompt = input("\nPrompt: ")
    if prompt.lower() in {"exit", "quit"}:
        break
    display_distribution(prompt)
    print("\nGeneration:")
    print(generate(prompt))