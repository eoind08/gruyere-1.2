import tiktoken
from transformers.integrations.tiktoken import convert_tiktoken_to_fast
from transformers import PreTrainedTokenizerFast

OUTPUT_DIR = "./Gruyere-1.2-HF"  # write straight into your final checkpoint folder

USER_TOKEN_ID = 50257
ASSISTANT_TOKEN_ID = 50258


def build_encoding():
    """Same extension used in prepare_data.py / train.py -- must stay in sync."""
    base = tiktoken.get_encoding("gpt2")
    special_tokens = dict(base._special_tokens)
    special_tokens["<|user|>"] = USER_TOKEN_ID
    special_tokens["<|assistant|>"] = ASSISTANT_TOKEN_ID
    return tiktoken.Encoding(
        name="gpt2_gruyere_chat",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens=special_tokens,
    )


print("Building extended GPT-2 + chat-role tiktoken encoding...")
enc = build_encoding()

print("Converting to Hugging Face tokenizer...")
convert_tiktoken_to_fast(enc, OUTPUT_DIR)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=f"{OUTPUT_DIR}/tokenizer.json",
    eos_token="<|endoftext|>",
    bos_token="<|endoftext|>",
    extra_special_tokens={"user_token": "<|user|>", "assistant_token": "<|assistant|>"},
)

tokenizer.model_max_length = 1024

tokenizer.save_pretrained(OUTPUT_DIR)

print("Tokenizer saved.")

test_strings = [
    "Hello world",
    "The capital of France is Paris.",
    "Plants need water to survive.",
    "This is a test with punctuation!",
    "Hello, world! How are you?",
    "The quick brown fox jumps over the lazy dog.",
    "  leading spaces",
    "trailing spaces  ",
]

print("\nVerifying tokenizer against tiktoken...")

for text in test_strings:
    tiktoken_ids = enc.encode_ordinary(text)  # ordinary text never touches specials
    hf_ids = tokenizer.encode(text, add_special_tokens=False)

    if tiktoken_ids != hf_ids:
        print(f"\nMismatch for: {repr(text)}")
        print("tiktoken:", tiktoken_ids)
        print("HF:      ", hf_ids)
        raise RuntimeError("Tokenizer conversion failed.")

print("All tokenization tests passed.")

# New: confirm the two role tokens landed at the right ids
for tok, expected_id in [("<|user|>", USER_TOKEN_ID), ("<|assistant|>", ASSISTANT_TOKEN_ID)]:
    got_id = tokenizer.convert_tokens_to_ids(tok)
    assert got_id == expected_id, f"{tok} got id {got_id}, expected {expected_id}"
    assert tok in tokenizer.all_special_tokens
print("Role token ids verified: <|user|>=%d, <|assistant|>=%d" % (USER_TOKEN_ID, ASSISTANT_TOKEN_ID))

print(f"\nVocabulary size: {len(tokenizer)}")
print(f"EOS token ID: {tokenizer.eos_token_id}")
print(f"Saved to: {OUTPUT_DIR}")