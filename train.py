"""
train.py -- SFT Gruyere-1.1 -> Gruyere-1.2 on ultrachat_200k

Run prepare_data.py first to produce token/mask shards, then:

    pip install torch transformers tiktoken numpy huggingface_hub
    python train.py --data_dir data/ultrachat --out_dir gruyere-1.2

Design notes
------------
* Loads Erbium08/Gruyere-1.1 through `transformers` (trust_remote_code=True):
  it ships config.json + modeling_gruyere.py, so AutoModelForCausalLM just
  works. No embedding resize is needed -- the model's vocab is already
  padded to 50304 and prepare_data.py's new <|user|>/<|assistant|> tokens
  (50257/50258) live in previously-unused rows of that same table.

* The model's forward(labels=...) already does
  F.cross_entropy(logits, labels)  with the default ignore_index=-100,
  so the loss-masking scheme is: real token id where we want it scored,
  -100 everywhere else. next_batch() below builds exactly that from the
  uint16 token shards + uint8 mask shards on the fly.

* Hybrid Muon (for 2D matmul weights) + AdamW (for everything else,
  embeddings included) optimizer, matching the training style of the
  original gruyere-1.1/train.py this is based on. torch.optim.Muon isn't
  in every torch release yet, so a small self-contained Muon
  implementation (Newton-Schulz orthogonalized momentum SGD) is used as
  a drop-in if the built-in one isn't available.

* An optional monkey-patch swaps the shipped attention forward (which
  builds an explicit additive mask so it can support KV-cache/padding
  during generation) for a plain `is_causal=True` SDPA call during
  training, where we never pad. That's mathematically identical for our
  case and lets SDPA pick its fastest (flash-attention) kernel.
"""

import argparse
import glob
import math
import os
import time

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

# ----------------------------------------------------------------------------
# Token ids (must match prepare_data.py)
# ----------------------------------------------------------------------------
EOT_TOKEN_ID = 50256
USER_TOKEN_ID = 50257
ASSISTANT_TOKEN_ID = 50258


def build_encoding():
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


# ----------------------------------------------------------------------------
# Data loading: contiguous shards of uint16 tokens + uint8 loss mask
# ----------------------------------------------------------------------------
class ShardedTokenLoader:
    def __init__(self, data_dir, split, B, T):
        self.B, self.T = B, T
        self.token_paths = sorted(glob.glob(os.path.join(data_dir, split, f"shard_{split}_*_tokens.npy")))
        self.mask_paths = sorted(glob.glob(os.path.join(data_dir, split, f"shard_{split}_*_mask.npy")))
        assert len(self.token_paths) > 0, f"no shards found under {data_dir}/{split} -- run prepare_data.py first"
        assert len(self.token_paths) == len(self.mask_paths)
        self.epoch = 0
        self.shard_idx = -1
        self.pos = 0
        self._load_shard(0)

    def _load_shard(self, idx):
        # mmap so we don't pull whole shards into RAM until sliced
        self.tokens = np.load(self.token_paths[idx], mmap_mode="r")
        self.mask = np.load(self.mask_paths[idx], mmap_mode="r")
        self.shard_idx = idx
        self.pos = 0

    def next_batch(self):
        B, T = self.B, self.T
        needed = B * T + 1
        while self.pos + needed > len(self.tokens):
            next_idx = self.shard_idx + 1
            if next_idx >= len(self.token_paths):
                next_idx = 0
                self.epoch += 1
            self._load_shard(next_idx)
        tok = np.asarray(self.tokens[self.pos:self.pos + needed], dtype=np.int64)
        msk = np.asarray(self.mask[self.pos:self.pos + needed], dtype=np.int64)
        self.pos += B * T

        x = torch.from_numpy(tok[:-1]).view(B, T)
        y = torch.from_numpy(tok[1:]).view(B, T)
        m = torch.from_numpy(msk[1:]).view(B, T).bool()
        labels = torch.where(m, y, torch.full_like(y, -100))
        return x, labels

    def reset(self):
        self._load_shard(0)
        self.epoch = 0

    @property
    def total_tokens(self):
        # cheap enough: just the shards' file sizes, computed once and cached
        if not hasattr(self, "_total_tokens"):
            self._total_tokens = sum(np.load(p, mmap_mode="r").shape[0] for p in self.token_paths)
        return self._total_tokens


# ----------------------------------------------------------------------------
# Muon optimizer (Newton-Schulz orthogonalized momentum SGD), single-GPU.
# Falls back to this if torch.optim.Muon isn't available in this torch build.
# ----------------------------------------------------------------------------
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B_ = b * A + c * A @ A
        X = a * X + B_ @ X
    if transposed:
        X = X.T
    return X


class MuonFallback(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).to(p.dtype)
                if wd:
                    p.mul_(1 - lr * wd)
                scale = max(1.0, g.size(0) / g.size(1)) ** 0.5
                p.add_(g, alpha=-lr * scale)


def get_muon_cls():
    if hasattr(torch.optim, "Muon"):
        return torch.optim.Muon
    print("torch.optim.Muon not found in this torch build -- using the built-in MuonFallback implementation.")
    return MuonFallback


# ----------------------------------------------------------------------------
# Optional speed patch: plain causal SDPA (no explicit mask tensor) for the
# common no-padding, no-KV-cache training case, so SDPA can use its fastest
# (flash-attention) backend instead of the generic masked path.
# ----------------------------------------------------------------------------
def patch_fast_attention(model):
    attn_cls = type(model.transformer.h[0].attn)
    if getattr(attn_cls, "_fast_patched", False):
        return
    orig_forward = attn_cls.forward

    def fast_forward(self, x, attention_mask=None, past_key_values=None, use_cache=False, cache_position=None):
        if attention_mask is not None or past_key_values is not None:
            return orig_forward(self, x, attention_mask, past_key_values, use_cache, cache_position)
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

    attn_cls.forward = fast_forward
    attn_cls._fast_patched = True
    print(f"Patched {attn_cls.__name__}.forward for fast causal SDPA during training.")


# ----------------------------------------------------------------------------
# LR schedule: warmup then cosine decay
# ----------------------------------------------------------------------------
def get_lr(step, max_lr, min_lr, warmup_steps, max_steps):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ----------------------------------------------------------------------------
# Qualitative sampling during training
# ----------------------------------------------------------------------------
@torch.no_grad()
def sample(model, enc, device, device_type, prompts, max_new_tokens=64, top_k=50):
    model.eval()
    rng = torch.Generator(device=device).manual_seed(42)
    for prompt in prompts:
        ids = [USER_TOKEN_ID] + enc.encode_ordinary(prompt) + [EOT_TOKEN_ID, ASSISTANT_TOKEN_ID]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = model(input_ids=x).logits
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1)
            nxt = torch.gather(topk_idx, -1, torch.multinomial(topk_probs, 1, generator=rng))
            x = torch.cat([x, nxt], dim=1)
            if nxt.item() == EOT_TOKEN_ID:
                break
        text = enc.decode(x[0].tolist())
        print(f"\n--- prompt: {prompt!r}\n{text}")
    model.train()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/ultrachat")
    ap.add_argument("--model_id", default="Erbium08/Gruyere-1.1")
    ap.add_argument("--out_dir", default="gruyere-1.2")

    ap.add_argument("--T", type=int, default=1024, help="sequence length (== model block_size)")
    ap.add_argument("--B", type=int, default=32, help="micro-batch size; lower this first if you OOM")
    ap.add_argument("--total_batch_size", type=int, default=262144,
                     help="tokens per optimizer step (grad-accum handles the rest)")

    ap.add_argument("--epochs", type=float, default=3.0, help="passes over the tokenized train_sft stream")
    ap.add_argument("--warmup_steps", type=int, default=100)

    ap.add_argument("--max_lr_adamw", type=float, default=1.5e-4,
                     help="peak AdamW LR. Kept well below typical from-scratch pretraining LRs "
                          "(e.g. 3e-4) since this is a continued fine-tune of an already-trained model.")
    ap.add_argument("--min_lr_adamw", type=float, default=1.5e-5)
    ap.add_argument("--max_lr_muon", type=float, default=5e-3,
                     help="peak Muon LR, similarly toned down from a from-scratch value (~1e-2).")
    ap.add_argument("--min_lr_muon", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--val_interval", type=int, default=250)
    ap.add_argument("--val_steps", type=int, default=40)
    ap.add_argument("--save_interval", type=int, default=500)
    ap.add_argument("--sample_interval", type=int, default=250)

    ap.add_argument("--compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--fast_attn", action="store_true", default=True)
    ap.add_argument("--no-fast_attn", dest="fast_attn", action="store_false")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device == "cuda" else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    print(f"device: {device}")

    assert args.total_batch_size % (args.B * args.T) == 0, \
        "total_batch_size must be divisible by B*T"
    grad_accum_steps = args.total_batch_size // (args.B * args.T)
    print(f"micro-batch tokens: {args.B * args.T:,} | grad_accum_steps: {grad_accum_steps} "
          f"| tokens/optimizer-step: {args.total_batch_size:,}")

    # ---- data ----
    train_loader = ShardedTokenLoader(args.data_dir, "train", args.B, args.T)
    val_loader = ShardedTokenLoader(args.data_dir, "val", args.B, args.T)
    total_train_tokens = train_loader.total_tokens
    max_steps = max(1, int(args.epochs * total_train_tokens) // args.total_batch_size)
    print(f"train stream: {total_train_tokens:,} tokens -> {args.epochs} epoch(s) => max_steps={max_steps:,}")

    # ---- model ----
    print(f"loading {args.model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True, dtype=torch.float32)
    assert model.config.vocab_size > ASSISTANT_TOKEN_ID, \
        "model vocab too small to hold the new role tokens -- check model_id"
    model.to(device)
    model.train()
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    if args.fast_attn:
        patch_fast_attention(model)

    raw_model = model
    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile failed ({e}); continuing uncompiled")

    # ---- optimizers: Muon for 2D matmul weights, AdamW for the rest ----
    muon_names = {"attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"}
    muon_params, adamw_params = [], []
    for name, p in raw_model.named_parameters():
        if any(name.endswith(suffix) for suffix in muon_names):
            muon_params.append(p)
        else:
            adamw_params.append(p)
    print(f"Muon params: {sum(p.numel() for p in muon_params):,} | AdamW params: {sum(p.numel() for p in adamw_params):,}")

    MuonCls = get_muon_cls()
    muon_optimizer = MuonCls(muon_params, lr=args.max_lr_muon, momentum=0.95, weight_decay=args.weight_decay)
    adamw_optimizer = torch.optim.AdamW(
        adamw_params, lr=args.max_lr_adamw, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=args.weight_decay, fused=(device_type == "cuda"),
    )

    enc = build_encoding()
    sample_prompts = [
        "What is the capital of France?",
        "Can you explain how photosynthesis works?",
        "Write a short poem about the ocean.",
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "log.txt")
    with open(log_path, "w"):
        pass

    for step in range(max_steps):
        t0 = time.time()
        last_step = step == max_steps - 1

        if step % args.val_interval == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(args.val_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        out = model(input_ids=x, labels=y)
                    val_loss += out.loss.detach() / args.val_steps
            print(f"step {step:6d} | val_loss {val_loss.item():.4f}")
            with open(log_path, "a") as f:
                f.write(f"{step} val {val_loss.item():.4f}\n")
            model.train()

        if step > 0 and (step % args.save_interval == 0 or last_step):
            ckpt_dir = os.path.join(args.out_dir, f"step_{step:06d}")
            raw_model.save_pretrained(ckpt_dir)
            print(f"saved checkpoint to {ckpt_dir}")

        if step % args.sample_interval == 0 or last_step:
            sample(model, enc, device, device_type, sample_prompts)

        adamw_optimizer.zero_grad(set_to_none=True)
        muon_optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                out = model(input_ids=x, labels=y)
            loss = out.loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)

        lr_adamw = get_lr(step, args.max_lr_adamw, args.min_lr_adamw, args.warmup_steps, max_steps)
        lr_muon = get_lr(step, args.max_lr_muon, args.min_lr_muon, args.warmup_steps, max_steps)
        for g in adamw_optimizer.param_groups:
            g["lr"] = lr_adamw
        for g in muon_optimizer.param_groups:
            g["lr"] = lr_muon

        adamw_optimizer.step()
        muon_optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        tok_per_sec = (args.B * args.T * grad_accum_steps) / dt
        cumulative_tokens = (step + 1) * args.total_batch_size
        print(f"step {step + 1:6d}/{max_steps} | epoch {train_loader.epoch} | "
              f"tokens {cumulative_tokens / 1e6:9.2f}M | loss {loss_accum.item():.4f} | "
              f"lr_adamw {lr_adamw:.2e} | lr_muon {lr_muon:.2e} | norm {norm:.3f} | "
              f"{tok_per_sec:,.0f} tok/s | {dt * 1000:.0f}ms")
        with open(log_path, "a") as f:
            f.write(f"{step + 1} train {loss_accum.item():.6f} tokens={cumulative_tokens} "
                    f"lr_adamw={lr_adamw:.4e} lr_muon={lr_muon:.4e} norm={norm:.4f} tok_s={tok_per_sec:.1f}\n")

    final_dir = os.path.join(args.out_dir, "final")
    raw_model.save_pretrained(final_dir)
    print(f"\nTraining complete. Final model saved to {final_dir}")
    print("Copy tokenizer.json / tokenizer_config.json from the base Gruyere-1.1 repo into that "
          "folder (or extend them with the <|user|>/<|assistant|> tokens) before pushing to the Hub.")


if __name__ == "__main__":
    main()