"""
prepare_data.py

Downloads HuggingFaceH4/ultrachat_200k and tokenizes it into fixed-size
shards ready for the nanoGPT-style DataLoaderLite in train.py.

Each shard is a pair of files:
    shard_{split}_{idx:05d}_tokens.npy   uint16, the packed token stream
    shard_{split}_{idx:05d}_mask.npy     uint8,  1 where that token counts
                                                  toward the loss, else 0

Token / vocab layout
---------------------
Gruyere-1.1 uses a GPT-2 tokenizer padded to vocab_size=50304 (the standard
nanoGPT "round up to a nice number for the GPU" trick). GPT-2 itself only
uses ids 0..50256, so ids 50257..50303 (47 slots) exist in the embedding
table but were never trained on. We commandeer two of them as chat-role
tokens, which is what lets us fine-tune on a conversational dataset
without resizing any embedding matrix:

    <|user|>       = 50257
    <|assistant|>  = 50258
    <|endoftext|>  = 50256   (already in GPT-2, used as end-of-turn)

Each conversation is laid out as, e.g. for a 2-turn chat:

    <|user|>      <user turn 1 text>      <|endoftext|>
    <|assistant|> <assistant turn 1 text> <|endoftext|>
    <|user|>      <user turn 2 text>      <|endoftext|>
    <|assistant|> <assistant turn 2 text> <|endoftext|>

and the loss mask is 1 only on assistant-turn content tokens and the
<|endoftext|> that closes an assistant turn -- i.e. exactly what the model
is being asked to learn to produce. Everything else (role tags, user
content) is context, masked out of the loss.

Conversations are concatenated back-to-back into one long token stream per
split (train_sft -> train, test_sft -> val), matching how the original
gruyere-1.1 DataLoaderLite samples: contiguous slices regardless of
document boundaries. This means a training window can span a shard/
conversation boundary; that's fine and standard for this style of loader.

Usage:
    pip install datasets tiktoken numpy tqdm
    python prepare_data.py --out_dir data/ultrachat --shard_size 50_000_000
"""

import argparse
import multiprocessing as mp
import os
import time

import numpy as np
import tiktoken
from datasets import load_dataset

# ----------------------------------------------------------------------------
# Token ids
# ----------------------------------------------------------------------------
EOT_TOKEN_ID = 50256          # GPT-2's native <|endoftext|>
USER_TOKEN_ID = 50257         # new, lives in a previously-unused pad slot
ASSISTANT_TOKEN_ID = 50258    # new, lives in a previously-unused pad slot
VOCAB_SIZE = 50304            # Gruyere-1.1's padded vocab size

assert ASSISTANT_TOKEN_ID < VOCAB_SIZE, "ran out of padding slots in the vocab"


def build_encoding() -> tiktoken.Encoding:
    """Extend tiktoken's stock 'gpt2' encoding with our two new role tokens.

    This only matters for *decoding* (so sampling/debug printouts can show
    "<|user|>" / "<|assistant|>" instead of erroring on an unknown id).
    Encoding of ordinary text never touches these ids -- we splice the role
    ids in manually as raw integers, so there's no risk of the literal
    string "<|user|>" appearing in a user's message being misinterpreted.
    """
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
# Per-example tokenization (runs in worker processes)
# ----------------------------------------------------------------------------
_ENC = None  # set once per worker process by _init_worker


def _init_worker():
    global _ENC
    _ENC = build_encoding()


def tokenize_example(messages):
    """messages: list of {"role": "user"|"assistant", "content": str}.

    Returns (tokens, mask) as plain python lists of ints (0/1 for mask).
    """
    tokens = []
    mask = []
    for turn in messages:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            role_id = USER_TOKEN_ID
            is_assistant = 0
        elif role == "assistant":
            role_id = ASSISTANT_TOKEN_ID
            is_assistant = 1
        else:
            # ultrachat_200k's messages field only contains user/assistant
            # turns, but skip anything unexpected defensively.
            continue

        content_ids = _ENC.encode_ordinary(content)

        tokens.append(role_id)
        mask.append(0)  # never score predicting the role tag itself

        tokens.extend(content_ids)
        mask.extend([is_assistant] * len(content_ids))

        tokens.append(EOT_TOKEN_ID)
        mask.append(is_assistant)  # learn to *emit* the stop, for assistant turns

    return tokens, mask


# ----------------------------------------------------------------------------
# Shard writer
# ----------------------------------------------------------------------------
class ShardWriter:
    """Accumulates a token/mask stream and flushes fixed-size .npy shards."""

    def __init__(self, out_dir, split, shard_size):
        self.dir = os.path.join(out_dir, split)
        os.makedirs(self.dir, exist_ok=True)
        self.shard_size = shard_size
        self.shard_idx = 0
        self.tok_buf = np.empty(shard_size, dtype=np.uint16)
        self.mask_buf = np.empty(shard_size, dtype=np.uint8)
        self.pos = 0
        self.total_tokens = 0
        self.total_loss_tokens = 0

    def add(self, tokens, mask):
        tokens = np.asarray(tokens, dtype=np.uint16)
        mask = np.asarray(mask, dtype=np.uint8)
        n = len(tokens)
        i = 0
        while i < n:
            space = self.shard_size - self.pos
            take = min(space, n - i)
            self.tok_buf[self.pos:self.pos + take] = tokens[i:i + take]
            self.mask_buf[self.pos:self.pos + take] = mask[i:i + take]
            self.pos += take
            i += take
            if self.pos == self.shard_size:
                self._flush()
        self.total_tokens += n
        self.total_loss_tokens += int(mask.sum())

    def _flush(self):
        if self.pos == 0:
            return
        tok_path = os.path.join(self.dir, f"shard_{self.split_name}_{self.shard_idx:05d}_tokens.npy")
        mask_path = os.path.join(self.dir, f"shard_{self.split_name}_{self.shard_idx:05d}_mask.npy")
        np.save(tok_path, self.tok_buf[:self.pos])
        np.save(mask_path, self.mask_buf[:self.pos])
        print(f"  wrote {tok_path}  ({self.pos:,} tokens)")
        self.shard_idx += 1
        self.pos = 0

    def close(self):
        self._flush()


def process_split(dataset, split_name, out_dir, shard_size, num_workers):
    writer = ShardWriter(out_dir, split_name, shard_size)
    writer.split_name = split_name

    t0 = time.time()
    n_examples = 0
    with mp.Pool(num_workers, initializer=_init_worker) as pool:
        for tokens, mask in pool.imap(tokenize_example, dataset["messages"], chunksize=64):
            if tokens:
                writer.add(tokens, mask)
            n_examples += 1
            if n_examples % 5000 == 0:
                dt = time.time() - t0
                print(f"[{split_name}] {n_examples:,}/{len(dataset):,} conversations | "
                      f"{writer.total_tokens:,} tokens so far | {n_examples / dt:.0f} conv/s")
    writer.close()

    print(f"[{split_name}] done: {writer.total_tokens:,} total tokens, "
          f"{writer.total_loss_tokens:,} of them scored (assistant) tokens "
          f"({100 * writer.total_loss_tokens / max(writer.total_tokens, 1):.1f}%), "
          f"{writer.shard_idx} shard(s)")
    return writer.total_tokens, writer.total_loss_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    ap.add_argument("--out_dir", default="data/ultrachat")
    ap.add_argument("--shard_size", type=int, default=50_000_000,
                     help="tokens per shard (default: 50M, as requested)")
    ap.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--train_split", default="train_sft")
    ap.add_argument("--val_split", default="test_sft")
    ap.add_argument("--limit", type=int, default=None,
                     help="optional cap on number of conversations per split, for a quick smoke test")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.dataset} ...")
    ds_train = load_dataset(args.dataset, split=args.train_split)
    ds_val = load_dataset(args.dataset, split=args.val_split)
    if args.limit:
        ds_train = ds_train.select(range(min(args.limit, len(ds_train))))
        ds_val = ds_val.select(range(min(args.limit, len(ds_val))))

    print(f"train_sft: {len(ds_train):,} conversations | test_sft: {len(ds_val):,} conversations")
    print(f"Tokenizing with {args.num_workers} worker processes, shard_size={args.shard_size:,} tokens")

    train_tokens, train_loss_tokens = process_split(ds_train, "train", args.out_dir, args.shard_size, args.num_workers)
    val_tokens, val_loss_tokens = process_split(ds_val, "val", args.out_dir, args.shard_size, args.num_workers)

    meta = {
        "vocab_size": VOCAB_SIZE,
        "eot_token_id": EOT_TOKEN_ID,
        "user_token_id": USER_TOKEN_ID,
        "assistant_token_id": ASSISTANT_TOKEN_ID,
        "train_tokens": train_tokens,
        "train_loss_tokens": train_loss_tokens,
        "val_tokens": val_tokens,
        "val_loss_tokens": val_loss_tokens,
    }
    import json
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n== summary ==")
    for k, v in meta.items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
    print(f"\nWrote shards + meta.json to {args.out_dir}")
    print("You can now point train.py --data_dir at this directory.")


if __name__ == "__main__":
    main()