"""Download smollm-corpus parquet shards and tokenize them into uint16 .bin files.

Each document is encoded and followed by <|EOS|>; documents are packed back to
back. One .bin per parquet shard, written atomically, so an interrupted run
resumes where it stopped. Parquet files are deleted once tokenized.

    python prepare_data.py --tokens 10e9 --mix fineweb-edu-dedup:0.7,cosmopedia-v2:0.3

Validation tokens come from the *last* shard of each source, training tokens
from the first ones, so the two never overlap.
"""
import argparse
import json
import os
import threading
import queue
import time

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO = "HuggingFaceTB/smollm-corpus"
N_FILES = {"cosmopedia-v2": 104, "fineweb-edu-dedup": 234}


def shard_name(source, i):
    return f"{source}/train-{i:05d}-of-{N_FILES[source]:05d}.parquet"


def tokenize_file(tok, eos, parquet_path, out_path, max_tokens):
    """Tokenize one parquet file into ``out_path`` (stops at ``max_tokens``).
    Returns the number of tokens written."""
    tmp = out_path + ".tmp"
    n = 0
    with open(tmp, "wb") as f:
        for batch in pq.ParquetFile(parquet_path).iter_batches(
                batch_size=8192, columns=["text"]):
            encs = tok.encode_batch_fast(batch.column(0).to_pylist())
            ids = np.fromiter(
                (t for e in encs for t in (*e.ids, eos)), dtype=np.uint16)
            if n + len(ids) > max_tokens:
                ids = ids[:max_tokens - n]
            f.write(ids.tobytes())
            n += len(ids)
            if n >= max_tokens:
                break
    os.replace(tmp, out_path)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--out", default="data/smollm")
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--tokens", type=float, default=10e9, help="training tokens")
    ap.add_argument("--val_tokens", type=float, default=20e6)
    ap.add_argument("--mix", default="fineweb-edu-dedup:0.7,cosmopedia-v2:0.3")
    ap.add_argument("--keep_raw", action="store_true")
    ap.add_argument("--eos_token", default=None,
                    help="document separator (default: <|EOS|> or <|endoftext|>)")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    for name in ([args.eos_token] if args.eos_token else ["<|EOS|>", "<|endoftext|>"]):
        eos = tok.token_to_id(name)
        if eos is not None:
            break
    assert eos is not None, "no EOS token found; pass --eos_token"
    assert tok.get_vocab_size() <= 65536, "uint16 storage needs vocab <= 65536"
    os.makedirs(args.out, exist_ok=True)

    # Work list: (source, file index, split, token budget for this split).
    jobs = []
    for pair in args.mix.split(","):
        src, frac = pair.split(":")
        frac = float(frac)
        jobs.append((src, N_FILES[src] - 1, "val", int(args.val_tokens * frac)))
        jobs.append((src, None, "train", int(args.tokens * frac)))

    def plan():
        for src, idx, split, budget in jobs:
            if split == "val":
                yield src, idx, split, budget
            else:
                for i in range(N_FILES[src] - 1):
                    yield src, i, split, budget

    meta_path = os.path.join(args.out, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"shards": {}}
    meta.update(vocab_size=tok.get_vocab_size(), eos_id=eos, dtype="uint16",
                tokenizer=args.tokenizer, mix=args.mix)

    done_tokens = {}
    for name, info in meta["shards"].items():
        key = (info["source"], info["split"])
        done_tokens[key] = done_tokens.get(key, 0) + info["tokens"]

    budgets = {(s, sp): b for s, _, sp, b in jobs}
    todo = [(src, i, split) for src, i, split, _ in plan()]
    # Shared with the consumer: files queued but not yet tokenized, and the
    # observed tokens per file, so we stop downloading once a budget is met.
    pending, per_file = {}, {}
    lock = threading.Lock()
    q = queue.Queue(maxsize=1)    # downloads run one file ahead of tokenizing

    def downloader():
        for src, i, split in todo:
            key = (src, split)
            out = os.path.join(args.out, f"{split}_{src}_{i:05d}.bin")
            if os.path.exists(out):
                continue
            # Wait while the files already queued are expected to fill the
            # budget (or, before any full file was seen, can't be estimated).
            while True:
                with lock:
                    est = done_tokens.get(key, 0) + pending.get(key, 0) * per_file.get(src, float("inf"))
                    if pending.get(key, 0) == 0 or est < budgets[key]:
                        break
                time.sleep(1)
            with lock:
                if done_tokens.get(key, 0) >= budgets[key]:
                    continue
                pending[key] = pending.get(key, 0) + 1
            path = hf_hub_download(REPO, shard_name(src, i), repo_type="dataset",
                                   local_dir=args.raw_dir)
            q.put((src, i, split, path, out))
        q.put(None)

    threading.Thread(target=downloader, daemon=True).start()

    t0 = time.time()
    total_new = 0
    while True:
        item = q.get()
        if item is None:
            break
        src, i, split, path, out = item
        key = (src, split)
        remaining = budgets[key] - done_tokens.get(key, 0)
        n = tokenize_file(tok, eos, path, out, remaining) if remaining > 0 else 0
        with lock:
            done_tokens[key] = done_tokens.get(key, 0) + n
            pending[key] -= 1
            if n < remaining:           # a truncated file is no estimate
                per_file[src] = max(per_file.get(src, 0), n)
        total_new += n
        if n == 0:
            if not args.keep_raw:
                os.remove(path)
            continue
        meta["shards"][os.path.basename(out)] = {"source": src, "split": split,
                                                 "file": i, "tokens": n}
        json.dump(meta, open(meta_path, "w"), indent=1)
        if not args.keep_raw:
            os.remove(path)
        el = time.time() - t0
        print(f"{split:5s} {src:18s} shard {i:4d}: {n/1e6:7.1f}M tokens "
              f"(source {done_tokens[(src, split)]/1e9:.2f}/{budgets[(src, split)]/1e9:.2f}B)"
              f"  {total_new/el/1e6:.2f}M tok/s overall", flush=True)

    for (src, split), b in budgets.items():
        print(f"{split} {src}: {done_tokens.get((src, split), 0)/1e9:.3f}B / {b/1e9:.3f}B tokens")


if __name__ == "__main__":
    main()
