"""Train the byte-level BPE tokenizer used for pretraining.

Small vocab on purpose (default 16384): at ~100M params a 49k vocab spends a
quarter of the budget on the embedding table. Trained with the Rust
``tokenizers`` library (the pure-Python ``tokenizer.py`` is far too slow for
billions of tokens), on a sample of the same smollm-corpus mix we pretrain on.
Special tokens match ``tokenizer.py``: <|BOS|>=0, <|EOS|>=1, ...

    python train_tokenizer.py --vocab 16384 --out tokenizer/
"""
import argparse
import os

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

REPO = "HuggingFaceTB/smollm-corpus"
SPECIAL_TOKENS = ["<|BOS|>", "<|EOS|>", "<|PAD|>", "<|UNK|>", "<|USER|>", "<|ASSISTANT|>"]


def sample_texts(source, file_idx, n_docs, raw_dir):
    """First ``n_docs`` documents of one parquet shard."""
    n_files = {"cosmopedia-v2": 104, "fineweb-edu-dedup": 234}[source]
    path = hf_hub_download(REPO, f"{source}/train-{file_idx:05d}-of-{n_files:05d}.parquet",
                           repo_type="dataset", local_dir=raw_dir)
    seen = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=["text"]):
        for t in batch.column(0).to_pylist():
            yield t
            seen += 1
            if seen >= n_docs:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--out", default="tokenizer")
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--docs", default="fineweb-edu-dedup:300000,cosmopedia-v2:150000",
                    help="source:n_docs pairs to sample from (first shard of each)")
    args = ap.parse_args()

    tok = Tokenizer(models.BPE())
    # Digits are split one per token (helps arithmetic in small models), then
    # GPT-2 byte-level splitting; no unknown bytes are possible.
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab, special_tokens=SPECIAL_TOKENS, min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)

    def corpus():
        for pair in args.docs.split(","):
            src, n = pair.split(":")
            yield from sample_texts(src, 0, int(n), args.raw_dir)

    tok.train_from_iterator(corpus(), trainer=trainer)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "tokenizer.json")
    tok.save(path)
    assert tok.token_to_id("<|EOS|>") == 1
    sample = "The mitochondria is the powerhouse of the cell. In 1953, 42 people..."
    enc = tok.encode(sample)
    print(f"saved {path}: vocab {tok.get_vocab_size()}, "
          f"{len(sample) / len(enc.ids):.2f} chars/token on sample")
    print(enc.tokens)


if __name__ == "__main__":
    main()
