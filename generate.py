"""Sample from a pretraining checkpoint.

    python generate.py "The history of Rome" --ckpt checkpoints/distill100m/latest.pt
"""
import argparse
import warnings

import torch
from tokenizers import Tokenizer

from model import ModelConfig, Transformer

warnings.filterwarnings("ignore", message=".*head_dim.*")
ap = argparse.ArgumentParser()
ap.add_argument("prompt")
ap.add_argument("--ckpt", default="checkpoints/distill100m/latest.pt")
ap.add_argument("--tokenizer", default="tokenizer/student16k/tokenizer.json")
ap.add_argument("--max_tokens", type=int, default=200)
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--top_p", type=float, default=0.95)
ap.add_argument("--n", type=int, default=1)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
model = Transformer.from_config(ModelConfig(**ckpt["config"]))
model.load_state_dict(ckpt["model"])
model.to(dev).eval()
tok = Tokenizer.from_file(args.tokenizer)
eos = next(i for i in (tok.token_to_id("<|EOS|>"), tok.token_to_id("<|endoftext|>")) if i is not None)

idx = torch.tensor([tok.encode(args.prompt).ids] * args.n, device=dev)
with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
    out = model.generate(idx, max_count=args.max_tokens, temperature=args.temperature,
                         top_p=args.top_p, eos_token_id=eos, repetition_penalty=1.1)
for row in out.tolist():
    if eos in row[idx.size(1):]:
        row = row[:idx.size(1) + row[idx.size(1):].index(eos)]
    print(tok.decode(row), "\n" + "-" * 60)
