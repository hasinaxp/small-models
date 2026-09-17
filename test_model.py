"""Quick smoke test: train tokenizer + a tiny hybrid model on the ghost corpus.

    python test_model.py            # STEPS=500 python test_model.py for a short run
"""

import logging
import math
import os
import time

import torch
import torch._dynamo

torch._dynamo.config.suppress_errors = True  # no triton on this box -> fall back to eager
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

from tokenizer import Tokenizer
from model import Transformer
from optim import build_optimizer

CORPUS = "dataset/corpus-ghost.txt"
VOCAB_SIZE = 1000
SEQ_LEN = 128
BATCH_SIZE = 8
STEPS = int(os.environ.get("STEPS", 10000))
LR = 1e-3
WARMUP = 100
COMPILE = False           # torch.compile needs triton; leave off on plain Windows installs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- tokenizer ---
tok = Tokenizer(vocab_size=VOCAB_SIZE)
tok.train_from_file(CORPUS)

with open(CORPUS, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()
ids = torch.tensor(tok.encode(text), dtype=torch.long)
print(f"corpus: {len(ids)} tokens, vocab_size={tok.vocab_size}")

# --- model ---
# Weights stay fp32 (the optimizer needs the precision); matmuls run in bf16
# under autocast. Casting the whole model to bf16 quietly loses small updates.
model = Transformer(
    vocab_size=tok.vocab_size,
    n_layer=4,
    n_head=4,
    n_dim=128,
    n_seq=SEQ_LEN,
    mixer_pattern="DS",      # DeltaNet memory + sliding-window attention
    window_size=64,          # attention sees 64 tokens; memory carries the rest
).to(DEVICE)
print(f"model: {model.get_param_count()/1e6:.2f}M params, layers {''.join(model.kinds)}")
train_model = torch.compile(model) if COMPILE else model

opt = build_optimizer(model, lr=LR, weight_decay=0.1)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / WARMUP)
    * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS))))


def get_batch():
    ix = torch.randint(0, len(ids) - SEQ_LEN - 1, (BATCH_SIZE,))
    xs = torch.stack([ids[i:i + SEQ_LEN] for i in ix]).to(DEVICE)
    ys = torch.stack([ids[i + 1:i + SEQ_LEN + 1] for i in ix]).to(DEVICE)
    return xs, ys


model.train()
t0 = time.time()
for step in range(STEPS):
    xs, ys = get_batch()
    with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        loss = train_model(xs, targets=ys, mode="loss")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 200 == 0:
        parts = " ".join(f"{k} {v.item():.3f}" for k, v in model.last_metrics.items())
        print(f"step {step}: loss {loss.item():.4f}  ({parts})  "
              f"{(time.time() - t0) / (step + 1) * 1000:.0f} ms/step")

# --- sample ---
model.eval()
prompt = torch.tensor([tok.encode("There was once a house")], device=DEVICE)
out = model.generate(prompt, max_count=80, temperature=0.8, top_k=50,
                     valid_vocab_size=tok.vocab_size)
print("\nsample:", repr(tok.decode(out[0].tolist())))
print("done.")
