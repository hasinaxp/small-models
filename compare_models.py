"""Controlled comparison on the ghost corpus: old dense model vs the new
hybrid, same tokenizer, same batches, same step budget, held-out eval.

    python compare_models.py            # ~ a few minutes on a laptop GPU
"""

import os
import sys
import time

import torch

from tokenizer import Tokenizer
import model as new_model
import model_v1_dense as old_model
from optim import build_optimizer

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CORPUS = "dataset/corpus-ghost.txt"
VOCAB = 2048
SEQ = 256
BATCH = 16
STEPS = int(os.environ.get("STEPS", 300))
EVAL_AT = {150, 300, 600}
EVAL_BATCHES = 20
LR = 1e-3
WARMUP = 50
ARCH = dict(n_layer=8, n_head=4, n_dim=256, n_seq=SEQ)

tok = Tokenizer(vocab_size=VOCAB)
tok.train_from_file(CORPUS)
text = open(CORPUS, encoding="utf-8", errors="ignore").read()
ids = torch.tensor(tok.encode(text), dtype=torch.long)
n_val = len(ids) // 10
train_ids, val_ids = ids[:-n_val], ids[-n_val:]
print(f"tokens: train {len(train_ids)}, val {len(val_ids)}")


def batch(src, gen):
    ix = torch.randint(0, len(src) - SEQ - 1, (BATCH,), generator=gen)
    xs = torch.stack([src[i:i + SEQ] for i in ix]).to(DEV)
    ys = torch.stack([src[i + 1:i + SEQ + 1] for i in ix]).to(DEV)
    return xs, ys


@torch.no_grad()
def evaluate(m):
    m.eval()
    gen = torch.Generator().manual_seed(123)
    tot = 0.0
    for _ in range(EVAL_BATCHES):
        xs, ys = batch(val_ids, gen)
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            m(xs, targets=ys, mode="loss")
        tot += m.last_metrics["ce"].item() if hasattr(m, "last_metrics") and m.last_metrics else 0.0
    m.train()
    return tot / EVAL_BATCHES


@torch.no_grad()
def evaluate_ce(m):
    """Plain next-token CE for either model (ignores z-loss / MTP)."""
    m.eval()
    gen = torch.Generator().manual_seed(123)
    tot = 0.0
    for _ in range(EVAL_BATCHES):
        xs, ys = batch(val_ids, gen)
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            logits, _ = m(xs)
        tot += torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), ys.reshape(-1)).item()
    m.train()
    return tot / EVAL_BATCHES


def run(name, m, opt):
    m.to(DEV).train()
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP) * (0.5 * (1 + torch.cos(
            torch.tensor(min(s, STEPS) / STEPS * 3.14159)).item()) * 0.9 + 0.1))
    gen = torch.Generator().manual_seed(1)
    n = m.get_param_count()
    print(f"\n== {name}: {n/1e6:.2f}M params ==")
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    vals = {}
    for step in range(STEPS):
        xs, ys = batch(train_ids, gen)
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            loss = m(xs, targets=ys, mode="loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) in EVAL_AT or step == STEPS - 1:
            vals[step + 1] = evaluate_ce(m)
            print(f"  step {step+1:4d}  train loss {loss.item():.4f}   val CE {vals[step+1]:.4f}")
    dt = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 2**20 if DEV == "cuda" else 0
    print(f"  -> {dt/STEPS*1000:.0f} ms/step   peak mem {mem:.0f} MiB")
    return vals



PLAIN = dict(attention_sink=False, attn_output_gate=False, value_residual=False,
             unet_skips=False, logit_softcap=0.0, mtp_depth=0)
CONFIGS = {
    "old+adamw":        ("old", "adamw", {}),
    "old+muon":         ("old", "muon", {}),
    "S plain":          ("new", "muon", dict(mixer_pattern="S", **PLAIN)),
    "S tricks":         ("new", "muon", dict(mixer_pattern="S", mtp_depth=0)),
    "S tricks+mtp":     ("new", "muon", dict(mixer_pattern="S")),
    "DS tricks":        ("new", "muon", dict(mtp_depth=0)),
    "DS tricks+mtp":    ("new", "muon", dict()),
    "DS plain":         ("new", "muon", dict(**PLAIN)),
    "D plain":          ("new", "muon", dict(mixer_pattern="D", **PLAIN)),
    "DS tricks+mtp adamw": ("new", "adamw", dict()),
    "S +sink":    ("new", "muon", dict(mixer_pattern="S", **{**PLAIN, "attention_sink": True})),
    "S +gate":    ("new", "muon", dict(mixer_pattern="S", **{**PLAIN, "attn_output_gate": True})),
    "S +vres":    ("new", "muon", dict(mixer_pattern="S", **{**PLAIN, "value_residual": True})),
    "S +unet":    ("new", "muon", dict(mixer_pattern="S", **{**PLAIN, "unet_skips": True})),
    "S +softcap": ("new", "muon", dict(mixer_pattern="S", **{**PLAIN, "logit_softcap": 30.0})),
    "S +mtp":     ("new", "muon", dict(mixer_pattern="S", **{**PLAIN, "mtp_depth": 1})),
}

results = {}
which = sys.argv[1:] or list(CONFIGS)
for name in which:
    kind, optname, kw = CONFIGS[name]
    mod = old_model if kind == "old" else new_model
    torch.manual_seed(0)
    m = mod.Transformer(vocab_size=tok.vocab_size, **ARCH, **kw)
    if optname == "adamw":
        opt = torch.optim.AdamW(m.param_groups(0.1), lr=LR, betas=(0.9, 0.95))
    else:
        opt = build_optimizer(m, lr=LR, weight_decay=0.1)
    results[name] = run(name, m, opt)

print("\n==== held-out next-token cross-entropy (lower is better) ====")
steps = sorted({s for v in results.values() for s in v})
print(f"  {'config':28s}" + "".join(f"{'step '+str(s):>12s}" for s in steps))
for k, v in results.items():
    print(f"  {k:28s}" + "".join(f"{v.get(s, float('nan')):12.4f}" for s in steps))
