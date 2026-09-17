"""Length / memory evaluation: train with an attention window shorter than the
training sequence, then score the SAME held-out tokens under different amounts
of preceding context. Context beyond the window can only help through the
DeltaNet memory (or global attention layers).

    python longctx_eval.py                       # ghost corpus, defaults
    CORPUS=dataset/tinystories-valid.txt VOCAB=4096 SEQ=512 WINDOW=128 STEPS=1000 \
        python longctx_eval.py "S" "DS" "old"

Configs (argv): old, S, DS, SD, SSSD, D, DS-nomtp, G.
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
CORPUS = os.environ.get("CORPUS", "dataset/corpus-ghost.txt")
VOCAB = int(os.environ.get("VOCAB", 2048))
SEQ = int(os.environ.get("SEQ", 256))
BATCH = int(os.environ.get("BATCH", 16))
STEPS = int(os.environ.get("STEPS", 300))
WINDOW = int(os.environ.get("WINDOW", 64))
LR, WARMUP = float(os.environ.get("LR", 1e-3)), 50
EVAL_LENS = tuple(int(x) for x in os.environ.get("EVAL_LENS", "128,256,512,1024,2048").split(","))
MAX_LEN = max(EVAL_LENS)
SCORE = 128
ARCH = dict(n_layer=int(os.environ.get("LAYERS", 8)), n_head=4,
            n_dim=int(os.environ.get("DIM", 256)), n_seq=max(SEQ, MAX_LEN))

# -- data (token ids cached next to the corpus) ------------------------------
cache = f"{CORPUS}.v{VOCAB}.pt"
if os.path.exists(cache):
    ids, tok_vocab = torch.load(cache)
else:
    tok = Tokenizer(vocab_size=VOCAB)
    tok.train_from_file(CORPUS)
    text = open(CORPUS, encoding="utf-8", errors="ignore").read()
    text = text.replace("<|endoftext|>", "<|EOS|>")
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    tok_vocab = tok.vocab_size
    torch.save((ids, tok_vocab), cache)
n_val = len(ids) // 10
train_ids, val_ids = ids[:-n_val], ids[-n_val:]
print(f"{CORPUS}: train {len(train_ids)} tokens, val {len(val_ids)} tokens, "
      f"{STEPS} steps x {BATCH}x{SEQ} = {STEPS*BATCH*SEQ/1e6:.1f}M tokens "
      f"({STEPS*BATCH*SEQ/len(train_ids):.1f} epochs)")

_starts = list(range(0, len(val_ids) - MAX_LEN - 1, 512))
EVAL_X = torch.stack([val_ids[i:i + MAX_LEN] for i in _starts])
EVAL_Y = torch.stack([val_ids[i + 1:i + MAX_LEN + 1] for i in _starts])


def batch(src, gen):
    ix = torch.randint(0, len(src) - SEQ - 1, (BATCH,), generator=gen)
    xs = torch.stack([src[i:i + SEQ] for i in ix]).to(DEV)
    ys = torch.stack([src[i + 1:i + SEQ + 1] for i in ix]).to(DEV)
    return xs, ys


@torch.no_grad()
def evaluate_ce(m, L, bs=16):
    """CE on the final SCORE tokens of each eval window given the last L tokens
    as input. Identical scored tokens for every L."""
    m.eval()
    tot, cnt = 0.0, 0
    for i in range(0, EVAL_X.size(0), bs):
        xs = EVAL_X[i:i + bs, -L:].to(DEV)
        ys = EVAL_Y[i:i + bs, -SCORE:].to(DEV)
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            logits, _ = m(xs)
        logits = logits[:, -SCORE:]
        tot += torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), ys.reshape(-1),
            reduction="sum").item()
        cnt += ys.numel()
    m.train()
    return tot / cnt


def train(name, m):
    m.to(DEV).train()
    opt = build_optimizer(m, lr=LR, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP) * (0.5 * (1 + torch.cos(
            torch.tensor(min(s, STEPS) / STEPS * 3.14159)).item()) * 0.9 + 0.1))
    gen = torch.Generator().manual_seed(1)
    print(f"\n== {name}: {m.get_param_count()/1e6:.2f}M params ==")
    t0 = time.time()
    for step in range(STEPS):
        xs, ys = batch(train_ids, gen)
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            loss = m(xs, targets=ys, mode="loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 200 == 0:
            print(f"  step {step+1:5d}  train loss {loss.item():.4f}")
    print(f"  {(time.time()-t0)/STEPS*1000:.0f} ms/step, final train loss {loss.item():.3f}")
    row = {}
    for L in EVAL_LENS:
        try:
            row[L] = evaluate_ce(m, L)
        except Exception as e:                      # old model beyond n_seq etc.
            row[L] = float("nan")
        print(f"  context {L:5d}: CE(same {SCORE} tokens) {row[L]:.4f}")
    return row


def new(**kw):
    return new_model.Transformer(vocab_size=tok_vocab, window_size=WINDOW, **ARCH, **kw)


CONFIGS = {
    "old":      lambda: old_model.Transformer(vocab_size=tok_vocab, **ARCH),
    "S":        lambda: new(mixer_pattern="S"),
    "DS":       lambda: new(mixer_pattern="DS"),
    "SD":       lambda: new(mixer_pattern="SD"),
    "SSSD":     lambda: new(mixer_pattern="SSSD"),
    "D":        lambda: new(mixer_pattern="D"),
    "DS-nomtp": lambda: new(mixer_pattern="DS", mtp_depth=0),
    "S-nomtp":  lambda: new(mixer_pattern="S", mtp_depth=0),
    "G":        lambda: new(mixer_pattern="G"),
}
which = sys.argv[1:] or ["old", "S", "DS"]
results = {}
for name in which:
    torch.manual_seed(0)
    results[name] = train(f"{name} (window {WINDOW})" if name != "old" else "old full attention", CONFIGS[name]())

print(f"\n==== held-out CE on the same {SCORE} tokens per window, varying context "
      f"(lower is better; {EVAL_X.size(0)} windows) ====")
print(f"  {'config':14s}" + "".join(f"{L:>10d}" for L in EVAL_LENS))
for k, v in results.items():
    print(f"  {k:14s}" + "".join(f"{v[L]:10.4f}" for L in EVAL_LENS))
