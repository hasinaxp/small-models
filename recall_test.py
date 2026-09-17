"""Multi-query associative recall (MQAR): does the DeltaNet memory work?

Each sequence: N (key, value) pairs, then a run of filler, then the same keys
queried in random order; the model must emit the matching value. Loss is on
the answer positions only. The attention window is set far shorter than the
distance from a pair to its query, so a window-only model *cannot* solve it
and must guess, while a model with a recurrent memory can.

    python recall_test.py
"""
import os, sys, time
import torch
import torch.nn.functional as F
from model import Transformer, IGNORE_INDEX
from optim import build_optimizer

DEV = "cuda"
N_KEYS, N_VALS = 32, 32          # vocab: keys 0..31, values 32..63, filler 64, query-mark 65
FILLER, QMARK = 64, 65
VOCAB = 66
PAIRS = 12
FILL = 64
STEPS = int(os.environ.get("STEPS", 600))
WINDOW = 16
BATCH = 64


def make_batch(bs, gen):
    keys = torch.stack([torch.randperm(N_KEYS, generator=gen)[:PAIRS] for _ in range(bs)])
    vals = torch.randint(0, N_VALS, (bs, PAIRS), generator=gen) + N_KEYS
    order = torch.stack([torch.randperm(PAIRS, generator=gen) for _ in range(bs)])
    qk = keys.gather(1, order)
    qv = vals.gather(1, order)
    pairs = torch.stack((keys, vals), -1).reshape(bs, -1)            # k v k v ...
    fill = torch.full((bs, FILL), FILLER)
    queries = torch.stack((torch.full_like(qk, QMARK), qk, qv), -1).reshape(bs, -1)  # Q k v
    x = torch.cat((pairs, fill, queries), 1)
    y = torch.full_like(x, IGNORE_INDEX)
    # predict v at the position of its key (target index = input index of key)
    qstart = pairs.size(1) + FILL
    for j in range(PAIRS):
        y[:, qstart + 3 * j + 1] = qv[:, j]
    return x[:, :-1].to(DEV), y[:, 1:].to(DEV)


def accuracy(m, gen, n=8):
    m.eval()
    ok = tot = 0
    with torch.no_grad():
        for _ in range(n):
            x, y = make_batch(BATCH, gen)
            logits, _ = m(x)
            mask = y != IGNORE_INDEX
            ok += (logits.argmax(-1)[mask] == y[mask]).sum().item()
            tot += mask.sum().item()
    m.train()
    return ok / tot


def run(name, **kw):
    torch.manual_seed(0)
    m = Transformer(vocab_size=VOCAB, n_layer=4, n_head=2, n_dim=128, n_seq=512,
                    window_size=WINDOW, mtp_depth=0, unet_skips=False, **kw).to(DEV)
    opt = build_optimizer(m, lr=2e-3, weight_decay=0.0)
    gen = torch.Generator().manual_seed(1)
    t0 = time.time()
    for step in range(STEPS):
        x, y = make_batch(BATCH, gen)
        loss = m(x, targets=y, mode="loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    acc = accuracy(m, torch.Generator().manual_seed(99))
    print(f"  {name:34s} final loss {loss.item():.3f}   recall accuracy {acc*100:5.1f}%   ({(time.time()-t0)/STEPS*1000:.0f} ms/step)")
    return acc


print(f"MQAR: {PAIRS} pairs, {FILL} filler tokens, attention window {WINDOW} "
      f"(pair->query distance ~{2*PAIRS+FILL}..{4*PAIRS+FILL}); chance = {100/N_VALS:.1f}%")
run("S  (window-only attention)", mixer_pattern="S")
run("DS (DeltaNet + window attn)", mixer_pattern="DS")
run("D  (DeltaNet only)", mixer_pattern="D")
run("G  (full attention, reference)", mixer_pattern="G")
