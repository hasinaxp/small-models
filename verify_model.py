"""Numerical checks for model.py. Run: python verify_model.py

1. chunked gated delta rule == token-by-token recurrence (with/without state)
2. cached, chunk-prefilled, token-by-token decoding == one full forward
3. document mask -> DeltaNet reset agrees with running documents separately
4. loss / backward / optimizer step run for every mixer kind, with and
   without activation checkpointing, MTP and looped blocks
"""

import sys
import torch

from model import (Transformer, chunk_gated_delta_rule,
                   recurrent_gated_delta_rule, build_document_mask)
from optim import build_optimizer

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ok = True


def check(name, a, b, tol):
    global ok
    err = (a.float() - b.float()).abs().max().item()
    flag = "ok " if err < tol else "FAIL"
    if err >= tol:
        ok = False
    print(f"[{flag}] {name}: max abs err {err:.2e} (tol {tol:.0e})")


# 1. delta rule ---------------------------------------------------------------
B, H, L, D = 2, 3, 150, 16
q = torch.nn.functional.normalize(torch.randn(B, H, L, D, device=DEV, dtype=torch.float64), dim=-1)
k = torch.nn.functional.normalize(torch.randn(B, H, L, D, device=DEV, dtype=torch.float64), dim=-1)
v = torch.randn(B, H, L, D, device=DEV, dtype=torch.float64)
g = -torch.rand(B, H, L, device=DEV, dtype=torch.float64) * 0.5
beta = torch.rand(B, H, L, device=DEV, dtype=torch.float64)
S0 = torch.randn(B, H, D, D, device=DEV, dtype=torch.float64) * 0.1
o_r, S_r = recurrent_gated_delta_rule(q, k, v, g, beta)
o_c, S_c = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=32)
check("delta rule out (zero init state)", o_c, o_r, 1e-4)
check("delta rule final state", S_c, S_r, 1e-4)
o_r, S_r = recurrent_gated_delta_rule(q, k, v, g, beta, S0)
o_c, S_c = chunk_gated_delta_rule(q, k, v, g, beta, S0, chunk_size=64)
check("delta rule out (given state)", o_c, o_r, 1e-4)
check("delta rule state (given state)", S_c, S_r, 1e-4)

# 2. cached decode == full forward ---------------------------------------------
for pattern, win in (("DS", 8), ("DSG", 8), ("S", 8), ("D", 8), ("S", 64), ("DS", 64)):
    m = Transformer(vocab_size=97, n_layer=4, n_head=4, n_dim=64, n_seq=64,
                    mixer_pattern=pattern, window_size=win, gdn_chunk_size=16,
                    repeat_start=1, repeat_end=3, repeat_times=2).to(DEV).eval()
    x = torch.randint(0, 97, (2, 40), device=DEV)
    with torch.no_grad():
        full, _ = m(x)
        cache = m.make_kv_cache(2, 40)
        outs = []
        for i, j in ((0, 5), (5, 10), (10, 13)):   # chunked prefill 5,5,3
            h = m.forward_hidden(x[:, i:j], kv_cache=cache)
            outs.append(m.logit_proj(h))
        for t in range(13, 40):             # token by token
            h = m.forward_hidden(x[:, t:t + 1], kv_cache=cache)
            outs.append(m.logit_proj(h))
        inc = torch.tanh(torch.cat(outs, 1) / 30) * 30
    check(f"cached decode == full forward [{pattern}, window {win}]", inc, full, 1e-4)
    with torch.no_grad():
        gen = m.generate(x[:, :5], max_count=20, temperature=0.0)
    print(f"      generate ok, shape {tuple(gen.shape)}")

# 3. document reset ---------------------------------------------------------------
m = Transformer(vocab_size=50, n_layer=2, n_head=2, n_dim=32, n_seq=64,
                mixer_pattern="DS", window_size=64, gdn_chunk_size=8,
                unet_skips=False).to(DEV).eval()
EOS = 1
a = torch.randint(2, 50, (1, 20), device=DEV)
b = torch.randint(2, 50, (1, 20), device=DEV)
packed = torch.cat((a, torch.tensor([[EOS]], device=DEV), b), 1)
mask = build_document_mask(packed, EOS)
with torch.no_grad():
    lp, _ = m(packed, attn_mask=mask)
    lb, _ = m(b)
check("packed-with-doc-mask == separate doc (DeltaNet reset + attention mask)",
      lp[:, 21:], lb, 1e-4)

# 4. training paths ---------------------------------------------------------------
for kw in (dict(mixer_pattern="DS"), dict(mixer_pattern="DSG", activation_checkpointing=True),
           dict(mixer_pattern="DDS", mtp_depth=0, unet_skips=False, value_residual=False,
                attention_sink=False, attn_output_gate=False)):
    m = Transformer(vocab_size=97, n_layer=3, n_head=4, n_dim=64, n_seq=32,
                    window_size=8, gdn_chunk_size=8, loss_chunk_size=8, **kw).to(DEV).train()
    opt = build_optimizer(m, lr=1e-3)
    x = torch.randint(0, 97, (2, 32), device=DEV)
    y = torch.randint(0, 97, (2, 32), device=DEV)
    y[0, :5] = -100
    with torch.autocast(DEV, dtype=torch.bfloat16, enabled=(DEV == "cuda")):
        loss = m(x, targets=y, mode="loss")
    loss.backward()
    grads = [n for n, p in m.named_parameters() if p.grad is None]
    opt.step()
    opt.zero_grad()
    print(f"[{'ok ' if not grads and torch.isfinite(loss) else 'FAIL'}] train step {kw}: "
          f"loss {loss.item():.3f} metrics={ {k: round(v.item(), 3) for k, v in m.last_metrics.items()} }"
          + (f"  NO GRAD: {grads}" if grads else ""))
    if grads or not torch.isfinite(loss):
        ok = False

print("\nALL OK" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
