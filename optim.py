"""Muon (for the 2-D weights inside the blocks) + AdamW (for everything else).

Muon (Jordan et al. 2024; used at scale by Kimi K2) replaces each matrix's
momentum-SGD update with its nearest semi-orthogonal matrix, computed by a
5-step Newton-Schulz iteration in bf16. On small language models it reaches a
given loss in roughly 1.35-2x fewer tokens than AdamW, at negligible cost.

Embeddings, the output head, norm gains, the DeltaNet decay parameters, the
short-conv kernels and every other non-matrix parameter go to AdamW.

The Muon learning rate is rescaled per matrix as ``lr * 0.2 * sqrt(max(m, n))``
(the Moonlight recipe), which makes the update RMS match AdamW's so that one
learning rate and one weight-decay value can be shared by both halves. Use it
like any optimizer::

    opt = build_optimizer(model, lr=3e-4, weight_decay=0.1)
"""

import math

import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalise G (approximately): returns U V^T for G = U S V^T. The
    quintic coefficients are tuned for speed, not for exact singular values,
    which Muon does not need."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class MuonAdamW(Optimizer):
    """Param groups carry ``use_muon``; Muon groups take ``momentum`` and
    ``nesterov``; AdamW groups take ``betas`` and ``eps``."""

    def __init__(self, param_groups, lr=3e-4, weight_decay=0.1,
                 momentum=0.95, nesterov=True, ns_steps=5,
                 betas=(0.9, 0.95), eps=1e-8, muon_lr_scale=0.2):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        nesterov=nesterov, ns_steps=ns_steps, betas=betas,
                        eps=eps, muon_lr_scale=muon_lr_scale, use_muon=False)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            if group["use_muon"]:
                # Same-shaped matrices are stacked and orthogonalised in one
                # batched Newton-Schulz: far fewer kernel launches.
                by_shape = {}
                for p in group["params"]:
                    if p.grad is not None:
                        by_shape.setdefault(tuple(p.shape), []).append(p)
                for shape, ps in by_shape.items():
                    upds = []
                    for p in ps:
                        g = p.grad.flatten(1) if p.grad.ndim > 2 else p.grad
                        state = self.state[p]
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(g)
                        buf = state["momentum_buffer"]
                        buf.lerp_(g, 1 - group["momentum"])
                        upds.append(g.lerp(buf, group["momentum"])
                                    if group["nesterov"] else buf)
                    stacked = zeropower_via_newtonschulz5(
                        torch.stack(upds), group["ns_steps"])
                    m, n = stacked.shape[-2:]
                    scale = group["muon_lr_scale"] * math.sqrt(max(m, n))
                    for p, upd in zip(ps, stacked.unbind(0)):
                        if wd:
                            p.mul_(1 - lr * wd)
                        p.add_(upd.view_as(p).to(p.dtype), alpha=-lr * scale)
            else:
                b1, b2 = group["betas"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad.float()
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(g)
                        state["exp_avg_sq"] = torch.zeros_like(g)
                    state["step"] += 1
                    t = state["step"]
                    m, v = state["exp_avg"], state["exp_avg_sq"]
                    m.lerp_(g, 1 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)
                    m_hat = m / (1 - b1 ** t)
                    v_hat = v / (1 - b2 ** t)
                    upd = m_hat / (v_hat.sqrt() + group["eps"])
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.add_(upd.to(p.dtype), alpha=-lr)
        return loss


def split_params(model):
    """(muon_params, adamw_decay, adamw_no_decay) for a ``model.Transformer``.

    Muon takes every 2-D weight inside ``blocks`` and ``mtp``. Conv1d kernels
    (3-D, depthwise) and everything outside the blocks go to AdamW."""
    muon, decay, no_decay = [], [], []
    block_prefixes = ("blocks.", "mtp.")
    linear_weights = {
        id(mod.weight) for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear) and name.startswith(block_prefixes)
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in linear_weights:
            muon.append(p)
        elif p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return muon, decay, no_decay


def build_optimizer(model, lr=3e-4, weight_decay=0.1, muon_lr=None,
                    momentum=0.95, betas=(0.9, 0.95), eps=1e-8):
    """One optimizer for the whole model. ``muon_lr`` defaults to ``lr``
    (the per-matrix rescaling already matches the two update scales)."""
    muon, decay, no_decay = split_params(model)
    groups = [
        {"params": muon, "use_muon": True, "lr": muon_lr or lr,
         "weight_decay": weight_decay, "momentum": momentum},
        {"params": decay, "use_muon": False, "lr": lr,
         "weight_decay": weight_decay, "betas": betas, "eps": eps},
        {"params": no_decay, "use_muon": False, "lr": lr,
         "weight_decay": 0.0, "betas": betas, "eps": eps},
    ]
    groups = [g for g in groups if g["params"]]
    return MuonAdamW(groups, lr=lr, weight_decay=weight_decay,
                     momentum=momentum, betas=betas, eps=eps)
