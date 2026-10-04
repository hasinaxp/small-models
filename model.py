"""A hybrid Gated-DeltaNet / sliding-window-attention language model for tiny LMs.

Every layer is ``mixer -> SwiGLU`` with pre-RMSNorm. The mixer is one of:

* ``D`` -- **Gated DeltaNet** (Yang et al. 2024; the recurrent layer in
  Qwen3-Next / Kimi Linear). Each head keeps a fixed-size matrix state
  ``S in R^{d_k x d_v}`` that is updated *every token* by a gated delta rule::

      S_t = a_t * S_{t-1} + b_t * k_t (v_t - a_t S_{t-1}^T k_t)^T ,   o_t = S_t^T q_t

  ``a_t`` (data-dependent forgetting) and ``b_t`` (write strength) are
  predicted from the input. This is the model's "active memory": constant size
  regardless of context, so it scales to arbitrary length. Training uses the
  chunkwise-parallel WY form (pure PyTorch, no Triton needed); decoding uses
  the one-token recurrence.
* ``S`` -- **sliding-window attention** (GQA + RoPE + QK-norm) over the last
  ``window_size`` tokens, with a learned *attention sink* key so heads can
  attend to nothing, and an optional sigmoid *output gate* (Qwen3-Next "gated
  attention"; off by default, it cost ~0.02 nats at 7M params). KV cache is
  bounded by the window, so inference memory is constant and context length
  is unbounded.
* ``G`` -- full causal attention (same module, no window). Optional; if the
  pattern has none, the model has no maximum context length at all.

The default pattern ``"DS"`` alternates memory and local attention (the Samba
layout), which is the layout best known to extrapolate far beyond the training
length: attention handles precise local copying, the recurrent state carries
long-range context, and RoPE never sees an out-of-distribution distance.

Training tricks (all cheap, all validated on ~100M-scale models):

* **Multi-token prediction** (``mtp_depth > 0``): a light extra block predicts
  token t+2 from ``[h_t ; emb(x_{t+1})]`` (DeepSeek-V3 form). Adds an auxiliary
  loss that densifies the training signal and improves planning.
* **Logit soft-capping** ``cap * tanh(logits / cap)`` (Gemma 2) plus **z-loss**.
* **Value residual** (ResFormer): later attention layers mix in the first
  attention layer's values. **U-Net skips** and learnable **residual lambdas**
  (modded-nanogpt). **Chunked loss** so the fp32 logit tensor never fully
  materialises.
* **Looped middle blocks** via ``layer_schedule`` (extra depth, no extra params).

Width knobs (all default to the classic tied-to-``n_dim`` sizes):

* ``mixer_dim`` -- inner width of attention / DeltaNet (heads * head_dim),
  decoupled from the residual width ``n_dim``.
* ``ffn_hidden`` -- SwiGLU hidden width (default ~8/3 * n_dim).
* ``emb_dim`` -- factorised embedding (ALBERT): a ``vocab x emb_dim`` table
  plus ``emb_dim -> n_dim`` up-projection; the tied output head goes
  ``n_dim -> emb_dim`` before the table.

With these set, per-layer params grow linearly in ``n_dim`` instead of
quadratically, so the residual stream can be widened cheaply.

See ``optim.py`` for the matching Muon + AdamW optimizer.
"""

import math
import warnings
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

DEFAULT_SEQ_LEN = 2048
DEFAULT_EMBEDDING_DIM = 896
DEFAULT_NUM_HEADS = 14
DEFAULT_NUM_LAYERS = 16
DEFAULT_NUM_EXPERTS = 1
DEFAULT_TOP_K = 1

IGNORE_INDEX = -100
MIXER_KINDS = ("D", "S", "G")

__all__ = [
    "Transformer", "ModelConfig", "RMSNorm", "GatedRMSNorm", "Attention",
    "AttentionGQA", "GatedDeltaNet", "SwiGLU", "FFN", "Block", "MTPHead",
    "HybridCache", "KVCache", "AttentionCache", "GDNCache", "IGNORE_INDEX",
    "swiglu_hidden_dim", "build_document_mask", "precompute_rope", "apply_rope",
    "default_n_kv_head", "layer_schedule", "expand_pattern",
    "chunk_gated_delta_rule", "recurrent_gated_delta_rule",
]


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int
    n_layer: int = 8
    n_head: int = 4
    n_dim: int = 128
    n_seq: int = 256
    # -- widths (None = derived from n_dim) --
    mixer_dim: Optional[int] = None
    ffn_hidden: Optional[int] = None
    emb_dim: Optional[int] = None
    # -- hybrid layout --
    mixer_pattern: str = "DS"      # tiled over n_layer; D=DeltaNet S=window G=global
    window_size: int = 512
    n_kv_head: Optional[int] = None
    attention_sink: bool = True
    attn_output_gate: bool = False
    value_residual: bool = True
    unet_skips: bool = True
    gdn_conv_kernel: int = 4
    gdn_chunk_size: int = 64
    # -- objective --
    logit_softcap: float = 30.0
    mtp_depth: int = 1
    mtp_loss_weight: float = 0.1
    z_loss_weight: float = 1e-4
    loss_chunk_size: int = 512
    # -- misc --
    repeat_start: int = 0
    repeat_end: int = 0
    repeat_times: int = 1
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    activation_checkpointing: bool = False
    init_std: float = 0.02

    def validate(self):
        if (self.mixer_dim or self.n_dim) % self.n_head:
            raise ValueError("mixer_dim (default n_dim) must be divisible by n_head")
        layer_schedule(self.n_layer, self.repeat_start, self.repeat_end,
                       self.repeat_times)
        expand_pattern(self.mixer_pattern, self.n_layer)

    def to_dict(self):
        return asdict(self)


def swiglu_hidden_dim(n_dim: int, multiple_of: int = 64) -> int:
    """SwiGLU hidden width: ~8/3 * n_dim rounded up to a multiple of 64."""
    h = int(8 * n_dim / 3)
    return ((h + multiple_of - 1) // multiple_of) * multiple_of


def layer_schedule(n_layer: int, repeat_start: int = 0, repeat_end: int = 0,
                   repeat_times: int = 1) -> Tuple[int, ...]:
    """Execution order over the unique blocks. Blocks in [repeat_start,
    repeat_end) run ``repeat_times`` times back to back."""
    if repeat_times <= 1 or repeat_end <= repeat_start:
        return tuple(range(n_layer))
    if not (0 <= repeat_start < repeat_end <= n_layer):
        raise ValueError(
            f"repeat range [{repeat_start}, {repeat_end}) not inside [0, {n_layer}]")
    order = list(range(repeat_start))
    order += list(range(repeat_start, repeat_end)) * repeat_times
    order += list(range(repeat_end, n_layer))
    return tuple(order)


def expand_pattern(pattern: str, n_layer: int) -> Tuple[str, ...]:
    """Tile a mixer pattern such as ``"DS"`` or ``"DDSG"`` over ``n_layer``."""
    pattern = (pattern or "S").upper()
    bad = set(pattern) - set(MIXER_KINDS)
    if bad:
        raise ValueError(f"unknown mixer kinds {sorted(bad)}; use {MIXER_KINDS}")
    return tuple(pattern[i % len(pattern)] for i in range(n_layer))


# --------------------------------------------------------------------------
# Norm & RoPE
# --------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, n_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(dtype)


class GatedRMSNorm(nn.Module):
    """``RMSNorm(x) * silu(z)`` -- the output normalisation of a DeltaNet head."""

    def __init__(self, n_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_dim))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float() * F.silu(z.float())).to(dtype)


def precompute_rope(head_dim: int, seq_len: int, device, base: float = 10000.0):
    if head_dim % 2:
        raise ValueError("head_dim must be even for RoPE")
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D). cos/sin: (T, D), already in x's dtype."""
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


def softcap(logits: torch.Tensor, cap: float) -> torch.Tensor:
    if not cap:
        return logits
    return torch.tanh(logits / cap) * cap


# --------------------------------------------------------------------------
# Caches
# --------------------------------------------------------------------------

class AttentionCache:
    """KV cache for one attention layer.

    ``window=None`` is a pre-allocated full cache (global attention). Otherwise
    only the last ``window - 1`` keys are retained: a query at position p may
    see keys with ``p - k_pos < window``, so once the new token arrives the
    visible set is exactly ``window`` keys including itself. Because RoPE is
    applied before caching, only relative order matters and the buffer can be
    truncated freely.
    """

    def __init__(self, batch_size, n_kv_head, head_dim, window, max_len,
                 device, dtype):
        self.window = window
        self.first_pos = 0
        self.n = 0
        if window is None:
            shape = (batch_size, n_kv_head, max_len, head_dim)
            self.k = torch.empty(shape, device=device, dtype=dtype)
            self.v = torch.empty_like(self.k)
        else:
            self.k = None
            self.v = None

    def update(self, k, v, pos):
        """Append keys for positions [pos, pos+T). Returns (k_all, v_all, k_pos)."""
        T = k.size(2)
        if self.window is None:
            end = self.n + T
            if end > self.k.size(2):
                raise RuntimeError(f"KV cache overflow: {end} > {self.k.size(2)}")
            self.k[:, :, self.n:end] = k.to(self.k.dtype)
            self.v[:, :, self.n:end] = v.to(self.v.dtype)
            self.n = end
            k_pos = torch.arange(0, end, device=k.device)
            return self.k[:, :, :end], self.v[:, :, :end], k_pos

        if self.k is not None:
            k = torch.cat((self.k, k), dim=2)
            v = torch.cat((self.v, v), dim=2)
        S = k.size(2)
        k_pos = torch.arange(pos + T - S, pos + T, device=k.device)
        keep = self.window - 1
        if keep <= 0:
            self.k = self.v = None
        else:
            self.k, self.v = k[:, :, -keep:], v[:, :, -keep:]
        return k, v, k_pos


class GDNCache:
    """Recurrent state of one DeltaNet layer: the memory matrix and the tail
    of the short convolution's input."""

    def __init__(self):
        self.state = None        # (B, H, d_k, d_v) fp32
        self.conv_state = None   # (B, C, kernel-1)


class HybridCache:
    """One cache slot per *executed* layer (a looped block gets one per pass).
    ``pos`` is the number of tokens the model has consumed so far."""

    def __init__(self, model: "Transformer", batch_size, max_seq_len, device,
                 dtype):
        self.pos = 0
        self.max_seq_len = max_seq_len
        self.layers: List[object] = []
        for layer in model.layer_schedule:
            blk = model.blocks[layer]
            if blk.kind == "D":
                self.layers.append(GDNCache())
            else:
                a = blk.mixer
                self.layers.append(AttentionCache(
                    batch_size, a.n_kv_head, a.head_dim, a.window, max_seq_len,
                    device, dtype))

    def __getitem__(self, slot):
        return self.layers[slot]


KVCache = HybridCache   # name kept for imports


# --------------------------------------------------------------------------
# Gated DeltaNet
# --------------------------------------------------------------------------

def recurrent_gated_delta_rule(q, k, v, g, beta, state=None):
    """Token-by-token reference form. q,k: (B,H,T,dk) v: (B,H,T,dv)
    g (log decay <= 0), beta (0..1): (B,H,T). Returns (o, state)."""
    B, H, T, dk = q.shape
    dv = v.size(-1)
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    q = q * dk ** -0.5
    S = (torch.zeros(B, H, dk, dv, device=q.device, dtype=torch.float32)
         if state is None else state.float())
    outs = []
    for t in range(T):
        S = S * g[:, :, t].exp()[..., None, None]
        kt, vt = k[:, :, t], v[:, :, t]
        err = vt - torch.einsum("bhk,bhkv->bhv", kt, S)
        S = S + beta[:, :, t][..., None, None] * kt[..., :, None] * err[..., None, :]
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], S))
    return torch.stack(outs, dim=2), S


def chunk_gated_delta_rule(q, k, v, g, beta, state=None, chunk_size=64):
    """Chunkwise-parallel gated delta rule (WY representation), pure PyTorch.

    Same semantics as ``recurrent_gated_delta_rule``: within a chunk the
    sequence of rank-1 delta updates is written as ``S_c = decay * S_{c-1} +
    K^T (T V)`` where ``T = (I + tril(diag(b) K K^T * D, -1))^{-1}`` is a unit
    lower-triangular solve; between chunks a plain recurrence. All fp32.
    """
    B, H, L, dk = q.shape
    dv = v.size(-1)
    C = chunk_size
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    pad = (C - L % C) % C
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        g, beta = (F.pad(t, (0, pad)) for t in (g, beta))
    n = (L + pad) // C
    q = q * dk ** -0.5

    q, k, v = (t.reshape(B, H, n, C, -1) for t in (q, k, v))
    g = g.reshape(B, H, n, C)
    beta = beta.reshape(B, H, n, C)
    v_b = v * beta[..., None]
    k_b = k * beta[..., None]

    gc = g.cumsum(-1)                                   # (B,H,n,C)
    tril = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    diff = gc[..., :, None] - gc[..., None, :]           # gamma_i - gamma_j
    decay = diff.masked_fill(~tril, float("-inf")).exp()  # 0 above diagonal
    gc_exp = gc.exp()[..., None]

    A = (k_b @ k.transpose(-1, -2)) * decay
    eye = torch.eye(C, device=q.device, dtype=q.dtype)
    Tm = torch.linalg.solve_triangular(
        eye + A.tril(-1), eye.expand(B, H, n, C, C), upper=False, unitriangular=True)
    u = Tm @ v_b                                        # (B,H,n,C,dv)
    w = Tm @ (k_b * gc_exp)                             # (B,H,n,C,dk)

    # Everything the loop needs is computed batched over chunks, then unbound:
    # unbind's backward is one stack, whereas per-iteration slicing would give
    # autograd a zero-fill + copy per slice.
    attn_all = ((q @ k.transpose(-1, -2)) * decay).unbind(2)   # intra-chunk
    q_dec = (q * gc_exp).unbind(2)
    k_dec = (k * (gc[..., -1:] - gc).exp()[..., None]).transpose(-1, -2).unbind(2)
    last_dec = gc[..., -1].exp()[..., None, None].unbind(2)
    u_all, w_all = u.unbind(2), w.unbind(2)

    S = (torch.zeros(B, H, dk, dv, device=q.device, dtype=torch.float32)
         if state is None else state.float())
    outs = []
    for c in range(n):
        u_c = u_all[c] - w_all[c] @ S
        outs.append(q_dec[c] @ S + attn_all[c] @ u_c)
        S = S * last_dec[c] + k_dec[c] @ u_c
    o = torch.stack(outs, dim=2).reshape(B, H, n * C, dv)
    return o[:, :, :L], S


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet mixer. The recurrent state is the model's memory."""

    def __init__(self, n_dim, n_head, conv_kernel=4, chunk_size=64,
                 inner_dim=None):
        super().__init__()
        inner = inner_dim or n_dim
        self.n_dim = n_dim
        self.inner_dim = inner
        self.n_head = n_head
        self.head_dim = inner // n_head
        self.conv_kernel = conv_kernel
        self.chunk_size = chunk_size

        self.qkv_proj = nn.Linear(n_dim, 3 * inner, bias=False)
        self.z_proj = nn.Linear(n_dim, inner, bias=False)        # output gate
        self.ab_proj = nn.Linear(n_dim, 2 * n_head, bias=False)  # decay, beta
        self.conv = nn.Conv1d(3 * inner, 3 * inner, conv_kernel,
                              groups=3 * inner, bias=False)
        # Mamba-2 style decay parametrisation: a_t = exp(-exp(A_log) * softplus(.)).
        self.A_log = nn.Parameter(torch.empty(n_head).uniform_(1, 16).log())
        dt = torch.exp(torch.empty(n_head).uniform_(math.log(1e-3), math.log(1e-1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # softplus^-1
        self.norm = GatedRMSNorm(self.head_dim)
        self.wo = nn.Linear(inner, n_dim, bias=False)

    def _causal_conv(self, xc, T, reset=None):
        """Depthwise causal conv written as K shifted taps. ``xc`` is already
        left-padded by K-1. With ``reset`` (B,T), taps reaching back across a
        document start are zeroed, so packed documents do not leak into each
        other through the convolution either."""
        K = self.conv_kernel
        if reset is None:
            return self.conv(xc)
        w = self.conv.weight[:, 0, :]                            # (C, K)
        doc = reset.to(torch.int32).cumsum(1)                    # (B,T)
        doc_pad = F.pad(doc, (K - 1, 0), value=-1)
        same = (doc_pad.unfold(1, K, 1) == doc[..., None])       # (B,T,K)
        taps = xc.unfold(2, K, 1)                                # (B,C,T,K)
        taps = taps * same[:, None].to(taps.dtype)
        return torch.einsum("bctk,ck->bct", taps, w.to(taps.dtype))

    def forward(self, x, cache: Optional[GDNCache] = None, reset=None):
        B, T, _ = x.shape
        H, d = self.n_head, self.head_dim

        qkv = self.qkv_proj(x).transpose(1, 2)                   # (B,3D,T)
        K = self.conv_kernel
        if cache is not None and cache.conv_state is not None:
            xc = torch.cat((cache.conv_state.to(qkv.dtype), qkv), dim=2)
        else:
            xc = F.pad(qkv, (K - 1, 0))
        if cache is not None and K > 1:
            cache.conv_state = xc[:, :, -(K - 1):]
        qkv = F.silu(self._causal_conv(xc, T, reset)).transpose(1, 2)  # (B,T,3D)
        q, k, v = qkv.chunk(3, dim=-1)
        q = F.normalize(q.view(B, T, H, d), dim=-1).transpose(1, 2)
        k = F.normalize(k.view(B, T, H, d), dim=-1).transpose(1, 2)
        v = v.view(B, T, H, d).transpose(1, 2)

        a, b = self.ab_proj(x).float().chunk(2, dim=-1)           # (B,T,H)
        beta = torch.sigmoid(b).transpose(1, 2)                  # (B,H,T)
        g = (-self.A_log.float().exp() * F.softplus(a + self.dt_bias.float())
             ).transpose(1, 2)                                   # (B,H,T) <= 0
        if reset is not None:
            # a_t -> ~0 at document starts: the memory is wiped, not carried over.
            g = g.masked_fill(reset[:, None, :], -50.0)

        state = cache.state if cache is not None else None
        with torch.autocast(device_type=x.device.type, enabled=False):
            if T <= 2:
                o, S = recurrent_gated_delta_rule(q, k, v, g, beta, state)
            else:
                o, S = chunk_gated_delta_rule(q, k, v, g, beta, state,
                                              self.chunk_size)
        if cache is not None:
            cache.state = S

        o = o.to(x.dtype).transpose(1, 2)                        # (B,T,H,d)
        z = self.z_proj(x).view(B, T, H, d)
        o = self.norm(o, z).reshape(B, T, self.inner_dim)
        return self.wo(o)


# --------------------------------------------------------------------------
# Attention (sliding window or global)
# --------------------------------------------------------------------------

def default_n_kv_head(n_head: int) -> int:
    if n_head <= 4:
        kv = 1
    elif n_head <= 8:
        kv = 2
    elif n_head <= 16:
        kv = 4
    else:
        kv = 8
    while n_head % kv:
        kv -= 1
    return kv


class Attention(nn.Module):
    """GQA + RoPE + QK-norm; optional window, learned sink key, output gate,
    and value residual. ``window=None`` is full causal attention."""

    def __init__(self, n_dim, n_head, n_kv_head=None, window=None,
                 rope_theta=10000.0, sink=True, output_gate=True,
                 value_residual=True, inner_dim=None):
        super().__init__()
        inner = inner_dim or n_dim
        if inner % n_head != 0:
            raise ValueError("inner_dim (default n_dim) must be divisible by n_head")
        self.n_dim = n_dim
        self.inner_dim = inner
        self.n_head = n_head
        self.head_dim = inner // n_head
        self.window = window
        self.rope_theta = rope_theta

        n_kv_head = n_kv_head or max(1, n_head // 4)
        if n_head % n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head")
        self.n_kv_head = n_kv_head
        self.n_rep = n_head // n_kv_head

        self.q_proj = nn.Linear(n_dim, inner, bias=False)
        self.k_proj = nn.Linear(n_dim, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_dim, n_kv_head * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(n_dim, inner, bias=True) if output_gate else None
        self.wo = nn.Linear(inner, n_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        # Sink: a learned key per kv head with a zero value. Attending to it
        # lets a head emit (almost) nothing, which sliding windows need once
        # the BOS token has scrolled out of view.
        self.sink_k = nn.Parameter(torch.zeros(n_kv_head, self.head_dim)) if sink else None
        # (1, 0): identical to plain attention at init; the mix is learned.
        self.v_lambda = nn.Parameter(torch.tensor([1.0, 0.0])) if value_residual else None

    def forward(self, x, cos, sin, pos=0, cache: Optional[AttentionCache] = None,
                attn_mask=None, v_first=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v_raw = v
        if v_first is not None and self.v_lambda is not None:
            v = self.v_lambda[0] * v + self.v_lambda[1] * v_first

        q = self.q_norm(q)
        k = self.k_norm(k)
        c = cos[pos:pos + T].to(q.dtype)
        s = sin[pos:pos + T].to(q.dtype)
        q = apply_rope(q, c, s)
        k = apply_rope(k, c, s)

        fast = (cache is None and attn_mask is None
                and (self.window is None or self.window >= T))
        if fast:
            # Plain causal attention -> flash kernel, no mask. With a sink we
            # prepend the sink key *and* a dummy query so that under is_causal
            # query i+1 sees exactly {sink, k_0..k_i}; the dummy row is dropped.
            mask = None
        else:
            # (T, S) band mask from positions; serves training with a short
            # window, chunked prefill and single-token decode alike.
            q_pos = torch.arange(pos, pos + T, device=x.device)
            if cache is not None:
                k, v, k_pos = cache.update(k, v, pos)
            else:
                k_pos = q_pos
            dist = q_pos[:, None] - k_pos[None, :]
            mask = dist >= 0
            if self.window is not None:
                mask = mask & (dist < self.window)
            mask = mask[None, None]
            if attn_mask is not None:
                if attn_mask.shape[-1] != mask.shape[-1]:
                    raise ValueError("attn_mask is only supported without a cache")
                mask = mask & attn_mask

        if self.sink_k is not None:
            sk = self.sink_k.to(k.dtype)[None, :, None, :].expand(B, -1, 1, -1)
            k = torch.cat((sk, k), dim=2)
            v = torch.cat((torch.zeros_like(sk), v), dim=2)
            if fast:
                q = torch.cat((torch.zeros_like(q[:, :, :1]), q), dim=2)
            else:
                mask = F.pad(mask, (1, 0), value=True)

        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).flatten(1, 2)
            v = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).flatten(1, 2)

        if fast:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            if self.sink_k is not None:
                out = out[:, :, 1:]
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).contiguous().view(B, T, self.inner_dim)
        if self.gate_proj is not None:
            out = out * torch.sigmoid(self.gate_proj(x))
        return self.wo(out), v_raw


AttentionGQA = Attention   # old name


# --------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, n_dim, hidden_dim):
        super().__init__()
        self.g = nn.Linear(n_dim, hidden_dim, bias=False)
        self.u = nn.Linear(n_dim, hidden_dim, bias=False)
        self.d = nn.Linear(hidden_dim, n_dim, bias=False)

    def forward(self, x):
        return self.d(F.silu(self.g(x)) * self.u(x))


class FFN(nn.Module):
    """Kept for name compatibility; a dense SwiGLU block."""

    def __init__(self, n_dim, num_experts=1, top_k=1, num_shared_experts=1,
                 hidden_dim=None):
        super().__init__()
        self.ffn = SwiGLU(n_dim, hidden_dim or swiglu_hidden_dim(n_dim))

    def forward(self, x):
        return self.ffn(x)


class Block(nn.Module):
    """Pre-norm ``mixer + SwiGLU``. ``kind`` selects the mixer."""

    def __init__(self, kind, n_dim, n_head, hidden_dim, *, n_kv_head=None,
                 window=None, rope_theta=10000.0, sink=True, output_gate=True,
                 value_residual=True, conv_kernel=4, chunk_size=64, dropout=0.0,
                 mixer_dim=None):
        super().__init__()
        if kind not in MIXER_KINDS:
            raise ValueError(f"unknown mixer kind {kind!r}")
        self.kind = kind
        self.mixer_norm = RMSNorm(n_dim)
        if kind == "D":
            self.mixer = GatedDeltaNet(n_dim, n_head, conv_kernel, chunk_size,
                                       mixer_dim)
        else:
            self.mixer = Attention(
                n_dim, n_head, n_kv_head, window if kind == "S" else None,
                rope_theta, sink, output_gate, value_residual, mixer_dim)
        self.ffn_norm = RMSNorm(n_dim)
        self.ffn = SwiGLU(n_dim, hidden_dim)
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, cos, sin, pos=0, cache=None, attn_mask=None,
                reset=None, v_first=None):
        h = self.mixer_norm(x)
        if self.kind == "D":
            m, v = self.mixer(h, cache, reset), None
        else:
            m, v = self.mixer(h, cos, sin, pos, cache, attn_mask, v_first)
        x = x + self.resid_dropout(m)
        x = x + self.resid_dropout(self.ffn(self.ffn_norm(x)))
        return x, v


class MTPHead(nn.Module):
    """Multi-token prediction module (DeepSeek-V3 form, depth 1): from the main
    model's hidden state at t and the embedding of the *true* token t+1,
    predict token t+2 through one extra sliding-window block. Trains the trunk
    to carry enough information to see two steps ahead."""

    def __init__(self, n_dim, n_head, hidden_dim, **block_kw):
        super().__init__()
        self.h_norm = RMSNorm(n_dim)
        self.e_norm = RMSNorm(n_dim)
        self.proj = nn.Linear(2 * n_dim, n_dim, bias=False)
        self.block = Block("S", n_dim, n_head, hidden_dim, **block_kw)
        self.norm = RMSNorm(n_dim)

    def forward(self, h, emb_next, cos, sin, attn_mask=None):
        x = self.proj(torch.cat((self.h_norm(h), self.e_norm(emb_next)), dim=-1))
        x, _ = self.block(x, cos, sin, 0, None, attn_mask)
        return self.norm(x)


# --------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------

def build_document_mask(idx: torch.Tensor, eos_id: int, n_head: int = 1):
    """Block-diagonal causal mask so tokens cannot attend across an EOS.
    Returns a bool mask of shape (B, 1, T, T), True = attend. The model also
    derives DeltaNet memory resets from it (a token that cannot see its
    predecessor starts a new document)."""
    B, T = idx.shape
    doc = (idx == eos_id).cumsum(dim=1)
    doc = doc - (idx == eos_id).long()
    same_doc = doc[:, :, None] == doc[:, None, :]
    causal = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
    return (same_doc & causal).unsqueeze(1)


def resets_from_mask(attn_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """(B, T) bool: True where a token may not see the previous token."""
    if attn_mask is None:
        return None
    m = attn_mask[:, 0]
    T = m.size(-1)
    if T < 2:
        return None
    sees_prev = m[:, 1:, :-1].diagonal(dim1=-2, dim2=-1)
    reset = torch.cat((torch.zeros_like(sees_prev[:, :1]), ~sees_prev), dim=1)
    return reset if bool(reset.any()) else None


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_layer=DEFAULT_NUM_LAYERS,
        n_head=DEFAULT_NUM_HEADS,
        n_dim=DEFAULT_EMBEDDING_DIM,
        n_seq=DEFAULT_SEQ_LEN,
        mixer_dim=None,
        ffn_hidden=None,
        emb_dim=None,
        mixer_pattern="DS",
        window_size=512,
        n_kv_head=None,
        attention_sink=True,
        attn_output_gate=False,
        value_residual=True,
        unet_skips=True,
        gdn_conv_kernel=4,
        gdn_chunk_size=64,
        logit_softcap=30.0,
        mtp_depth=1,
        mtp_loss_weight=0.1,
        z_loss_weight=1e-4,
        loss_chunk_size=512,
        repeat_start=0,
        repeat_end=0,
        repeat_times=1,
        rope_theta=10000.0,
        dropout=0.0,
        tie_embeddings=True,
        activation_checkpointing=False,
        init_std=0.02,
        num_experts=DEFAULT_NUM_EXPERTS,      # accepted, unused (dense)
        top_k=DEFAULT_TOP_K,                  # accepted, unused (dense)
        num_shared_experts=1,                 # accepted, unused (dense)
        debug_token_range=False,
    ):
        super().__init__()
        mixer_dim = mixer_dim or n_dim
        emb_dim = emb_dim or n_dim
        if mixer_dim % n_head != 0:
            raise ValueError("mixer_dim (default n_dim) must be divisible by n_head")
        head_dim = mixer_dim // n_head
        if head_dim % 2:
            raise ValueError(f"head_dim ({head_dim}) must be even for RoPE")
        if head_dim not in (32, 64, 128):
            warnings.warn(
                f"head_dim={head_dim} misses the fast attention kernels; "
                f"n_head={mixer_dim // 64} would give head_dim 64 at "
                f"mixer_dim={mixer_dim}",
                stacklevel=2,
            )

        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.layer_schedule = layer_schedule(
            n_layer, repeat_start, repeat_end, repeat_times)
        self.n_executed_layer = len(self.layer_schedule)
        self.repeat_start, self.repeat_end, self.repeat_times = \
            repeat_start, repeat_end, repeat_times
        self.n_head = n_head
        self.n_dim = n_dim
        self.mixer_dim = mixer_dim
        self.head_dim = head_dim
        self.emb_dim = emb_dim
        self.n_seq = n_seq
        self.rope_theta = rope_theta
        self.kinds = expand_pattern(mixer_pattern, n_layer)
        self.mixer_pattern = mixer_pattern
        self.window_size = window_size
        self.has_global = "G" in self.kinds
        self.activation_checkpointing = activation_checkpointing
        self.loss_chunk_size = loss_chunk_size
        self.z_loss_weight = z_loss_weight
        self.logit_softcap = logit_softcap
        self.mtp_loss_weight = mtp_loss_weight
        self.debug_token_range = debug_token_range
        self.init_std = init_std
        self.unet_skips = unet_skips
        self.last_metrics = {}

        self.n_kv_head = n_kv_head or default_n_kv_head(n_head)
        if n_head % self.n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head")

        self.l_embeddings = nn.Embedding(vocab_size, emb_dim)
        # Factorised embedding: vocab x emb_dim table, projected up to n_dim on
        # the way in and down to emb_dim before the (tied) head on the way out.
        factored = emb_dim != n_dim
        self.emb_proj = nn.Linear(emb_dim, n_dim, bias=False) if factored else None
        self.logit_down = nn.Linear(n_dim, emb_dim, bias=False) if factored else None
        self.emb_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        hidden_dim = ffn_hidden or swiglu_hidden_dim(n_dim)
        self.ffn_hidden = hidden_dim
        block_kw = dict(n_kv_head=self.n_kv_head, window=window_size,
                        rope_theta=rope_theta, sink=attention_sink,
                        output_gate=attn_output_gate,
                        value_residual=value_residual,
                        conv_kernel=gdn_conv_kernel, chunk_size=gdn_chunk_size,
                        dropout=dropout, mixer_dim=mixer_dim)
        # The first *executed* attention layer is the value-residual source
        # and has no lambda of its own (an unused parameter breaks DDP).
        attn_layers = [l for l in self.layer_schedule if self.kinds[l] != "D"]
        first_attn = attn_layers[0] if attn_layers else -1
        self.blocks = nn.ModuleList([
            Block(self.kinds[i], n_dim, n_head, hidden_dim,
                  **{**block_kw, "value_residual": value_residual and i != first_attn})
            for i in range(n_layer)
        ])

        n_exec = self.n_executed_layer
        # x = l0 * x + l1 * emb before every layer (modded-nanogpt "lambdas").
        self.resid_lambdas = nn.Parameter(
            torch.tensor([1.0, 0.0]).repeat(n_exec, 1))
        self.n_encoder = n_exec // 2 if unet_skips else 0
        # Zero-init: the skips are a no-op at step 0 and grow if useful.
        self.skip_weights = (nn.Parameter(torch.zeros(self.n_encoder))
                             if self.n_encoder else None)

        self.mtp = (MTPHead(n_dim, n_head, hidden_dim,
                            **{**block_kw, "value_residual": False})
                    if mtp_depth > 0 else None)

        self.final_norm = RMSNorm(n_dim)
        self.logit_proj = nn.Linear(emb_dim, vocab_size, bias=False)
        if tie_embeddings:
            self.logit_proj.weight = self.l_embeddings.weight

        self._rope_cos = None
        self._rope_sin = None
        self._rope_cached_len = 0
        self._rope_device = None

        self.apply(self._init_weights)

        # GPT-2 residual scaling over *executed* layers (see model_v1 notes).
        residual_depth = n_exec + (1 if self.mtp is not None else 0)
        for name, p in self.named_parameters():
            if name.endswith(("wo.weight", "d.weight")):
                nn.init.normal_(
                    p, mean=0.0, std=init_std / math.sqrt(2 * residual_depth))
            elif name.endswith("gate_proj.bias"):
                # sigmoid(3) ~ 0.95: the attention output gate starts open.
                nn.init.constant_(p, 3.0)
            elif name in ("emb_proj.weight", "logit_down.weight"):
                # 1/sqrt(fan_in): the up-projected embedding keeps init_std
                # scale, and the head input keeps the final norm's unit RMS.
                nn.init.normal_(p, mean=0.0, std=p.size(1) ** -0.5)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "Transformer":
        cfg.validate()
        return cls(**cfg.to_dict())

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _get_rope(self, seq_len, device):
        if (self._rope_cos is None
                or seq_len > self._rope_cached_len
                or device != self._rope_device):
            self._rope_cos, self._rope_sin = precompute_rope(
                self.head_dim, seq_len, device, self.rope_theta)
            self._rope_cached_len = seq_len
            self._rope_device = device
        return self._rope_cos, self._rope_sin

    # -- forward ------------------------------------------------------------

    def embed(self, idx):
        """Token ids -> residual-width vectors (through the up-projection
        when the embedding is factorised)."""
        e = self.l_embeddings(idx)
        return self.emb_proj(e) if self.emb_proj is not None else e

    def head(self, h):
        """Hidden states -> raw logits (before soft-capping)."""
        if self.logit_down is not None:
            h = self.logit_down(h)
        return self.logit_proj(h)

    def forward_hidden(self, idx, start_pos=0, kv_cache=None, attn_mask=None):
        """Everything up to and including final_norm.

        With ``kv_cache`` the tokens are appended at ``kv_cache.pos``
        (``start_pos`` is accepted for compatibility and must agree)."""
        if self.debug_token_range and idx.numel():
            lo, hi = idx.min().item(), idx.max().item()
            if lo < 0 or hi >= self.vocab_size:
                raise ValueError(
                    f"Token id range [{lo}, {hi}] outside [0, {self.vocab_size - 1}]")

        B, T = idx.shape
        if kv_cache is None:
            if T > self.n_seq:
                raise ValueError(f"Sequence length {T} exceeds n_seq={self.n_seq}")
            pos = 0
        else:
            pos = kv_cache.pos
            if start_pos not in (0, pos):
                raise ValueError(f"start_pos={start_pos} but cache is at {pos}")
            if attn_mask is not None:
                raise ValueError("attn_mask is not supported with a kv_cache")

        x = self.emb_dropout(self.embed(idx))
        x0 = x
        cos, sin = self._get_rope(max(pos + T, self.n_seq), x.device)
        reset = resets_from_mask(attn_mask)

        skips = []
        v_first = None
        n_enc = self.n_encoder
        for slot, layer in enumerate(self.layer_schedule):
            if n_enc and slot >= n_enc and slot - n_enc < n_enc:
                x = x + self.skip_weights[slot - n_enc] * skips.pop()
            lam = self.resid_lambdas[slot]
            x = lam[0] * x + lam[1] * x0
            block = self.blocks[layer]
            cache_i = kv_cache[slot] if kv_cache is not None else None
            if self.activation_checkpointing and self.training:
                x, v = torch.utils.checkpoint.checkpoint(
                    block, x, cos, sin, pos, cache_i, attn_mask, reset, v_first,
                    use_reentrant=False)
            else:
                x, v = block(x, cos, sin, pos, cache_i, attn_mask, reset, v_first)
            if v is not None and v_first is None:
                v_first = v
            if n_enc and slot < n_enc:
                skips.append(x)

        if kv_cache is not None:
            kv_cache.pos = pos + T
        return self.final_norm(x)

    def forward(self, idx, start_pos=0, kv_cache=None, attn_mask=None,
                mode="logits", targets=None, length_normalize=True,
                loss_weights=None):
        """Returns (logits, None) in the default mode. ``mode`` routes every
        training objective through ``forward`` so DDP's reducer sees it."""
        if mode == "loss":
            return self.calculate_loss(idx, targets, attn_mask=attn_mask,
                                       weights=loss_weights)
        if mode == "seq_logprobs":
            return self.sequence_logprobs(
                idx, targets, attn_mask=attn_mask,
                length_normalize=length_normalize)
        if mode == "hidden":
            return self.forward_hidden(idx, start_pos, kv_cache, attn_mask)
        x = self.forward_hidden(idx, start_pos, kv_cache, attn_mask)
        return softcap(self.head(x), self.logit_softcap), None

    # -- loss ---------------------------------------------------------------

    def _loss_chunk(self, h, targets, weights=None):
        """Projection + CE for one slice -> stacked (ce_sum, z_sum, w_sum)."""
        logits = softcap(self.head(h).float(), self.logit_softcap)
        flat = logits.reshape(-1, self.vocab_size)
        tgt = targets.reshape(-1)
        valid = tgt != IGNORE_INDEX
        if weights is None:
            ce = F.cross_entropy(flat, tgt, reduction="sum",
                                 ignore_index=IGNORE_INDEX)
            w = valid.to(ce.dtype)
        else:
            per = F.cross_entropy(flat, tgt, reduction="none",
                                  ignore_index=IGNORE_INDEX)
            w = weights.reshape(-1).to(per.dtype) * valid
            ce = (per * w).sum()
        if self.z_loss_weight:
            z = torch.logsumexp(flat, dim=-1)
            z_sum = (w * z ** 2).sum()
        else:
            z_sum = ce.new_zeros(())
        return torch.stack((ce, z_sum, w.sum()))

    def _chunked_ce(self, h, ys, weights=None):
        T = h.size(1)
        chunk = self.loss_chunk_size or T
        totals = None
        for i in range(0, T, chunk):
            hs = h[:, i:i + chunk]
            ts = ys[:, i:i + chunk]
            ws = None if weights is None else weights[:, i:i + chunk]
            if self.training and torch.is_grad_enabled():
                part = torch.utils.checkpoint.checkpoint(
                    self._loss_chunk, hs, ts, ws, use_reentrant=False)
            else:
                part = self._loss_chunk(hs, ts, ws)
            totals = part if totals is None else totals + part
        ce_sum, z_sum, w_sum = totals[0], totals[1], totals[2].clamp(min=1.0)
        return ce_sum / w_sum, z_sum / w_sum

    def calculate_loss(self, xs, ys, attn_mask=None, weights=None, **_legacy):
        """Token-level CE (+ z-loss, + weighted MTP loss), chunked over the
        sequence. Per-part values are left in ``self.last_metrics``."""
        h = self.forward_hidden(xs, attn_mask=attn_mask)
        ce, z = self._chunked_ce(h, ys, weights)
        loss = ce + self.z_loss_weight * z if self.z_loss_weight else ce
        metrics = {"ce": ce.detach(), "z": z.detach()}

        if self.mtp is not None and self.mtp_loss_weight and xs.size(1) > 2:
            cos, sin = self._get_rope(self.n_seq, h.device)
            emb_next = self.embed(xs[:, 1:])
            m = None if attn_mask is None else attn_mask[:, :, :-1, :-1]
            h2 = self.mtp(h[:, :-1], emb_next, cos, sin, m)
            w2 = None if weights is None else weights[:, 1:]
            ce2, _ = self._chunked_ce(h2, ys[:, 1:], w2)
            loss = loss + self.mtp_loss_weight * ce2
            metrics["mtp_ce"] = ce2.detach()
        self.last_metrics = metrics
        return loss

    def sequence_logprobs(self, xs, ys, attn_mask=None,
                          length_normalize: bool = True):
        """log p(target) summed over unmasked positions, one value per sequence."""
        hidden = self.forward_hidden(xs, attn_mask=attn_mask)
        chunk = self.loss_chunk_size or hidden.size(1)
        total = torch.zeros(xs.size(0), device=xs.device, dtype=torch.float32)
        count = torch.zeros(xs.size(0), device=xs.device, dtype=torch.float32)
        for i in range(0, hidden.size(1), chunk):
            h = hidden[:, i:i + chunk]
            t = ys[:, i:i + chunk]
            logits = softcap(self.head(h).float(), self.logit_softcap)
            valid = t != IGNORE_INDEX
            safe = t.masked_fill(~valid, 0)
            logp = torch.log_softmax(logits, dim=-1)
            picked = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
            total = total + (picked * valid).sum(dim=-1)
            count = count + valid.sum(dim=-1)
        if length_normalize:
            return total / count.clamp(min=1.0)
        return total

    # -- inference ----------------------------------------------------------

    def make_kv_cache(self, batch_size, max_seq_len, device=None, dtype=None):
        p = self.l_embeddings.weight
        return HybridCache(self, batch_size, max_seq_len,
                           device or p.device, dtype or p.dtype)

    def _sample(self, logits, temperature, top_k, top_p, min_p):
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / max(temperature, 1e-5)
        if top_k:
            k = min(top_k, logits.size(-1))
            kth = logits.topk(k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if min_p and min_p > 0:
            probs = F.softmax(logits, dim=-1)
            thresh = min_p * probs.max(dim=-1, keepdim=True).values
            logits = logits.masked_fill(probs < thresh, float("-inf"))
        if top_p and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1)
            remove = probs.cumsum(-1) - probs > top_p
            remove = remove.scatter(-1, sorted_idx, remove)
            logits = logits.masked_fill(remove, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.inference_mode()
    def generate(
        self,
        idx,
        max_count=128,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        min_p=0.0,
        eos_token_id=None,
        repetition_penalty=1.0,
        valid_vocab_size=None,
        prefill_chunk=1024,
    ):
        """Autoregressive sampling. The prompt is prefilled in chunks so memory
        stays bounded; without global layers there is no length limit."""
        B, T = idx.shape
        total = T + max_count
        if self.has_global and total > self.n_seq:
            raise ValueError(
                f"Generation needs {total} positions but n_seq={self.n_seq} "
                f"(global attention layers present)")

        cache = self.make_kv_cache(B, total, idx.device)
        self._get_rope(max(total, self.n_seq), idx.device)

        for i in range(0, T, prefill_chunk):
            h = self.forward_hidden(idx[:, i:i + prefill_chunk], kv_cache=cache)
        logits = softcap(self.head(h[:, -1]).float(), self.logit_softcap)

        done = torch.zeros(B, dtype=torch.bool, device=idx.device)
        for _ in range(max_count):
            if valid_vocab_size is not None and valid_vocab_size < self.vocab_size:
                logits[:, valid_vocab_size:] = float("-inf")
            if repetition_penalty != 1.0:
                gathered = torch.gather(logits, 1, idx)
                gathered = torch.where(gathered > 0,
                                       gathered / repetition_penalty,
                                       gathered * repetition_penalty)
                logits = logits.scatter(1, idx, gathered)

            next_token = self._sample(logits, temperature, top_k, top_p, min_p)
            if eos_token_id is not None:
                next_token = torch.where(done[:, None],
                                         torch.full_like(next_token, eos_token_id),
                                         next_token)
                done = done | (next_token.squeeze(1) == eos_token_id)
            idx = torch.cat((idx, next_token), dim=1)
            if eos_token_id is not None and bool(done.all()):
                break
            h = self.forward_hidden(next_token, kv_cache=cache)
            logits = softcap(self.head(h[:, -1]).float(), self.logit_softcap)
        return idx

    # -- bookkeeping --------------------------------------------------------

    def get_param_count(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.l_embeddings.weight.numel()
        return n

    def param_groups(self, weight_decay=0.1):
        """Decay matrices, not gains/biases/scalars -- the standard split."""
        decay, no_decay = [], []
        for p in self.parameters():
            if p.requires_grad:
                (decay if p.dim() >= 2 else no_decay).append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def estimate_flops_per_token(self):
        """6 * (weights multiplied through, per executed layer) + mixer cost."""
        matmul = self.vocab_size * self.emb_dim
        if self.logit_down is not None:
            matmul += 2 * self.emb_dim * self.n_dim
        mixer = 0
        for layer in self.layer_schedule:
            blk = self.blocks[layer]
            matmul += sum(p.numel() for p in blk.parameters())
            if blk.kind == "D":
                mixer += 12 * self.mixer_dim * blk.mixer.chunk_size
            else:
                span = self.n_seq if blk.kind == "G" else min(self.window_size, self.n_seq)
                mixer += 12 * self.mixer_dim * span
        if self.mtp is not None:
            matmul += sum(p.numel() for p in self.mtp.parameters())
        return 6 * matmul + mixer

    def estimate_mfu(self, tokens_per_sec: float, peak_flops: float) -> float:
        return (self.estimate_flops_per_token() * tokens_per_sec) / max(1.0, peak_flops)
