# Model architecture

`model.py` is a hybrid **Gated DeltaNet + sliding-window attention** language
model written in plain PyTorch (no Triton, no custom kernels; it runs on a
6 GB laptop GPU in eager mode). `optim.py` is the matching Muon + AdamW
optimizer. `model_v1_dense.py` is the previous dense transformer, kept for
comparison.

## Layers

Each layer is `RMSNorm -> mixer -> residual -> RMSNorm -> SwiGLU -> residual`.
`mixer_pattern` chooses the mixer per layer and is tiled over `n_layer`:

| kind | mixer | memory at inference | context |
|------|-------|---------------------|---------|
| `D`  | Gated DeltaNet: per-head matrix state `S (d_k x d_v)` updated every token by a gated delta rule | constant | unbounded |
| `S`  | sliding-window attention (GQA, RoPE, QK-norm, learned sink key) | `window_size` keys | unbounded |
| `G`  | full causal attention | grows with length | `n_seq` max |

The DeltaNet state is the model's *active memory*: it is overwritten in place
as tokens arrive (`S_t = a_t S_{t-1} + b_t k_t (v_t - a_t S_{t-1}^T k_t)^T`),
with a learned per-token forget gate `a_t` and write strength `b_t`. Training
uses the chunkwise-parallel WY form; decoding uses the one-token recurrence.
Both are verified equal to 1e-7 (`verify_model.py`).

Without any `G` layer there is **no maximum context length**: RoPE only ever
sees distances below `window_size`, and inference memory is constant.

## Training objective

* Chunked cross-entropy with **logit soft-capping** (`tanh`, cap 30) and z-loss.
* **Multi-token prediction** (`mtp_depth=1`): a small extra block predicts
  token t+2 from `[h_t ; emb(x_{t+1})]`; `mtp_loss_weight` (0.1) scales it.
  Per-part values are in `model.last_metrics` after every loss call.
* Document masks (`build_document_mask`) block attention across EOS and also
  reset the DeltaNet memory and its short convolution at document starts, so a
  packed batch equals running the documents separately (verified).

## Other pieces, all optional

| flag | default | what it does | measured effect (7M params, 1.2M tokens) |
|------|---------|--------------|-------------------------------------------|
| `attention_sink` | on | learned sink key per KV head (lets heads attend to nothing; matters once BOS scrolls out of the window) | neutral |
| `logit_softcap` | 30 | bounds logits, stability at high LR | neutral |
| `value_residual` | on | later attention layers mix in the first layer's values (init = identity) | neutral |
| `unet_skips` | on | encoder-to-decoder residual skips (init = zero) | neutral |
| `attn_output_gate` | off | sigmoid gate on attention output (Qwen3-Next) | -0.016 nats worse at this scale, so off |
| `mtp_depth` | 1 | multi-token prediction head | -0.02 nats better in most runs, +25% step time |
| `repeat_*` | off | run a middle group of blocks more than once | unchanged from v1 |

## Optimizer

```python
from optim import build_optimizer
opt = build_optimizer(model, lr=1e-3, weight_decay=0.1)   # Muon for block matrices, AdamW for the rest
```

Muon orthogonalises each matrix update (batched Newton-Schulz); the LR is
rescaled per matrix so one `lr` works for both halves. On the ghost corpus it
reached the old model's 600-step loss in roughly 400 steps.

## What the measurements say (be honest with yourself)

All numbers: held-out next-token cross-entropy, 8 layers, dim 256, Muon.

**Ghost corpus (600K training tokens), sequence 256, 300 steps:**

| layout | window | CE @256 ctx | CE @2048 ctx |
|--------|--------|-------------|--------------|
| old dense (full attention) | -- | 4.34 | 4.68 (degrades past training length) |
| `S` | 64 | **4.29** | **4.29** |
| `SSSD` | 64 | 4.32 | 4.32 |
| `DS` | 64 | 4.42 | 4.42 |
| `D` | -- | 4.53 | 4.53 |

* Both new layouts are flat out to 8x the training length.
* On this corpus, context beyond ~64 tokens carries no usable signal even for
  full attention, so the memory layers cannot pay for themselves and each one
  costs a little quality. **At tiny data budgets, `mixer_pattern="S"` is the
  best next-token model.**

**Associative recall (12 key-value pairs, ~100 tokens away, window 16):**

| layout | recall accuracy |
|--------|-----------------|
| `S` (window only) | 3% (chance) |
| `DS` | 31%, equal to full attention |
| `D` | 31%, equal to full attention |
| `G` (full attention) | 30% |

(1500 steps, 4 layers x 128 dim; all three memory-capable models plateau together, so 30% is the task/size ceiling here, not a memory limit.)

The memory demonstrably stores and retrieves facts the window cannot see.

**TinyStories (5M training tokens, 1.6 epochs), sequence 512, window 128:**

| layout | CE @256 ctx | CE @512 | CE @1024 |
|--------|-------------|---------|----------|
| old dense (full attention) | 2.006 | 2.006 | 2.239 (past training length) |
| `S` | **1.987** | **1.986** | **1.986** |
| `SSSD` | 1.998 | 1.997 | 1.997 |
| `DS` | 2.004 | 2.003 | 2.003 |

* The hybrid's cost versus window-only attention fell from 0.13 nats (ghost
  corpus) to 0.017 nats here; it shrinks as the data budget grows.
* Stories are ~200 tokens, so nothing (not even full attention) gains from
  context beyond 256 tokens on this corpus either. The recall task above is
  the test where the memory matters.

## Recommendations

* Tiny corpus (< ~5M tokens): `mixer_pattern="S"`, `window_size` 128-512.
* You need facts recalled from far back, streaming input, or unbounded
  context: `"DS"` or `"SSD"`. Expect a small perplexity cost until the data
  budget is tens of millions of tokens.
* Keep `mtp_depth=1` unless step time is the constraint.
* Train with fp32 weights under `torch.autocast(dtype=bfloat16)`; do not cast
  the model to bf16 (see `test_model.py`).
* Speed (RTX 3060 laptop, 8x256 model, 16x512 tokens, forward+backward):
  old dense 87 ms; `S` 114 ms; `S` + MTP 150 ms; `DS` + MTP 215 ms. When
  `window_size` is shorter than the training sequence, attention takes the
  masked (non-flash) kernel; the DeltaNet layer is ~2x an attention layer in
  eager mode. Installing `triton-windows` (`pip install triton-windows`)
  makes `torch.compile` work on Windows and removes most of that overhead.

## Files

* `verify_model.py` -- numerical checks (delta rule, cached decode, document
  reset, training paths). Run after any change.
* `compare_models.py` -- old vs new on the ghost corpus, named ablations.
* `longctx_eval.py` -- fixed-token context-length evaluation; env vars pick
  corpus, window, sequence length.
* `recall_test.py` -- synthetic associative recall.

## Training on smollm-corpus

Two pipelines share `pretrain.py`'s data loader, checkpointing and flags.
Every `ModelConfig` field is a command-line flag. The default model
(`MODEL_DEFAULTS` in `pretrain.py`) uses the whole design: `DS`, window 512
over 2048-token sequences, residual 896 with mixer 512 / FFN 1792,
factorised embedding 256, blocks 5-8 looped twice (14 unique -> 18 executed
layers). That is 96M params at inference plus a 7.6M training-only MTP head.
Packed documents use document masks, so attention and the DeltaNet state
never cross an EOS.

### Distillation (recommended)

```bash
./run_distill.sh                   # student vocab -> teacher-token data -> distill.py
python generate.py "Once upon a time"
```

* Teacher: SmolLM2-1.7B (49152 vocab). Student vocab: the teacher's BPE
  truncated to ids < 16384 (`distill_vocab.py`). SmolLM2 numbers tokens in
  merge order, so this is a valid BPE. Every teacher token is a fixed
  sequence of 1-8 student tokens, and teacher token boundaries are always
  student boundaries (checked on 3,000 documents: 100% identical, 1.097
  student tokens per teacher token).
* Mapping teacher distributions (`distill.py`), for a student token at depth
  `d` inside teacher token `t_j`, using the teacher's distribution `p` at
  `j - 1`:
  * `d = 0` (~90% of positions) is exact: `q(s) = sum_t p(t) [first piece of t = s]`.
  * `d > 0` is the teacher's conditional over the next piece given the
    pieces already emitted (prefix trie). The paths "teacher token ends
    here, next one starts with s" would need a second teacher pass. Their
    share of the prefix mass (logged as `dropped_mass`, ~10% of interior
    mass, about 1% of all target mass) is dropped and the rest renormalised.
  * Checked against a brute-force implementation, and the per-token
    probabilities telescope back to the teacher's.
* Loss: `alpha * H(q, student) + (1 - alpha) * CE` with alpha 0.9, the same
  for the MTP head (its soft target is `q` at the next position), plus z-loss.
* Data is stored in teacher ids (`data/smollm_t49k`). Student ids are
  derived exactly on the GPU. The teacher only runs on the prefix that
  covers the student window.
* Speed (A30): ~13K student tok/s. The teacher forward (~22K tok/s, 46%
  MFU) is ~75% of the time, so 2B student tokens take ~40 h.

### Pretraining from scratch

```bash
./run_pretrain.sh                  # own 16k BPE -> data -> pretrain.py
```

### Kernels

* Gated DeltaNet uses the `flash-linear-attention` Triton kernel when it is
  installed and on CUDA. Otherwise it uses the pure-PyTorch chunk form, which
  stays the reference (`verify_model.py` check 5).
* Under `torch.compile`, windowed / document-masked attention runs as
  block-sparse FlexAttention, with the sink as key 0 (check 6). Eager mode
  and decoding keep the SDPA path.
* The document-reset short conv uses shifted slices instead of `unfold`,
  whose bf16 backward took 60% of the step.
* Together these took the hybrid student from 15.7K to 49K tok/s
  (forward + backward, 8 x 2048, A30).
