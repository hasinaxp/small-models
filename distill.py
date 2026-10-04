"""Distil SmolLM2-1.7B (49k vocab) into the hybrid small LM (16k vocab).

    python distill_vocab.py                    # once: student tokenizer + vocab map
    python prepare_data.py --tokenizer tokenizer/smollm2/tokenizer.json \
        --out data/smollm_t49k --tokens 3e9    # data in *teacher* token ids
    python distill.py                          # resumes from out_dir/latest.pt

Vocabulary mapping (see ``distill_vocab.py``): the student vocab is the
teacher's BPE truncated to ids < 16384, so each teacher token is a fixed
sequence of student tokens and teacher boundaries are student boundaries.
For the student token at depth ``d`` inside teacher token ``t_j`` (pieces
``s_0 .. s_{L-1}``), the target comes from the teacher's next-token
distribution ``p`` at teacher position ``j - 1``:

* ``d = 0`` (a teacher boundary, ~91% of positions) -- exact marginal:
  ``q(s) = sum_t p(t) [pieces(t)[0] = s]``.
* ``d > 0`` (inside a split teacher token) -- the teacher's conditional
  over the next piece given the pieces already emitted:
  ``q(s) ~ sum_t p(t) [pieces(t)[:d] = s_0..s_{d-1}, pieces(t)[d] = s]``.
  The paths where a teacher token *ends* exactly after ``s_{d-1}`` and the
  next one starts with ``s`` would need another teacher pass on a
  non-canonical prefix; their share of the prefix's mass is dropped, the
  rest renormalised, and that share logged as ``dropped_mass``.

Loss: ``alpha * H(q, student) + (1 - alpha) * CE(hard label)`` (``H(q,.)`` is
KL up to the constant entropy of ``q``) + z-loss, and the same mix for the MTP
head, whose target at t+2 is simply ``q`` at the next position.
"""
import json
import math
import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tokenizers import Tokenizer

from model import build_document_mask, softcap
from pretrain import (A30_BF16_PEAK, KL_GRADES, Condition, PackedData, build_parser,
                      load_or_init, loader, lr_mult, model_summary, progress, sample,
                      save_checkpoint)


# -- vocabulary mapping ---------------------------------------------------------

class VocabMap:
    def __init__(self, path, device):
        d = torch.load(path)
        self.V, self.TV = d["student_vocab"], d["teacher_vocab"]
        self.pieces = d["pieces"].long().to(device)        # (TV, L)
        self.prefix = d["prefix"].long().to(device)        # (TV, L+1)
        self.lengths = d["lengths"].long().to(device)      # (TV,)
        self.L = self.pieces.size(1)
        # Per depth d: the teacher tokens with a piece at d, that piece, and
        # their prefix-node id at d.
        # Also the tokens that *end* at depth d (for the dropped-mass stat).
        self.ids, self.piece, self.pre, self.end_ids, self.end_pre = [], [], [], [], []
        for dd in range(self.L):
            ids = (self.lengths > dd).nonzero().squeeze(1)
            self.ids.append(ids)
            self.piece.append(self.pieces[ids, dd])
            self.pre.append(self.prefix[ids, dd])
            end = (self.lengths == dd).nonzero().squeeze(1)
            self.end_ids.append(end)
            self.end_pre.append(self.prefix[end, dd])

    def expand(self, t_ids, n):
        """Teacher ids (B, Tt) -> the first n+1 student tokens of the same text,
        with, per student token, its teacher position, depth, and teacher id."""
        B, Tt = t_ids.shape
        lens = self.lengths[t_ids]
        ends = lens.cumsum(1)
        pos = torch.arange(n + 1, device=t_ids.device).expand(B, -1).contiguous()
        tpos = torch.searchsorted(ends, pos, right=True)
        assert int(tpos.max()) < Tt, "teacher window too short for n_seq"
        depth = pos - (ends.gather(1, tpos) - lens.gather(1, tpos))
        tid = t_ids.gather(1, tpos)
        return self.pieces[tid, depth], tpos, depth, tid

    @torch.no_grad()
    def targets(self, probs, depth, tid):
        """Teacher next-token probs (R, TV) for R student targets at ``depth``
        inside teacher token ``tid`` -> (q (R, V) normalised, dropped (R,)):
        the share of the prefix's mass on teacher tokens that end exactly
        at the prefix (0 at teacher boundaries)."""
        R = probs.size(0)
        # Depth 0 for every row in one pass (ids < V map to themselves, the
        # rest to their first piece); interior rows are overwritten below.
        q = probs[:, :self.V].clone()
        q.index_add_(1, self.piece[0][self.V:], probs[:, self.V:])
        dropped = probs.new_zeros(R)
        for dd in range(1, self.L):
            rows = (depth == dd).nonzero().squeeze(1)
            if rows.numel() == 0:
                continue
            p = probs[rows]
            node = self.prefix[tid[rows], dd][:, None]
            vals = p[:, self.ids[dd]] * (self.pre[dd][None, :] == node)
            ended = (p[:, self.end_ids[dd]] * (self.end_pre[dd][None, :] == node)).sum(-1)
            kept = vals.sum(-1)
            dropped[rows] = ended / (ended + kept).clamp(min=1e-20)
            q[rows] = q.new_zeros(rows.numel(), self.V).index_add_(1, self.piece[dd], vals)
        return q / q.sum(-1, keepdim=True).clamp(min=1e-20), dropped


@torch.no_grad()
def teacher_targets(teacher, vm, t_ids, n, tau, row_chunk=2048):
    """Student batch + soft targets for one micro-batch of teacher windows.
    Returns x, y (B, n), q (B, n, V) bf16, w (B, n) (1 where a teacher
    distribution exists), and stats."""
    sid, tpos, depth, tid = vm.expand(t_ids, n)
    x, y = sid[:, :-1], sid[:, 1:]
    j, d, t_true = tpos[:, 1:], depth[:, 1:], tid[:, 1:]
    B = t_ids.size(0)
    # Only the teacher prefix that covers the n+1 student tokens is needed
    # (~n / 1.1 tokens); causal, so its logits are unchanged by truncation.
    # Rounded up to 128 to keep the number of compiled shapes small.
    need = min(t_ids.size(1), -(-(int(j.max()) + 1) // 128) * 128)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        H = teacher.model(input_ids=t_ids[:, :need]).last_hidden_state   # (B, need, D)
    rows_h = H[torch.arange(B, device=H.device)[:, None], (j - 1).clamp(min=0)]
    rows_h, d, t_true = rows_h.flatten(0, 1), d.flatten(), t_true.flatten()
    w = (j >= 1).flatten().float()
    q = torch.empty(rows_h.size(0), vm.V, device=H.device, dtype=torch.bfloat16)
    dropped = torch.empty(rows_h.size(0), device=H.device)
    ent = torch.empty_like(dropped)
    t_ce = torch.empty_like(dropped)
    y_flat = y.flatten()
    for i in range(0, rows_h.size(0), row_chunk):
        sl = slice(i, i + row_chunk)
        logits = teacher.lm_head(rows_h[sl]).float() / tau
        qq, dropped[sl] = vm.targets(logits.softmax(-1), d[sl], t_true[sl])
        ent[sl] = -(qq * qq.clamp(min=1e-30).log()).sum(-1)
        t_ce[sl] = -qq.gather(1, y_flat[sl, None]).squeeze(1).clamp(min=1e-30).log()
        q[sl] = qq.to(torch.bfloat16)
    interior = (d > 0) & (w > 0)
    stats = {"teacher_ce": (t_ce * w).sum() / w.sum(),          # teacher, student vocab
             "q_entropy": (ent * w).sum() / w.sum(),
             "interior_frac": interior.float().mean(),
             "dropped_mass": (dropped * interior).sum() / interior.sum().clamp(min=1)}
    return x, y, q.view(B, n, vm.V), w.view(B, n), stats


# -- student loss -----------------------------------------------------------------

def _kd_chunk(model, h, y, q, w):
    logits = softcap(model.head(h).float(), model.logit_softcap)
    logp = logits.log_softmax(-1)
    ce = -logp.gather(-1, y[..., None]).squeeze(-1)
    soft = -(q.float() * logp).sum(-1)
    z = torch.logsumexp(logits, -1) ** 2
    return torch.stack((ce.sum(), (soft * w).sum(), z.sum(), w.sum()))


def _kd_chunked(model, h, y, q, w):
    chunk = model.loss_chunk_size or h.size(1)
    tot = None
    for i in range(0, h.size(1), chunk):
        sl = slice(i, i + chunk)
        part = torch.utils.checkpoint.checkpoint(
            _kd_chunk, model, h[:, sl], y[:, sl], q[:, sl], w[:, sl], use_reentrant=False)
        tot = part if tot is None else tot + part
    n = y.numel()
    return tot[0] / n, tot[1] / tot[3].clamp(min=1), tot[2] / n


def distill_loss(model, x, y, q, w, attn_mask, alpha):
    h = model.forward_hidden(x, attn_mask=attn_mask)
    ce, soft, z = _kd_chunked(model, h, y, q, w)
    loss = alpha * soft + (1 - alpha) * ce + model.z_loss_weight * z
    ce2 = soft2 = torch.zeros_like(ce)
    if model.mtp is not None and model.mtp_loss_weight:
        cos, sin = model._get_rope(model.n_seq, h.device)
        m = None if attn_mask is None else attn_mask[:, :, :-1, :-1]
        h2 = model.mtp(h[:, :-1], model.embed(x[:, 1:]), cos, sin, m)
        ce2, soft2, _ = _kd_chunked(model, h2, y[:, 1:], q[:, 1:], w[:, 1:])
        loss = loss + model.mtp_loss_weight * (alpha * soft2 + (1 - alpha) * ce2)
    return loss, torch.stack((ce, soft, ce2, soft2)).detach()


# -- eval -------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, teacher, vm, val, n_tokens, micro_batch, eos_id, tau):
    """Student CE (hard labels) and KL(q || student) on evenly spaced val
    windows; also the teacher's own CE in the student vocabulary."""
    model.eval()
    n = model.n_seq
    n_win = max(1, min(val.n, int(n_tokens) // n))
    windows = np.linspace(0, val.n - 1, n_win).astype(np.int64)
    acc = torch.zeros(4, device="cuda")
    for i in range(0, n_win, micro_batch):
        tx, ty = val.get(windows[i:i + micro_batch])
        t_ids = torch.cat((tx, ty[:, -1:]), 1).cuda()
        x, y, q, w, st = teacher_targets(teacher, vm, t_ids, n, tau)
        mask = build_document_mask(x, eos_id) if eos_id is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(x, mode="hidden", attn_mask=mask)
            ce, soft, _ = _kd_chunked(model, h, y, q, w)
        acc += torch.stack((ce, soft - st["q_entropy"], st["teacher_ce"], torch.ones_like(ce)))
    model.train()
    ce, kl, t_ce, k = (acc / acc[3]).tolist()
    return {"val_ce": ce, "val_kl": kl, "teacher_ce": t_ce}


# -- main -------------------------------------------------------------------------

def main():
    ap = build_parser()
    # One step is 512 x 2048 student tokens (~80 s here), so log every step.
    ap.set_defaults(data="data/smollm_t49k", out_dir="checkpoints/distill100m",
                    train_tokens=2e9, eval_tokens=5e5, log_every=1, eval_every=50,
                    ckpt_every=50)
    ap.add_argument("--teacher", default="HuggingFaceTB/SmolLM2-1.7B")
    ap.add_argument("--vocab_map", default="tokenizer/student16k/vocab_map.pt")
    ap.add_argument("--student_tokenizer", default="tokenizer/student16k/tokenizer.json")
    ap.add_argument("--alpha", type=float, default=0.9, help="weight of the soft (teacher) loss")
    ap.add_argument("--tau", type=float, default=1.0, help="teacher softmax temperature")
    ap.add_argument("--teacher_compile", action="store_true",
                    help="torch.compile the teacher (no measurable gain on an A30)")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    device = "cuda"
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    warnings.filterwarnings("ignore", message=".*head_dim.*")
    os.makedirs(args.out_dir, exist_ok=True)

    meta = json.load(open(os.path.join(args.data, "meta.json")))
    vm = VocabMap(args.vocab_map, device)
    assert meta["vocab_size"] == vm.TV, "data must be in teacher token ids"
    tok = Tokenizer.from_file(args.student_tokenizer)
    eos_id = None if args.no_doc_mask else meta["eos_id"]
    assert eos_id is None or eos_id < vm.V

    from transformers import AutoModelForCausalLM
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    teacher.requires_grad_(False)
    assert teacher.config.vocab_size == vm.TV
    if args.teacher_compile:
        teacher.model = torch.compile(teacher.model)

    model, opt, step, cfg = load_or_init(args, vm.V, device)
    n = cfg.n_seq
    accum, rem = divmod(args.batch_size, args.micro_batch)
    assert rem == 0, "batch_size must be a multiple of micro_batch"
    tokens_per_step = args.batch_size * n          # student tokens
    total_steps = math.ceil(args.train_tokens / tokens_per_step)

    # Teacher windows of n+1 tokens always expand to >= n+1 student tokens.
    train = PackedData(args.data, "train", n, args.seed)
    val = PackedData(args.data, "val", n, args.seed)
    print(model_summary(model, cfg))
    print(f"doc masks {'on' if eos_id is not None else 'off'}")
    print(f"teacher: {args.teacher}, {sum(p.numel() for p in teacher.parameters())/1e9:.2f}B "
          f"params, vocab {vm.TV} -> student {vm.V}; alpha {args.alpha}, tau {args.tau}")
    print(f"data: train {train.n_tokens/1e9:.2f}B teacher tokens, val {val.n_tokens/1e6:.1f}M")
    print(f"plan: {total_steps} steps x {args.batch_size} seqs x {n} = "
          f"{total_steps*tokens_per_step/1e9:.2f}B student tokens, accum {accum}")
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({"model": cfg.to_dict(), "args": vars(args)}, f, indent=1)

    loss_fn = distill_loss if args.no_compile else torch.compile(distill_loss)
    batches = loader(train, step * args.batch_size, args.micro_batch, device)
    flops_per_step = model.estimate_flops_per_token() * tokens_per_step
    log = open(os.path.join(args.out_dir, "log.jsonl"), "a")
    last_step = min(total_steps, args.max_steps) if args.max_steps else total_steps

    model.train()
    t_last, t_teacher = time.time(), 0.0
    bar = progress(step, last_step)
    cond = Condition(KL_GRADES)
    post = ""                   # last logged metrics, shown on the bar
    while step < last_step:
        mult = lr_mult(step, total_steps, args.warmup_steps, args.decay_frac)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * mult
        parts = torch.zeros(4, device=device)
        stats = {}
        for i in range(accum):
            bar.set_postfix_str(f"{post}micro-batch {i + 1}/{accum}", refresh=False)
            tx, ty = next(batches)
            t0 = time.time()
            x, y, q, w, st = teacher_targets(teacher, vm, torch.cat((tx, ty[:, -1:]), 1),
                                             n, args.tau)
            t_teacher += time.time() - t0
            mask = build_document_mask(x, eos_id) if eos_id is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, p = loss_fn(model, x, y, q, w, mask, args.alpha)
            (loss / accum).backward()
            parts += p / accum
            for k, v in st.items():
                stats[k] = stats.get(k, 0) + v / accum
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        bar.update(1)

        if step % args.log_every == 0 or step == last_step:
            ce, soft, ce2, soft2 = parts.tolist()
            st = {k: v.item() for k, v in stats.items()}
            if not math.isfinite(soft):
                raise RuntimeError(f"non-finite loss at step {step}")
            dt = (time.time() - t_last) / args.log_every
            t_last = time.time()
            tag = cond(soft - st["q_entropy"])
            rec = {"step": step, "ce": ce, "kl": soft - st["q_entropy"], "condition": tag,
                   "mtp_ce": ce2,
                   "mtp_kl": soft2 - st["q_entropy"], **st, "lr": args.lr * mult,
                   "grad_norm": gnorm.item(), "tok_s": tokens_per_step / dt,
                   "teacher_frac": t_teacher / args.log_every / dt,
                   "mfu": flops_per_step / dt / A30_BF16_PEAK,
                   "tokens": step * tokens_per_step}
            t_teacher = 0.0
            post = (f"ce={ce:.3f} kl={rec['kl']:.3f} {rec['tok_s']/1e3:.1f}K tok/s "
                    f"model={tag} | ")
            bar.write(f"step {step:6d}/{total_steps} ce {ce:.4f} kl {rec['kl']:.4f} "
                  f"(teacher ce {st['teacher_ce']:.3f}) lr {rec['lr']:.2e} "
                  f"gnorm {rec['grad_norm']:.2f} {rec['tok_s']/1e3:.1f}K tok/s "
                  f"teacher {rec['teacher_frac']*100:.0f}% of time "
                  f"eta {(total_steps - step) * dt / 3600:.1f}h [{tag}]")
            log.write(json.dumps(rec) + "\n")
            log.flush()

        if step % args.eval_every == 0 or step == last_step:
            ev = evaluate(model, teacher, vm, val, args.eval_tokens, args.micro_batch,
                          eos_id, args.tau)
            bar.write(f"step {step:6d} val ce {ev['val_ce']:.4f} kl {ev['val_kl']:.4f} "
                      f"(teacher ce {ev['teacher_ce']:.4f})")
            for s in sample(model, tok, device, eos_id=tok.token_to_id("<|endoftext|>")):
                bar.write("   > " + s.replace("\n", " "))
            log.write(json.dumps({"step": step, **ev}) + "\n")
            log.flush()
            t_last = time.time()

        if step % args.ckpt_every == 0 or step == last_step:
            save_checkpoint(os.path.join(args.out_dir, f"step_{step:06d}.pt"), model, opt,
                            step, cfg, args, step * tokens_per_step, args.keep_ckpts)
            t_last = time.time()

    bar.close()
    print("done")


if __name__ == "__main__":
    main()
