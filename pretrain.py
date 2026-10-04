"""Pretrain the hybrid small LM (``model.py``) on the tokenized smollm-corpus.

    python pretrain.py                        # defaults below, resumes from out_dir/latest.pt
    python pretrain.py --n_dim 1024 --repeat_times 3 --train_tokens 2e9

* Every ``ModelConfig`` field is a flag (``--mixer_dim``, ``--emb_dim``,
  ``--repeat_start`` ...); ``MODEL_DEFAULTS`` below is the ~100M setup, any
  field not listed there keeps its ``model.py`` default.
* The default model uses the whole design: the ``DS`` hybrid with a 512-token
  window trained on 2048-token sequences (so the DeltaNet memory has to carry
  everything past the window), a wide residual stream with narrower mixer /
  FFN widths, a factorised embedding, and looped middle blocks.
* Packed documents are separated with document masks: attention never
  crosses an EOS and the DeltaNet state and short conv reset at document
  starts (``--no_doc_mask`` turns this off).
* bf16 autocast over fp32 master weights, ``torch.compile``d model.
* Global batch = ``batch_size`` sequences (default 512) reached by gradient
  accumulation over ``micro_batch`` sequences.
* Muon + AdamW (``optim.py``), warmup-stable-decay LR schedule.
* Checkpoints (model, optimizer, step) every ``ckpt_every`` steps and at the
  end; rerunning the same command resumes exactly (the data order is a pure
  function of the step).
"""
import argparse
import dataclasses
import glob
import json
import math
import os
import queue
import threading
import time
import warnings

import numpy as np
import torch

from model import ModelConfig, Transformer, build_document_mask
from optim import build_optimizer

A30_BF16_PEAK = 165e12

# ~100M unique params (96M without the training-only MTP head), 18 executed
# layers from 14 unique blocks.
MODEL_DEFAULTS = dict(
    n_seq=2048,
    n_dim=896,             # residual width
    mixer_dim=512,         # attention / DeltaNet inner width: 8 heads x 64
    n_head=8,
    ffn_hidden=1792,
    emb_dim=256,           # factorised 16k x 256 table
    n_layer=14,
    repeat_start=5, repeat_end=9, repeat_times=2,
    mixer_pattern="DS",
    window_size=512,
)


def _model_flags(ap):
    """One flag per ModelConfig field (vocab_size comes from the data)."""
    g = ap.add_argument_group("model (model.ModelConfig)")
    for f in dataclasses.fields(ModelConfig):
        if f.name == "vocab_size":
            continue
        default = MODEL_DEFAULTS.get(f.name, f.default)
        if isinstance(f.default, bool):
            g.add_argument(f"--{f.name}", default=default,
                           action=argparse.BooleanOptionalAction)
        else:
            typ = type(f.default) if f.default is not None else int
            g.add_argument(f"--{f.name}", type=typ, default=default)


def build_parser():
    """Shared by pretrain.py and distill.py."""
    ap = argparse.ArgumentParser()
    _model_flags(ap)
    # data
    ap.add_argument("--data", default="data/smollm")
    ap.add_argument("--train_tokens", type=float, default=5e9)
    ap.add_argument("--no_doc_mask", action="store_true")
    # optimisation
    ap.add_argument("--batch_size", type=int, default=512, help="sequences per optimizer step")
    ap.add_argument("--micro_batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--decay_frac", type=float, default=0.2,
                    help="final fraction of steps with linear decay to 0 (WSD)")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    # infra
    ap.add_argument("--out_dir", default="checkpoints/hybrid100m")
    ap.add_argument("--ckpt_every", type=int, default=250)
    ap.add_argument("--keep_ckpts", type=int, default=3)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_tokens", type=float, default=4e6)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--max_steps", type=int, default=0, help="stop early (smoke tests)")
    ap.add_argument("--init_from", default="", help="'' = resume from out_dir/latest.pt if present")
    return ap


def model_config(args, vocab_size):
    return ModelConfig(vocab_size=vocab_size, **{
        f.name: getattr(args, f.name) for f in dataclasses.fields(ModelConfig)
        if f.name != "vocab_size"})


def load_or_init(args, vocab_size, device):
    """(model, optimizer, step, cfg): resumes from ``init_from`` or
    ``out_dir/latest.pt`` when present (the checkpoint's config wins)."""
    resume = args.init_from or os.path.join(args.out_dir, "latest.pt")
    ckpt = torch.load(resume, map_location="cpu", weights_only=False) \
        if os.path.exists(resume) else None
    cfg = ModelConfig(**ckpt["config"]) if ckpt else model_config(args, vocab_size)
    model = Transformer.from_config(cfg).to(device)
    opt = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    step = 0
    if ckpt:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"resumed from {resume} at step {step}")
    for g in opt.param_groups:
        g["base_lr"] = args.lr
    return model, opt, step, cfg


def model_summary(model, cfg):
    """Parameter breakdown and the executed layer stack, as printable text."""
    count = lambda m: sum(p.numel() for p in m.parameters()) if m is not None else 0
    M = lambda n: f"{n / 1e6:7.2f}M" if n >= 1e5 else f"{n / 1e3:7.1f}K"
    total = count(model)
    emb = model.l_embeddings.weight.numel() + count(model.emb_proj) + count(model.logit_down)
    blocks, mtp = count(model.blocks), count(model.mtp)
    other = total - emb - blocks - mtp
    kinds = {"D": "Gated DeltaNet", "S": f"sliding attn w={cfg.window_size}", "G": "global attn"}
    head_dim = model.head_dim
    lines = [
        "=" * 78,
        f"model  {cfg.mixer_pattern} hybrid, {cfg.n_layer} unique blocks -> "
        f"{model.n_executed_layer} executed layers, seq {cfg.n_seq}, vocab {cfg.vocab_size}",
        f"widths residual {cfg.n_dim} | mixer {model.mixer_dim} = {cfg.n_head} heads x {head_dim} "
        f"(kv heads {model.n_kv_head}) | ffn {model.ffn_hidden} | embedding {model.emb_dim}"
        + (" (factorised)" if model.emb_proj is not None else ""),
        f"extras MTP depth {cfg.mtp_depth}, U-Net skips {cfg.unet_skips}, value residual "
        f"{cfg.value_residual}, sink {cfg.attention_sink}, softcap {cfg.logit_softcap}, "
        f"tied head {cfg.tie_embeddings}",
        "-" * 78,
        f"  embedding (+ projections)   {M(emb)}",
        f"  blocks ({cfg.n_layer} unique)          {M(blocks)}",
        f"  norms / lambdas / skips     {M(other)}",
        f"  = inference parameters      {M(total - mtp)}",
        f"  MTP head (training only)    {M(mtp)}",
        f"  = training parameters       {M(total)}",
        f"  compute: {model.estimate_flops_per_token() / 1e6:.0f} MFLOPs/token (training, "
        f"executed layers)",
        "-" * 78,
        f"  {'layer':>5} {'block':>5}  {'mixer':22s} {'mixer':>9} {'ffn':>9}  notes",
    ]
    n_enc = model.n_encoder
    seen = {}
    for slot, li in enumerate(model.layer_schedule):
        b = model.blocks[li]
        seen[li] = seen.get(li, 0) + 1
        shared = seen[li] > 1
        notes = []
        if model.layer_schedule.count(li) > 1:
            notes.append(f"loop pass {seen[li]}" + (" (shared weights)" if shared else ""))
        if n_enc and n_enc <= slot < 2 * n_enc:
            notes.append(f"+skip from layer {2 * n_enc - 1 - slot}")
        mix = "     --  " if shared else M(count(b.mixer))
        ffn = "     --  " if shared else M(count(b.ffn))
        lines.append(f"  {slot:>5} {li:>5}  {kinds[b.kind]:22s} {mix:>9} {ffn:>9}  "
                     + ", ".join(notes))
    if model.mtp is not None:
        lines.append(f"  {'mtp':>5} {'':>5}  {kinds['S']:22s} {M(count(model.mtp)):>9} "
                     f"{'':>9}  predicts t+2 (training only)")
    lines.append("=" * 78)
    return "\n".join(lines)


# Model-condition tag on the progress bar: (upper bound, label), checked in
# order on a smoothed metric. Heuristics for a ~100M model with a 16k vocab
# on fineweb-edu/cosmopedia; adjust to taste.
CE_GRADES = ((3.0, "excellent"), (3.4, "good"), (4.0, "fair"), (5.5, "poor"),
             (float("inf"), "bad"))           # hard-label CE, nats/token
KL_GRADES = ((0.6, "excellent"), (0.9, "good"), (1.5, "fair"), (3.0, "poor"),
             (float("inf"), "bad"))           # KL(teacher || student), nats/token


class Condition:
    """EMA of a loss-like metric -> a bad..excellent label."""

    def __init__(self, grades, beta=0.9):
        self.grades, self.beta, self.ema = grades, beta, None

    def __call__(self, value):
        if not math.isfinite(value):
            return "bad (non-finite)"
        self.ema = value if self.ema is None else self.beta * self.ema + (1 - self.beta) * value
        return next(label for bound, label in self.grades if self.ema < bound)


def progress(step, total):
    """Optimizer-step progress bar. Log lines go through ``tqdm.write`` so
    they don't break it; when output is a file (nohup) it redraws rarely."""
    import sys
    from tqdm import tqdm
    return tqdm(total=total, initial=step, unit="step", dynamic_ncols=True,
                smoothing=0.05, mininterval=0.5 if sys.stderr.isatty() else 30)


def get_args():
    return build_parser().parse_args()


# -- data --------------------------------------------------------------------

class PackedData:
    """Fixed (T+1)-token windows over all shards of a split. Window order is a
    seeded permutation per epoch, so batch k is the same on every run."""

    def __init__(self, data_dir, split, seq_len, seed):
        files = sorted(glob.glob(os.path.join(data_dir, f"{split}_*.bin")))
        if not files:
            raise FileNotFoundError(f"no {split}_*.bin in {data_dir}; run prepare_data.py")
        self.maps = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]
        self.T = seq_len
        shard, off = [], []
        for s, m in enumerate(self.maps):
            n = (len(m) - 1) // seq_len
            shard.append(np.full(n, s, dtype=np.int32))
            off.append(np.arange(n, dtype=np.int64) * seq_len)
        self.shard, self.off = np.concatenate(shard), np.concatenate(off)
        self.n = len(self.shard)
        self.n_tokens = sum(len(m) for m in self.maps)
        self.seed = seed
        self._perm_epoch, self._perm = -1, None

    def window_ids(self, start, count):
        out = np.empty(count, dtype=np.int64)
        for j in range(count):
            i = start + j
            epoch, k = divmod(i, self.n)
            if epoch != self._perm_epoch:
                self._perm = np.random.default_rng(self.seed + epoch).permutation(self.n)
                self._perm_epoch = epoch
            out[j] = self._perm[k]
        return out

    def get(self, ids):
        x = np.stack([self.maps[self.shard[w]][self.off[w]:self.off[w] + self.T + 1]
                      for w in ids]).astype(np.int64)
        x = torch.from_numpy(x).pin_memory()
        return x[:, :-1], x[:, 1:]


def loader(data, start_window, micro_batch, device):
    """Background thread: pinned micro-batches, copied asynchronously."""
    q = queue.Queue(maxsize=8)

    def work():
        w = start_window
        while True:
            q.put(data.get(data.window_ids(w, micro_batch)))
            w += micro_batch

    threading.Thread(target=work, daemon=True).start()
    while True:
        x, y = q.get()
        yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# -- schedule / eval / checkpoint --------------------------------------------

def lr_mult(step, total, warmup, decay_frac):
    if step < warmup:
        return (step + 1) / warmup
    decay_start = int(total * (1 - decay_frac))
    if step < decay_start:
        return 1.0
    return max(0.0, (total - step) / max(1, total - decay_start))


@torch.no_grad()
def evaluate(model, val, n_tokens, micro_batch, device, eos_id=None):
    """Plain next-token CE (no MTP, no z-loss) on a fixed slice of val, with
    the same document masking as training."""
    model.eval()
    n_win = max(1, min(val.n, int(n_tokens) // val.T))
    # Evenly spaced over all val shards, so every source is represented.
    windows = np.linspace(0, val.n - 1, n_win).astype(np.int64)
    tot, cnt = 0.0, 0
    for i in range(0, n_win, micro_batch):
        ids = windows[i:i + micro_batch]
        x, y = val.get(ids)
        x, y = x.to(device), y.to(device)
        mask = build_document_mask(x, eos_id) if eos_id is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(x, mode="hidden", attn_mask=mask)
            ce, _ = model._chunked_ce(h, y)
        tot += ce.item() * y.numel()
        cnt += y.numel()
    model.train()
    return tot / cnt


@torch.no_grad()
def sample(model, tok, device, prompts=("Once upon a time,", "The theory of evolution",
                                        "The capital of France is"), eos_id=None):
    if eos_id is None:
        eos_id = tok.token_to_id("<|EOS|>")
    model.eval()
    outs = []
    for p in prompts:
        idx = torch.tensor([tok.encode(p).ids], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(idx, max_count=48, temperature=0.8, top_k=50,
                                 top_p=0.95, eos_token_id=eos_id)
        outs.append(tok.decode(out[0].tolist()))
    model.train()
    return outs


def save_checkpoint(path, model, opt, step, cfg, args, tokens, keep):
    ckpt = {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
            "config": cfg.to_dict(), "args": vars(args), "tokens": tokens}
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    latest = os.path.join(os.path.dirname(path), "latest.pt")
    if os.path.lexists(latest):
        os.remove(latest)
    os.symlink(os.path.basename(path), latest)
    old = sorted(glob.glob(os.path.join(os.path.dirname(path), "step_*.pt")))
    for f in old[:-keep]:
        os.remove(f)


# -- main ----------------------------------------------------------------------

def main():
    args = get_args()
    assert torch.cuda.is_available()
    device = "cuda"
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    warnings.filterwarnings("ignore", message=".*head_dim.*")
    os.makedirs(args.out_dir, exist_ok=True)

    meta = json.load(open(os.path.join(args.data, "meta.json")))
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(meta["tokenizer"])

    accum, rem = divmod(args.batch_size, args.micro_batch)
    assert rem == 0, "batch_size must be a multiple of micro_batch"
    model, opt, step, cfg = load_or_init(args, meta["vocab_size"], device)
    tokens_per_step = args.batch_size * cfg.n_seq
    total_steps = math.ceil(args.train_tokens / tokens_per_step)
    train = PackedData(args.data, "train", cfg.n_seq, args.seed)
    val = PackedData(args.data, "val", cfg.n_seq, args.seed)
    eos_id = None if args.no_doc_mask else meta["eos_id"]
    print(model_summary(model, cfg))
    print(f"doc masks {'on' if eos_id is not None else 'off'}")
    print(f"data: train {train.n_tokens/1e9:.2f}B tokens, val {val.n_tokens/1e6:.1f}M tokens")
    print(f"plan: {total_steps} steps x {args.batch_size} seqs x {cfg.n_seq} = "
          f"{total_steps*tokens_per_step/1e9:.2f}B tokens "
          f"({total_steps*tokens_per_step/train.n_tokens:.2f} epochs), accum {accum}")
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({"model": cfg.to_dict(), "args": vars(args)}, f, indent=1)

    fwd = model if args.no_compile else torch.compile(model)
    batches = loader(train, step * args.batch_size, args.micro_batch, device)
    flops_per_step = model.estimate_flops_per_token() * tokens_per_step
    log = open(os.path.join(args.out_dir, "log.jsonl"), "a")
    last_step = min(total_steps, args.max_steps) if args.max_steps else total_steps

    model.train()
    t_last = time.time()
    bar = progress(step, last_step)
    cond = Condition(CE_GRADES)
    post = ""                   # last logged metrics, shown on the bar
    while step < last_step:
        mult = lr_mult(step, total_steps, args.warmup_steps, args.decay_frac)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * mult
        loss_acc = torch.zeros((), device=device)
        ce_acc = torch.zeros((), device=device)
        for i in range(accum):
            bar.set_postfix_str(f"{post}micro-batch {i + 1}/{accum}", refresh=False)
            x, y = next(batches)
            mask = build_document_mask(x, eos_id) if eos_id is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = fwd(x, targets=y, mode="loss", attn_mask=mask)
            (loss / accum).backward()
            loss_acc += loss.detach()
            ce_acc += model.last_metrics["ce"]
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        bar.update(1)

        if step % args.log_every == 0 or step == last_step:
            loss_v = (loss_acc / accum).item()
            if not math.isfinite(loss_v):
                raise RuntimeError(f"non-finite loss at step {step}")
            dt = (time.time() - t_last) / args.log_every
            t_last = time.time()
            tps = tokens_per_step / dt
            rec = {"step": step, "loss": loss_v, "ce": ce_v, "condition": tag, "lr": args.lr * mult,
                   "grad_norm": gnorm.item(), "tok_s": tps,
                   "mfu": flops_per_step / dt / A30_BF16_PEAK,
                   "tokens": step * tokens_per_step}
            eta = (total_steps - step) * dt / 3600
            ce_v = (ce_acc / accum).item()
            tag = cond(ce_v)
            post = f"ce={ce_v:.3f} {tps/1e3:.1f}K tok/s model={tag} | "
            bar.write(f"step {step:6d}/{total_steps} loss {loss_v:.4f} lr {rec['lr']:.2e} "
                  f"gnorm {rec['grad_norm']:.2f} {tps/1e3:.1f}K tok/s mfu {rec['mfu']*100:.1f}% "
                  f"eta {eta:.1f}h [{tag}]")
            log.write(json.dumps(rec) + "\n")
            log.flush()

        if step % args.eval_every == 0 or step == last_step:
            vl = evaluate(model, val, args.eval_tokens, args.micro_batch, device, eos_id)
            bar.write(f"step {step:6d} val CE {vl:.4f} (ppl {math.exp(vl):.1f})")
            for s in sample(model, tok, device):
                bar.write("   > " + s.replace("\n", " "))
            log.write(json.dumps({"step": step, "val_loss": vl}) + "\n")
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
