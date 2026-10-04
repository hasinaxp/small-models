"""Student vocabulary for distillation: the teacher's BPE truncated to its
first ``V`` ids, plus the tables that map teacher distributions onto it.

SmolLM2's byte-level BPE numbers tokens in merge order (id = 252 + merge
rank), so keeping ids < V and merges[:V - 252] is itself a valid BPE.
BPE applies merges in increasing rank and a merge never creates a pair of
lower rank, so running the full teacher BPE passes through the student
tokenization first: every teacher token is a fixed concatenation of 1..L
student tokens, and teacher token boundaries are student token boundaries.

Tables (saved to ``tokenizer/<name>/vocab_map.pt``), for teacher id t:

* ``pieces[t, :len[t]]``  student ids that t splits into (-1 padded)
* ``prefix[t, d]``        id of the trie node for pieces[t, :d] (d = 0..L);
                          equal ids <=> equal piece prefixes. -1 if d > len.

    python distill_vocab.py --vocab 16384
"""
import argparse
import copy
import glob
import json
import os

import torch
from tokenizers import Tokenizer

TEACHER_TOKENIZER = "tokenizer/smollm2/tokenizer.json"


def build_student(teacher_json, V):
    j = json.load(open(teacher_json))
    m = j["model"]
    assert m["type"] == "BPE"
    merges = m["merges"]
    mk = lambda x: "".join(x) if isinstance(x, list) else x.replace(" ", "")
    ids = [m["vocab"][mk(x)] for x in merges]
    n_base = ids[0]
    assert ids == list(range(n_base, n_base + len(ids))), "ids must follow merge rank"
    s = copy.deepcopy(j)
    s["model"]["vocab"] = {k: v for k, v in m["vocab"].items() if v < V}
    s["model"]["merges"] = merges[:V - n_base]
    s["added_tokens"] = [a for a in j["added_tokens"] if a["id"] < V]
    return Tokenizer.from_str(json.dumps(s)), s


def build_tables(teacher, student, V):
    TV = teacher.get_vocab_size()
    dec = []
    for t in range(TV):
        if t < V:
            dec.append([t])
        else:   # BPE the raw byte-level string; no decode (partial UTF-8 safe)
            dec.append([x.id for x in student.model.tokenize(teacher.id_to_token(t))])
    L = max(map(len, dec))
    pieces = torch.full((TV, L), -1, dtype=torch.int32)
    prefix = torch.full((TV, L + 1), -1, dtype=torch.int32)
    lengths = torch.tensor([len(d) for d in dec], dtype=torch.int32)
    nodes = {(): 0}
    for t, d in enumerate(dec):
        pieces[t, :len(d)] = torch.tensor(d, dtype=torch.int32)
        for k in range(len(d) + 1):
            prefix[t, k] = nodes.setdefault(tuple(d[:k]), len(nodes))
    return pieces, prefix, lengths


def check(teacher, student, pieces, lengths, n_docs=2000):
    import pyarrow.parquet as pq
    files = sorted(glob.glob("data/raw/*/*.parquet"))
    if not files:
        print("no raw parquet in data/raw; skipping corpus check")
        return
    texts = []
    for f in files:
        texts += pq.ParquetFile(f).read_row_group(0, columns=["text"]).column(0).to_pylist()[:n_docs]
    bad = n_t = n_s = 0
    for t in texts:
        T = teacher.encode(t).ids
        S = student.encode(t).ids
        cat = [p for tid in T for p in pieces[tid, :lengths[tid]].tolist()]
        bad += cat != S
        n_t += len(T)
        n_s += len(S)
    print(f"exactness: {len(texts) - bad}/{len(texts)} docs identical; "
          f"{n_s / n_t:.3f} student tokens per teacher token")
    assert bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_tokenizer", default=TEACHER_TOKENIZER)
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"tokenizer/student{args.vocab // 1024}k"
    os.makedirs(out, exist_ok=True)

    teacher = Tokenizer.from_file(args.teacher_tokenizer)
    student, sjson = build_student(args.teacher_tokenizer, args.vocab)
    with open(os.path.join(out, "tokenizer.json"), "w") as f:
        json.dump(sjson, f, ensure_ascii=False)
    pieces, prefix, lengths = build_tables(teacher, student, args.vocab)
    torch.save({"pieces": pieces, "prefix": prefix, "lengths": lengths,
                "student_vocab": args.vocab, "teacher_vocab": teacher.get_vocab_size(),
                "teacher_tokenizer": args.teacher_tokenizer},
               os.path.join(out, "vocab_map.pt"))
    hist = torch.bincount(lengths.long()).tolist()
    print(f"student vocab {student.get_vocab_size()} -> {out}; teacher tokens by piece "
          f"count: { {k: v for k, v in enumerate(hist) if v} }")
    check(teacher, student, pieces, lengths)


if __name__ == "__main__":
    main()
