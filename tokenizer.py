import heapq
from collections import Counter, defaultdict
import regex as re
from tqdm import tqdm

_PATTERN = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


class Tokenizer:
    SPECIAL_TOKENS = [
        '<|BOS|>', '<|EOS|>', '<|PAD|>', '<|UNK|>', '<|USER|>', '<|ASSISTANT|>'
    ]

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []
        self.merge_ranks = {}
        self.special_tokens = {t: 256 + i for i, t in enumerate(self.SPECIAL_TOKENS)}
        self.merge_id_offset = 256 + len(self.SPECIAL_TOKENS)
        self._cache = {}
        self._register_special_tokens()

    def _register_special_tokens(self):
        self.vocab.update({
            token_id: token.encode("utf-8")
            for token, token_id in self.special_tokens.items()
        })
        self.vocab_size = max(self.vocab_size, max(self.vocab) + 1)

    def _merge_ids(self, ids):
        if len(ids) < 2:
            return ids
        while True:
            best_pair, best_rank = None, None
            for i in range(len(ids) - 1):
                r = self.merge_ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_pair, best_rank = (ids[i], ids[i + 1]), r
            if best_pair is None:
                return ids
            id1, id2 = best_pair
            new_id = self.merge_id_offset + best_rank
            new_ids, i = [], 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == id1 and ids[i + 1] == id2:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids

    def train_from_text(self, text):
        for t in self.SPECIAL_TOKENS:
            text = text.replace(t, ' ')

        word_freq = Counter()
        for m in _PATTERN.finditer(text):
            word_freq[m.group(0).encode('utf-8')] += 1

        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []
        self.merge_ranks = {}
        self._cache = {}
        self._register_special_tokens()

        words, freqs = [], []
        for wbytes, f in word_freq.items():
            seq = list(wbytes)
            if len(seq) >= 2:
                words.append(seq)
                freqs.append(f)

        pair_counts = {}
        pair_positions = defaultdict(set)
        for w, seq in enumerate(words):
            f = freqs[w]
            for i in range(len(seq) - 1):
                p = (seq[i], seq[i + 1])
                pair_counts[p] = pair_counts.get(p, 0) + f
                pair_positions[p].add(w)

        heap = [(-c, p[0], p[1]) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        target = self.vocab_size - self.merge_id_offset
        pbar = tqdm(total=max(target, 0), desc="BPE training", unit="merge")

        while len(self.merges) < target:
            best_pair, best_freq = None, 0
            while heap:
                neg_c, a, b = heapq.heappop(heap)
                p = (a, b)
                cur = pair_counts.get(p, 0)
                if cur != -neg_c or cur <= 0:
                    continue
                best_pair, best_freq = p, cur
                break
            if best_pair is None or best_freq < 2:
                break

            id1, id2 = best_pair
            new_id = self.merge_id_offset + len(self.merges)
            self.vocab[new_id] = self.vocab[id1] + self.vocab[id2]
            self.merges.append((id1, id2))
            self.merge_ranks[(id1, id2)] = len(self.merges) - 1

            for w in list(pair_positions.get(best_pair, ())):
                seq, f = words[w], freqs[w]
                new_seq, i = [], 0
                while i < len(seq):
                    if i + 1 < len(seq) and seq[i] == id1 and seq[i + 1] == id2:
                        left = new_seq[-1] if new_seq else None
                        right = seq[i + 2] if i + 2 < len(seq) else None
                        if left is not None:
                            lp = (left, id1)
                            pair_counts[lp] = pair_counts.get(lp, 0) - f
                            heapq.heappush(heap, (-pair_counts[lp], lp[0], lp[1]))
                        if right is not None:
                            rp = (id2, right)
                            pair_counts[rp] = pair_counts.get(rp, 0) - f
                            heapq.heappush(heap, (-pair_counts[rp], rp[0], rp[1]))
                        new_seq.append(new_id)
                        if left is not None:
                            np1 = (left, new_id)
                            pair_counts[np1] = pair_counts.get(np1, 0) + f
                            pair_positions[np1].add(w)
                            heapq.heappush(heap, (-pair_counts[np1], np1[0], np1[1]))
                        if right is not None:
                            np2 = (new_id, right)
                            pair_counts[np2] = pair_counts.get(np2, 0) + f
                            pair_positions[np2].add(w)
                            heapq.heappush(heap, (-pair_counts[np2], np2[0], np2[1]))
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                words[w] = new_seq

            pair_counts[best_pair] = 0
            pair_positions.pop(best_pair, None)
            pbar.update(1)
        pbar.close()

    def train_from_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        self.train_from_text(text)

    def encode(self, text):
        atomic = self.special_tokens
        pattern = "(" + "|".join(re.escape(t) for t in sorted(atomic, key=len, reverse=True)) + ")"
        ids = []
        for chunk in re.split(pattern, text):
            if not chunk:
                continue
            if chunk in atomic:
                ids.append(atomic[chunk])
                continue
            for m in _PATTERN.finditer(chunk):
                wbytes = m.group(0).encode('utf-8')
                cached = self._cache.get(wbytes)
                if cached is None:
                    cached = self._merge_ids(list(wbytes))
                    self._cache[wbytes] = cached
                ids.extend(cached)
        return ids

    def decode(self, tokens):
        id_to_special = {v: k for k, v in self.special_tokens.items()}
        parts, buf = [], bytearray()
        for i in tokens:
            if i in id_to_special:
                if buf:
                    parts.append(bytes(buf).decode('utf-8', errors='replace'))
                    buf = bytearray()
                parts.append(id_to_special[i])
            else:
                b = self.vocab.get(i)
                if b is not None:
                    buf.extend(b)
        if buf:
            parts.append(bytes(buf).decode('utf-8', errors='replace'))
        return "".join(parts)

    def save(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"{self.vocab_size}\n")
            for id1, id2 in self.merges:
                f.write(f"{id1} {id2}\n")

    def load(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        self.vocab_size = int(lines[0])
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []
        self.merge_ranks = {}
        for idx, line in enumerate(lines[1:]):
            if not line:
                continue
            a, b = map(int, line.split())
            new_id = self.merge_id_offset + idx
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            self.merges.append((a, b))
            self.merge_ranks[(a, b)] = idx
        self._cache = {}
        self._register_special_tokens()


if __name__ == "__main__":
    tokenizer = Tokenizer(vocab_size=1000)
    tokenizer.train_from_file("dataset/sample.txt")

    text = """
Away, you fool! it more becomes a man
Than gilt his trophy: the breasts of Hecuba,
When she did suckle Hector, look'd not lovelier
Than Hector's forehead when it spit forth blood
At Grecian sword, contemning. Tell Valeria,
We are fit to bid her welcome.
    """
    tokens = tokenizer.encode(text)
    print(f"Encoded: {tokens}, length: {len(tokens)}, char-token compression: {len(text)/len(tokens):.2f}")
    decoded_text = tokenizer.decode(tokens)
    print(f"Decoded: {decoded_text}")
