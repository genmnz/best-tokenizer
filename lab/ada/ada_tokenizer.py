"""
Ada — best-of-all-worlds byte tokenizer (v0, statistical, CPU-only, pure stdlib).

Fuses, at vocabulary-construction time (no neural net, no GPU):
  * SuperBPE (arXiv:2503.13423) — two-stage: subwords, then cross-whitespace "superwords".
  * BLT      (arXiv:2412.09871) — merge predictable spans, keep surprising boundaries,
             via branching-entropy H(R|L) computed from corpus statistics.
  * Length-MAX (arXiv:2511.20849) — score merges by realised token length, not raw frequency.

Every knob is a field on AdaConfig (lab rule 10). Base units are raw bytes (256), so the
tokenizer is language-agnostic and robust to typos/scripts (BLT / EvaByte property).

See ../../FINDINGS.md §1,§4 and ../../ADA.md for the full derivation and citations.
"""
from __future__ import annotations

import heapq
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# A pre-token is: an optional single leading space + a run of non-spaces, OR a run of spaces.
# Operates on the latin-1 view of the bytes so 1 char == 1 byte (no unicode dependency).
_PRETOK = re.compile(r" ?[^ ]+| +")

BASE_VOCAB = 256


@dataclass
class AdaConfig:
    """All Ada knobs. No magic numbers live in the code."""
    vocab_size: int = 1024            # total ids incl. the 256 base bytes
    superword_start_frac: float = 0.75  # fraction of *learned merges* before whitespace is dropped
    entropy_lambda: float = 1.0       # BLT guard strength: reward merging predictable pairs
    length_weight: float = 0.05       # Length-MAX weight (only used if enable_length_bonus)
    enable_superword: bool = True     # off => plain 2-stage-less BPE baseline (whitespace kept)
    enable_entropy_guard: bool = True
    # Length-bonus is OFF by default: it monotonically HURT held-out compression in the
    # ablation sweep (FINDINGS §5). Kept configurable for natural-text re-evaluation.
    enable_length_bonus: bool = False
    min_pair_count: int = 2           # don't learn merges seen fewer times than this
    seed: int = 0

    def merges_budget(self) -> int:
        return max(0, self.vocab_size - BASE_VOCAB)


def pretokenize(text: str) -> List[str]:
    """Split into whitespace-respecting chunks (stage-1 constraint)."""
    return _PRETOK.findall(text)


def _to_symbols(chunk: str) -> List[int]:
    return list(chunk.encode("utf-8", errors="replace"))


class AdaTokenizer:
    def __init__(self, config: Optional[AdaConfig] = None):
        self.config = config or AdaConfig()
        # merges in learned order: list of ((a,b) -> new_id)
        self.merges: List[Tuple[int, int]] = []
        self.ranks: Dict[Tuple[int, int], int] = {}
        # id -> raw bytes it expands to (for decode + length scoring)
        self.id_to_bytes: Dict[int, bytes] = {i: bytes([i]) for i in range(BASE_VOCAB)}
        self.transition_merge: Optional[int] = None  # merge index where superwords began

    # ---------------------------------------------------------------- training
    def train(self, corpus: List[str], verbose: bool = False) -> "AdaTokenizer":
        cfg = self.config
        budget = cfg.merges_budget()
        stage1_target = int(round(budget * cfg.superword_start_frac)) if cfg.enable_superword else budget

        # STAGE 1 units: pre-token chunks, deduplicated with counts (whitespace enforced).
        stage1_units = Counter()
        for line in corpus:
            for chunk in pretokenize(line):
                stage1_units[chunk] += 1
        units: List[List[int]] = []
        counts: List[int] = []
        for chunk, c in stage1_units.items():
            units.append(_to_symbols(chunk))
            counts.append(c)

        next_id = BASE_VOCAB
        t0 = time.time()
        stage = 1
        while len(self.merges) < budget:
            # transition into superword stage: rebuild units as whole lines so merges cross spaces
            if stage == 1 and cfg.enable_superword and len(self.merges) >= stage1_target:
                units, counts = self._rebuild_line_units(corpus)
                self.transition_merge = len(self.merges)
                stage = 2
                if verbose:
                    print(f"  [stage2 superwords @ merge {self.transition_merge}]")

            pair = self._best_pair(units, counts)
            if pair is None:
                break
            new_id = next_id
            next_id += 1
            self.id_to_bytes[new_id] = self.id_to_bytes[pair[0]] + self.id_to_bytes[pair[1]]
            self.ranks[pair] = len(self.merges)
            self.merges.append(pair)
            self._apply_merge(units, pair, new_id)
            if verbose and len(self.merges) % 200 == 0:
                print(f"  merges={len(self.merges)}  vocab={next_id}  {time.time()-t0:.1f}s")

        if verbose:
            print(f"  trained {len(self.merges)} merges, vocab={next_id} in {time.time()-t0:.2f}s")
        return self

    def _rebuild_line_units(self, corpus: List[str]) -> Tuple[List[List[int]], List[int]]:
        """Encode each line with current merges, concatenating chunks so spaces are crossable."""
        line_units = Counter()
        for line in corpus:
            seq: List[int] = []
            for chunk in pretokenize(line):
                seq.extend(self._encode_symbols(_to_symbols(chunk)))
            line_units[tuple(seq)] += 1
        units = [list(seq) for seq in line_units]
        counts = list(line_units.values())
        return units, counts

    def _best_pair(self, units: List[List[int]], counts: List[int]) -> Optional[Tuple[int, int]]:
        cfg = self.config
        pair_counts: Counter = Counter()
        for seq, c in zip(units, counts):
            for a, b in zip(seq, seq[1:]):
                pair_counts[(a, b)] += c
        if not pair_counts:
            return None

        # branching entropy H(R|L): how predictable is the right symbol given the left one.
        if cfg.enable_entropy_guard:
            left_right: Dict[int, Counter] = defaultdict(Counter)
            for (a, b), n in pair_counts.items():
                left_right[a][b] += n
            branching_H: Dict[int, float] = {}
            for a, rights in left_right.items():
                tot = sum(rights.values())
                H = 0.0
                for n in rights.values():
                    p = n / tot
                    H -= p * math.log(p)
                Hmax = math.log(len(rights)) if len(rights) > 1 else 1.0
                branching_H[a] = H / Hmax  # normalised predictability in [0,1]; 0 == fully predictable
        else:
            branching_H = {}

        best_pair, best_score = None, -1.0
        for pair, n in pair_counts.items():
            if n < cfg.min_pair_count:
                continue
            score = float(n)
            if cfg.enable_entropy_guard:
                predictability = 1.0 - branching_H.get(pair[0], 1.0)  # high => safe to merge
                score *= (1.0 + cfg.entropy_lambda * predictability)
            if cfg.enable_length_bonus:
                blen = len(self.id_to_bytes[pair[0]]) + len(self.id_to_bytes[pair[1]])
                score *= (1.0 + cfg.length_weight * blen)
            if score > best_score or (score == best_score and (best_pair is None or pair < best_pair)):
                best_pair, best_score = pair, score
        return best_pair

    @staticmethod
    def _apply_merge(units: List[List[int]], pair: Tuple[int, int], new_id: int) -> None:
        a, b = pair
        for seq in units:
            i = 0
            out = []
            n = len(seq)
            while i < n:
                if i < n - 1 and seq[i] == a and seq[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            seq[:] = out

    # ---------------------------------------------------------------- encoding
    def _encode_symbols(self, symbols: List[int]) -> List[int]:
        """Rank-greedy BPE merge, O(n log n) via a heap + doubly-linked list.

        Always merges the globally lowest-rank adjacent pair first (identical result to the
        naive scan, but without the O(n^2) rescans). Shared by train stage-2 and encode.
        """
        n = len(symbols)
        if n < 2:
            return list(symbols)
        ranks = self.ranks
        sym = list(symbols)
        nxt = list(range(1, n + 1)); nxt[-1] = -1
        prev = list(range(-1, n - 1))
        alive = [True] * n
        heap: List[Tuple[int, int, int]] = []
        for i in range(n - 1):
            r = ranks.get((sym[i], sym[i + 1]))
            if r is not None:
                heap.append((r, i, i + 1))
        heapq.heapify(heap)
        while heap:
            r, i, j = heapq.heappop(heap)
            if not alive[i] or not alive[j] or nxt[i] != j:
                continue
            if ranks.get((sym[i], sym[j])) != r:
                continue
            sym[i] = BASE_VOCAB + r          # merge rank r == id assigned to this pair
            alive[j] = False
            k = nxt[j]
            nxt[i] = k
            if k != -1:
                prev[k] = i
            p = prev[i]
            if p != -1:
                rr = ranks.get((sym[p], sym[i]))
                if rr is not None:
                    heapq.heappush(heap, (rr, p, i))
            if k != -1:
                rr = ranks.get((sym[i], sym[k]))
                if rr is not None:
                    heapq.heappush(heap, (rr, i, k))
        out: List[int] = []
        i = 0
        while i != -1:
            out.append(sym[i])
            i = nxt[i]
        return out

    def encode(self, text: str) -> List[int]:
        """Encode text -> token ids. Superword merges (if learned) fire across spaces."""
        if self.config.enable_superword and self.transition_merge is not None:
            # superwords may cross pre-token boundaries: encode the whole line at once
            return self._encode_symbols(_to_symbols(text))
        # baseline / stage-1-only: keep pre-token boundaries
        out: List[int] = []
        for chunk in pretokenize(text):
            out.extend(self._encode_symbols(_to_symbols(chunk)))
        return out

    def decode(self, ids: List[int]) -> str:
        raw = b"".join(self.id_to_bytes[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    # ---------------------------------------------------------------- io + stats
    @property
    def vocab_size(self) -> int:
        return BASE_VOCAB + len(self.merges)

    def save(self, path: str) -> None:
        payload = {
            "config": asdict(self.config),
            "merges": [[a, b] for (a, b) in self.merges],
            "transition_merge": self.transition_merge,
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "AdaTokenizer":
        with open(path) as f:
            payload = json.load(f)
        tok = cls(AdaConfig(**payload["config"]))
        for a, b in payload["merges"]:
            new_id = BASE_VOCAB + len(tok.merges)
            tok.id_to_bytes[new_id] = tok.id_to_bytes[a] + tok.id_to_bytes[b]
            tok.ranks[(a, b)] = len(tok.merges)
            tok.merges.append((a, b))
        tok.transition_merge = payload["transition_merge"]
        return tok
