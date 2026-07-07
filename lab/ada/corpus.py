"""
Synthetic multi-domain corpus generator (lab rule 11: no dataset downloads on this box).

Produces repetitive-but-varied text across a few "domains" (english prose, code, log lines,
urls, numbers) so that both subword and *superword* structure exists to be discovered. The
generator is seeded and deterministic. Returns a list of lines.
"""
from __future__ import annotations

import random
from typing import List

_WORDS = (
    "the of and to in a is that it for on with as by at from this be are was were will can "
    "model token byte patch entropy merge vocab compression tokenizer boundary dynamic chunk "
    "learned neural sequence encoder decoder attention latent quantize codebook language data "
    "we show that our method improves over the baseline by a large margin on every benchmark"
).split()

_PHRASES = [
    "by the way", "of the", "in the", "to the", "on the other hand", "as well as",
    "in order to", "such that", "with respect to", "state of the art", "best of all worlds",
]

_CODE = [
    "def {f}(x): return x + {n}",
    "for i in range({n}): total += arr[i]",
    "if x is None: raise ValueError('missing {f}')",
    "class {C}: pass",
    "import numpy as np  # {f}",
    "result = model.encode(text, vocab_size={n})",
]

_LOG = [
    "[INFO] {ts} request id={n} status=200 path=/api/{f}",
    "[WARN] {ts} retry attempt {n} for {f} timeout",
    "[ERROR] {ts} failed to open {f}: code {n}",
]


def _rng_word(rng): return rng.choice(_WORDS)


def generate_corpus(n_lines: int = 6000, seed: int = 0) -> List[str]:
    rng = random.Random(seed)
    lines: List[str] = []
    for _ in range(n_lines):
        d = rng.random()
        if d < 0.55:  # english prose with recurring phrases (superword bait)
            parts = []
            for _ in range(rng.randint(6, 16)):
                if rng.random() < 0.25:
                    parts.append(rng.choice(_PHRASES))
                else:
                    parts.append(_rng_word(rng))
            lines.append(" ".join(parts) + ".")
        elif d < 0.75:  # code
            tmpl = rng.choice(_CODE)
            lines.append(tmpl.format(f=_rng_word(rng), C=_rng_word(rng).capitalize(),
                                     n=rng.randint(0, 999)))
        elif d < 0.9:   # logs with timestamps + paths
            tmpl = rng.choice(_LOG)
            ts = f"2026-07-{rng.randint(1,28):02d}T{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:00"
            lines.append(tmpl.format(ts=ts, n=rng.randint(1, 99999), f=_rng_word(rng)))
        else:           # urls / numbers
            lines.append(f"https://example.com/{_rng_word(rng)}/{_rng_word(rng)}?id={rng.randint(1,99999)}")
    return lines


def split_corpus(lines: List[str], train_frac: float = 0.85):
    k = int(len(lines) * train_frac)
    return lines[:k], lines[k:]
