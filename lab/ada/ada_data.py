"""
Ada corpus loader — aggressive, quality-tiered, recency-weighted, and RESILIENT.

Tokenizer training is UNSUPERVISED: it needs raw text, not labels. Corpus quality and domain
coverage set tokenizer quality, so this module is greedy about both — a big registry of the
2026-frontier open corpora (FINDINGS §8), grouped by quality tier, blended by weight, with
per-source quality filtering and exact-line dedup.

RESILIENCE (this is the important bit): every source is tried with several candidate configs and
several candidate text-field names; any source that is gated / missing / errors is *warned and
skipped*, never fatal. If a whole recipe yields nothing (offline, no token), it falls back to the
synthetic generator. So a gated dataset like `bigcode/the-stack-smol` can no longer crash a run —
it is simply skipped (and we default to the ungated `codeparrot/github-code-clean` for code).

On this CPU-only box nothing downloads (rule 11); the real corpora stream on Colab.
For gated/premium sources set HF_TOKEN (env) or DataConfig.hf_token after accepting the license.
"""
from __future__ import annotations

import hashlib
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class Source:
    repo: str
    configs: Sequence[Optional[str]]   # tried in order; first that loads wins
    fields: Sequence[str]              # candidate text columns, tried in order per row
    tier: str                          # premium | web | multilingual | code | math | knowledge
    note: str = ""
    needs_token: bool = False          # gated: requires an accepted license + HF token


# --- The registry. Greedy: the best + most recent open corpora as of 2026-07-07 (FINDINGS §8). ---
SOURCES: Dict[str, Source] = {
    # premium English web (classifier-filtered / rephrased). Highest signal per token.
    "fineweb-edu":   Source("HuggingFaceFW/fineweb-edu", ["sample-10BT", "sample-100BT", "default"], ["text"], "premium", "quality-filtered edu web (default best)"),
    "ultra-fineweb": Source("openbmb/Ultra-FineWeb", ["default", "en", None], ["content", "text"], "premium", "verified/efficient re-filter of FineWeb (2505.05427)"),
    "dclm":          Source("mlfoundations/dclm-baseline-1.0-parquet", [None, "default"], ["text"], "premium", "DCLM-baseline 3.8T, fastText-screened", needs_token=False),
    "nemotron-cc-hq":Source("nvidia/Nemotron-CC-HQ", [None, "default"], ["text"], "premium", "NVIDIA HQ CC subset, +5.6 MMLU vs DCLM", needs_token=True),
    # broad + multilingual web
    "fineweb":       Source("HuggingFaceFW/fineweb", ["sample-10BT", "sample-100BT", "default"], ["text"], "web", "broad English web"),
    "fineweb-2":     Source("HuggingFaceFW/fineweb-2", ["fra_Latn", "deu_Latn", "spa_Latn", "rus_Cyrl", "arb_Arab", "zho_Hans", "hin_Deva", "jpn_Jpan"], ["text"], "multilingual", "1000+ langs / 1893 language-script pairs"),
    # code (ungated — the workaround for the gated the-stack)
    "code":          Source("codeparrot/github-code-clean", ["all-all", None], ["code"], "code", "ungated GitHub code, 30+ langs (the-stack-smol is gated!)"),
    "code-raw":      Source("codeparrot/github-code", ["all-all", None], ["code"], "code", "ungated GitHub code (uncleaned)"),
    # math / reasoning (strong recent signal for structured text)
    "finemath":      Source("HuggingFaceFW/finemath", ["finemath-4plus", "finemath-3plus", "default"], ["text"], "math", "highest-quality open math web"),
    "open-web-math": Source("open-web-math/open-web-math", [None, "default"], ["text"], "math", "10B+ math tokens"),
    "nemotron-math": Source("nvidia/Nemotron-CC-Math", [None, "default"], ["text"], "math", "133B math tokens, beats FineMath (2508.15096)", needs_token=True),
    # curated knowledge (clean, dense, canonical spellings — good for tokenizer vocab)
    "wikipedia":     Source("wikimedia/wikipedia", ["20231101.en", "20231101.simple"], ["text"], "knowledge", "clean encyclopedic prose"),
    "stackexchange": Source("HuggingFaceH4/stack-exchange-preferences", [None, "default"], ["question", "text"], "knowledge", "Q&A / technical prose"),
    # permissive-first (license-clean) option
    "mixturevitae":  Source("ontocord/MixtureVitae", [None, "default"], ["text"], "premium", "permissive-first web + instruction/reasoning (2509.25531)"),
}

# Named recipes: source -> blend weight. `default` is all-ungated and just works.
RECIPES: Dict[str, Dict[str, float]] = {
    # balanced, ungated, byte-tokenizer stress test (English + code + math + multilingual + knowledge)
    "default":      {"fineweb-edu": 0.34, "code": 0.24, "fineweb-2": 0.18, "finemath": 0.12, "wikipedia": 0.12},
    # go wide: everything, premium gated included (needs HF_TOKEN; gated ones skipped if absent)
    "aggressive":   {"fineweb-edu": 0.20, "ultra-fineweb": 0.12, "dclm": 0.12, "code": 0.18,
                     "fineweb-2": 0.14, "finemath": 0.10, "open-web-math": 0.06, "wikipedia": 0.08},
    "english":      {"fineweb-edu": 0.6, "wikipedia": 0.2, "finemath": 0.2},
    "multilingual": {"fineweb-2": 0.7, "fineweb-edu": 0.15, "wikipedia": 0.15},
    "code-heavy":   {"code": 0.6, "fineweb-edu": 0.25, "finemath": 0.15},
}


@dataclass
class DataConfig:
    """Every knob (rule 10). Pick a recipe OR a single source, set a byte budget + quality gates."""
    recipe: str = "default"            # a RECIPES key; ignored if `source` set
    source: Optional[str] = None       # a single SOURCES key (or "synthetic"); overrides recipe
    config_name: Optional[str] = None  # force a specific config for the single source (e.g. fineweb-2 lang)
    max_chars: int = 30_000_000        # ~30 MB total across the blend (greedy; tokenizers need little, LM more)
    # quality gates
    min_line_len: int = 16
    max_line_len: int = 20_000
    dedup: bool = True                 # drop exact-duplicate lines (hash-based)
    min_alnum_frac: float = 0.55       # drop lines that are mostly punctuation/markup noise
    shuffle: bool = True
    seed: int = 0
    # access
    hf_token: Optional[str] = None     # or set HF_TOKEN in the environment
    verbose: bool = True

    def token(self) -> Optional[str]:
        return self.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


# --------------------------------------------------------------------------- quality filter
def _line_ok(ln: str, cfg: DataConfig) -> bool:
    n = len(ln)
    if n < cfg.min_line_len or n > cfg.max_line_len:
        return False
    alnum = sum(c.isalnum() or c.isspace() for c in ln)
    return (alnum / n) >= cfg.min_alnum_frac


def _iter_clean_lines(text: str, cfg: DataConfig, seen: Optional[set]):
    for ln in text.splitlines():
        ln = ln.strip()
        if not _line_ok(ln, cfg):
            continue
        if seen is not None:
            h = hashlib.blake2b(ln.encode("utf-8", "ignore"), digest_size=8).digest()
            if h in seen:
                continue
            seen.add(h)
        yield ln


def _warn(cfg: DataConfig, msg: str):
    if cfg.verbose:
        print(f"[ada_data] {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- one source
def load_hf_source(name: str, budget_chars: int, cfg: DataConfig,
                   seen: Optional[set], config_override: Optional[str] = None) -> List[str]:
    """Stream up to `budget_chars` of clean text from one registry source. Never raises."""
    try:
        from datasets import load_dataset  # lazy; not on this box
    except Exception as e:  # noqa
        _warn(cfg, f"`datasets` not installed ({e}); skipping '{name}'")
        return []

    if name not in SOURCES:
        _warn(cfg, f"unknown source '{name}'; skipping")
        return []
    src = SOURCES[name]
    tok = cfg.token()
    if src.needs_token and not tok:
        _warn(cfg, f"'{name}' ({src.repo}) is gated and no HF_TOKEN set; skipping")
        return []

    configs = [config_override] if config_override else list(src.configs)
    for conf in configs:
        try:
            ds = load_dataset(src.repo, name=conf, split="train", streaming=True, token=tok)
        except Exception as e:  # noqa: gated/missing/config error -> try next config
            _warn(cfg, f"'{name}' config={conf!r} failed to open ({type(e).__name__}); trying next")
            continue
        lines: List[str] = []
        total = 0
        try:
            for row in ds:
                text = ""
                for fld in src.fields:
                    v = row.get(fld)
                    if v:
                        text = v if isinstance(v, str) else str(v)
                        break
                if not text:
                    continue
                for ln in _iter_clean_lines(text, cfg, seen):
                    lines.append(ln)
                    total += len(ln)
                if total >= budget_chars:
                    break
        except Exception as e:  # noqa: mid-stream error -> keep what we have
            _warn(cfg, f"'{name}' streaming stopped early ({type(e).__name__}); kept {total} chars")
        if lines:
            _warn(cfg, f"'{name}' [{src.tier}] {src.repo} config={conf!r}: {len(lines):,} lines / {total/1e6:.1f} MB")
            return lines
    _warn(cfg, f"'{name}' produced nothing across all configs; skipping")
    return []


# --------------------------------------------------------------------------- build the corpus
def build_corpus(cfg: DataConfig) -> List[str]:
    """One entry point. Blends a recipe (or a single source), quality-filters, dedups, shuffles.
    Falls back to synthetic text if every real source is unavailable."""
    import random
    rng = random.Random(cfg.seed)
    seen: Optional[set] = set() if cfg.dedup else None

    if cfg.source == "synthetic":
        from corpus import generate_corpus
        return generate_corpus(n_lines=8000, seed=cfg.seed)

    if cfg.source:  # single explicit source
        weights = {cfg.source: 1.0}
        override = cfg.config_name
    else:
        weights = RECIPES.get(cfg.recipe)
        if weights is None:
            _warn(cfg, f"unknown recipe '{cfg.recipe}', using 'default'")
            weights = RECIPES["default"]
        override = None

    total_w = sum(weights.values()) or 1.0
    out: List[str] = []
    for name, w in weights.items():
        budget = int(cfg.max_chars * (w / total_w))
        out.extend(load_hf_source(name, budget, cfg, seen,
                                  config_override=override if name == cfg.source else None))

    if not out:
        _warn(cfg, "no real data loaded (offline / gated / no token) -> synthetic fallback")
        from corpus import generate_corpus
        return generate_corpus(n_lines=8000, seed=cfg.seed)

    if cfg.shuffle:
        rng.shuffle(out)
    _warn(cfg, f"corpus ready: {len(out):,} lines, {sum(len(l) for l in out)/1e6:.1f} MB "
               f"(recipe={cfg.recipe if not cfg.source else cfg.source}, dedup={cfg.dedup})")
    return out


# convenience: list what's available
def catalog() -> str:
    rows = [f"{n:16s} {s.tier:12s} {'GATED' if s.needs_token else 'open ':5s} {s.repo}" for n, s in SOURCES.items()]
    return "recipes: " + ", ".join(RECIPES) + "\n" + "\n".join(rows)
