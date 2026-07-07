"""
Ada corpus loader — real datasets (streaming) with a synthetic fallback.

Tokenizer training is UNSUPERVISED: it needs raw text, not labels. On this CPU-only box we
never download (lab rule 11) so `generate_corpus` (synthetic) is the default here; on Colab,
`load_hf_corpus` streams a real 2026-frontier corpus (see FINDINGS §8 for the sweep). Both
return a `List[str]` of lines, the shape AdaTokenizer.train expects.

Dataset registry (winners of the 2026-07-07 sweep):
  fineweb-edu  English, quality-filtered web            HuggingFaceFW/fineweb-edu  (sample-10BT)
  fineweb      English, broad web                       HuggingFaceFW/fineweb      (sample-10BT)
  fineweb-2    Multilingual (1000+ langs, per script)   HuggingFaceFW/fineweb-2    (e.g. fra_Latn)
  code         Source code, 600+ langs (content-bearing) bigcode/the-stack-smol
The byte-level, multilingual, code-robust Ada wants a MIX: default recipe below is
fineweb-edu + code + one non-Latin language, which stresses exactly what byte tokenizers win on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# name -> (hf_repo, config_name, text_field)
REGISTRY: Dict[str, tuple] = {
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    "fineweb":     ("HuggingFaceFW/fineweb",     "sample-10BT", "text"),
    "fineweb-2":   ("HuggingFaceFW/fineweb-2",   "fra_Latn",    "text"),
    "code":        ("bigcode/the-stack-smol",    None,          "content"),
}


@dataclass
class DataConfig:
    """Config for building a training corpus (rule 10)."""
    source: str = "fineweb-edu"       # a REGISTRY key, or "synthetic"
    config_name: Optional[str] = None  # override REGISTRY config (e.g. a fineweb-2 language)
    max_chars: int = 20_000_000        # ~20 MB of text — plenty to train a tokenizer
    min_line_len: int = 8
    seed: int = 0
    # For a robust byte tokenizer, blend several sources (each capped at max_chars/len(mix)):
    mix: List[str] = field(default_factory=lambda: [])  # e.g. ["fineweb-edu","code","fineweb-2"]


def _docs_to_lines(text: str, min_line_len: int) -> List[str]:
    return [ln for ln in text.splitlines() if len(ln) >= min_line_len]


def load_hf_corpus(cfg: DataConfig) -> List[str]:
    """Stream up to `max_chars` of text from one HF dataset. Requires `datasets` (Colab)."""
    from datasets import load_dataset  # noqa: imported lazily; not installed on this box

    repo, default_name, field_name = REGISTRY[cfg.source]
    name = cfg.config_name or default_name
    stream = load_dataset(repo, name=name, split="train", streaming=True)
    lines: List[str] = []
    total = 0
    for row in stream:
        text = row.get(field_name) or ""
        for ln in _docs_to_lines(text, cfg.min_line_len):
            lines.append(ln)
            total += len(ln)
        if total >= cfg.max_chars:
            break
    return lines


def build_corpus(cfg: DataConfig) -> List[str]:
    """One entry point. Handles 'synthetic', a single source, or a `mix` of sources."""
    if cfg.source == "synthetic" and not cfg.mix:
        from corpus import generate_corpus
        return generate_corpus(n_lines=6000, seed=cfg.seed)

    sources = cfg.mix if cfg.mix else [cfg.source]
    per = max(1, cfg.max_chars // len(sources))
    out: List[str] = []
    for s in sources:
        if s == "synthetic":
            from corpus import generate_corpus
            out.extend(generate_corpus(n_lines=6000, seed=cfg.seed))
            continue
        sub = DataConfig(source=s, config_name=cfg.config_name if s == cfg.source else None,
                         max_chars=per, min_line_len=cfg.min_line_len, seed=cfg.seed)
        out.extend(load_hf_corpus(sub))
    return out
