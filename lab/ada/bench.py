"""
Ada benchmark harness — equal-footing comparison (lab rule 6,7).

Trains several tokenizer configurations on the SAME synthetic corpus at the SAME vocab size,
then measures on a held-out split:
  * compression  : bytes / token  (higher is better) and tokens per 1000 bytes
  * speed        : train seconds, encode throughput (MB/s)
  * memory       : merge-table size (# merges), roundtrip losslessness

Configurations (ablation of Ada's three stolen ideas):
  raw-bytes            : every byte a token (compression floor, bytes/token = 1.0)
  bpe-baseline         : plain 2-stage-less BPE (whitespace kept, no superword/entropy/length)
  +superword           : add SuperBPE cross-whitespace merges
  +superword+entropy   : add BLT entropy guard
  ada-full             : + Length-MAX length bonus  (all three)

Run:  python3 bench.py           (writes bench_results.json, prints a table)
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Dict, List

from ada_tokenizer import AdaTokenizer, AdaConfig
from corpus import generate_corpus, split_corpus


def _measure(tok: AdaTokenizer, texts: List[str]) -> Dict:
    total_bytes = sum(len(t.encode("utf-8")) for t in texts)
    t0 = time.time()
    total_tokens = 0
    lossless = True
    for t in texts:
        ids = tok.encode(t)
        total_tokens += len(ids)
        if tok.decode(ids) != t:
            lossless = False
    dt = time.time() - t0
    return {
        "bytes": total_bytes,
        "tokens": total_tokens,
        "bytes_per_token": round(total_bytes / max(1, total_tokens), 4),
        "tokens_per_1k_bytes": round(1000 * total_tokens / max(1, total_bytes), 2),
        "encode_MB_per_s": round((total_bytes / 1e6) / max(1e-9, dt), 3),
        "lossless": lossless,
    }


def run(vocab_size: int = 1024, n_lines: int = 6000, seed: int = 0) -> Dict:
    lines = generate_corpus(n_lines=n_lines, seed=seed)
    train, test = split_corpus(lines)
    base = AdaConfig(vocab_size=vocab_size, seed=seed)

    configs = {
        "bpe-baseline": replace(base, enable_superword=False, enable_entropy_guard=False,
                                enable_length_bonus=False),
        "+superword": replace(base, enable_superword=True, enable_entropy_guard=False,
                              enable_length_bonus=False),
        "+superword+entropy": replace(base, enable_superword=True, enable_entropy_guard=True,
                                      enable_length_bonus=False),
        "ada-full": replace(base, enable_superword=True, enable_entropy_guard=True,
                            enable_length_bonus=True),
    }

    results: Dict[str, Dict] = {}
    # raw-bytes floor (no training)
    raw_bytes = sum(len(t.encode("utf-8")) for t in test)
    results["raw-bytes"] = {"vocab": 256, "train_s": 0.0, "merges": 0,
                            "bytes_per_token": 1.0,
                            "tokens_per_1k_bytes": round(1000 * raw_bytes / raw_bytes, 2),
                            "encode_MB_per_s": None, "lossless": True}

    for name, cfg in configs.items():
        print(f"[train] {name} ...", flush=True)
        t0 = time.time()
        tok = AdaTokenizer(cfg).train(train, verbose=False)
        train_s = round(time.time() - t0, 2)
        m = _measure(tok, test)
        m.update({"vocab": tok.vocab_size, "train_s": train_s, "merges": len(tok.merges)})
        results[name] = m

    return {"vocab_size": vocab_size, "n_lines": n_lines, "seed": seed, "results": results}


def _fmt(v):
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def print_table(bundle: Dict) -> None:
    r = bundle["results"]
    base_bpt = r["bpe-baseline"]["bytes_per_token"]
    cols = ["config", "vocab", "merges", "bytes/tok", "vs_bpe", "tok/1kB", "enc_MB/s", "train_s", "lossless"]
    print("\n" + "  ".join(f"{c:>12}" for c in cols))
    print("  ".join("-" * 12 for _ in cols))
    order = ["raw-bytes", "bpe-baseline", "+superword", "+superword+entropy", "ada-full"]
    for name in order:
        d = r[name]
        bpt = d["bytes_per_token"]
        vs = "" if name in ("raw-bytes", "bpe-baseline") else f"{100*(bpt-base_bpt)/base_bpt:+.1f}%"
        row = [name, d.get("vocab"), d.get("merges"), f"{bpt:.3f}", vs,
               d.get("tokens_per_1k_bytes"), _fmt(d.get("encode_MB_per_s")),
               d.get("train_s"), d.get("lossless")]
        print("  ".join(f"{str(x):>12}" for x in row))


if __name__ == "__main__":
    bundle = run()
    print_table(bundle)
    with open("bench_results.json", "w") as f:
        json.dump(bundle, f, indent=2)
    print("\nwrote bench_results.json")
