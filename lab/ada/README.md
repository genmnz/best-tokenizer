# Ada tokenizer (lab)

Best-of-all-worlds byte tokenizer. v0 is statistical, pure-stdlib, CPU-only. See
[`../../ADA.md`](../../ADA.md) for the charter and [`../../FINDINGS.md`](../../FINDINGS.md)
§1/§4/§5 for the donor citations and measured results.

## What it is

A byte-level (256 base) two-stage BPE that:
1. learns ordinary **subwords** (whitespace enforced), then
2. resumes learning **superwords** that cross spaces (SuperBPE, arXiv:2503.13423), with
3. merge selection **guarded by branching entropy** `H(R|L)` from corpus stats — merge
   predictable spans, keep surprising boundaries (BLT principle, arXiv:2412.09871, but **no
   neural entropy model** — pure counts, runs on CPU at build time).

Result on a fresh synthetic corpus at vocab 1024: **+43.4% bytes/token vs plain BPE, equal
encode speed, equal memory, lossless.**

## Files

| File | Purpose |
|---|---|
| `ada_tokenizer.py` | `AdaTokenizer` + `AdaConfig` (every knob is a config field, rule 10) |
| `corpus.py` | synthetic multi-domain corpus generator (no downloads, rule 11) |
| `ada_data.py` | real-dataset loader (streams FineWeb-Edu / FineWeb-2 / code) + synthetic fallback |
| `bench.py` | equal-footing ablation benchmark → `bench_results.json` |
| `test_ada.py` | test suite (pytest **or** `python3 test_ada.py`) |
| `ada_train.ipynb` | **Colab: train on a real dataset** — v0 tokenizer + ablation, then v1 neural |

## Run

```bash
cd lab/ada
python3 test_ada.py     # 7/7
python3 bench.py         # prints the ablation table, writes bench_results.json
```

## Use

```python
from ada_tokenizer import AdaTokenizer, AdaConfig
from corpus import generate_corpus

tok = AdaTokenizer(AdaConfig(vocab_size=1024)).train(generate_corpus(6000))
ids = tok.encode("of the best of all worlds tokenizer")
assert tok.decode(ids) == "of the best of all worlds tokenizer"
tok.save("ada.json"); AdaTokenizer.load("ada.json")
```

## Config knobs (`AdaConfig`)

| field | default | meaning |
|---|---|---|
| `vocab_size` | 1024 | total ids incl. 256 base bytes |
| `superword_start_frac` | 0.75 | fraction of merges as subwords before whitespace is dropped |
| `entropy_lambda` | 1.0 | BLT guard strength (0 = off) |
| `enable_length_bonus` | **False** | Length-MAX term — CUT: hurt held-out compression (FINDINGS §5) |
| `enable_superword` | True | False = plain BPE baseline |
| `min_pair_count` | 2 | ignore merges rarer than this |

## Known walls / v1 work

- Encode is O(n·log n) but still Python; a Rust/C port or vectorized batch encode is the
  throughput lever.
- Entropy-guard is under-tested on synthetic text; needs a natural-corpus eval (Colab).
- v1 replaces the statistical boundary heuristic with an H-Net differentiable router + FSQ
  bottleneck + MrT5 deletion gate, and must beat this v0 in ablation to ship.
