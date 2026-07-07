# Ada — the best-of-all-worlds tokenizer

**Ada** is a byte-level tokenizer that fuses the strongest idea from each donor into one
small, fast, CPU-first design. Named for Ada Lovelace. One model, not an ensemble.

## Donor roster → what Ada steals

| Donor | Idea stolen | Where it lands in Ada |
|---|---|---|
| **SuperBPE** (§1.4) | cross-whitespace "superword" merges, two-stage | vocabulary backbone; `superword_start_frac` transition |
| **BLT** (§1.1) | merge predictable spans, keep surprising boundaries (entropy) | `entropy_lambda` guard on merge selection, from **corpus stats, no NN** |
| **Length-MAX** (§1.5) | score merges by realised length reduction, not raw frequency | `length_bonus` term in the merge objective |
| **H-Net** (§1.2) | boundaries = semantic dissimilarity, learned end-to-end | **v1 neural upgrade** (Colab); v0 approximates with branching entropy |
| **MrT5** (§1.5) | delete redundant units mid-network | **v1 neural upgrade** (Colab decoder-side) |
| **Cosmos FSQ** (§1.6) | codebook-free quantized bottleneck (no collapse) | **v1** discrete-latent quantizer if Ada goes neural |

## Decision matrix (per axis)

| Axis | Choice for v0 (CPU, shippable now) | Why |
|---|---|---|
| Base units | raw bytes (256) | tokenizer-free base, multilingual/robust (BLT, EvaByte) |
| Vocab construction | 2-stage BPE: subword → superword | SuperBPE; biggest compression win, no NN |
| Merge selection | frequency × entropy-guard × length-bonus | OI-1: BLT principle + Length-MAX, CPU-only |
| Encode | greedy longest-merge, deterministic | fast, simple, reproducible |
| Boundary learning | branching-entropy proxy (statistical) | H-Net's signal without training |
| Bottleneck | none (v0 is lossless discrete IDs) | v0 is a tokenizer, not a codec |

## v0 vs v1

- **v0 (this box, now):** the statistical entropy-guarded superword BPE. Trains in seconds on
  a synthetic corpus, encodes in pure Python, benchmarked here. Lossless (invertible).
- **v1 (Colab, later):** a tiny neural front-end — an H-Net-style differentiable router over
  byte embeddings with an FSQ bottleneck and an MrT5 mid-network deletion gate — trained on
  Colab, brought back as a checkpoint and benchmarked here. Ships as
  `lab/ada/ada_colab_train.ipynb` (self-contained, one config cell). v0's entropy-guard is the
  ablation baseline the neural router must beat.

## Success criteria (must hold together)

1. **Compression:** ≥ plain BPE's bytes/token at the *same* vocab size, ideally +10–30%.
2. **Speed:** encode throughput within the same order as the baseline; train in seconds.
3. **Memory:** merge table ≤ baseline's; no auxiliary model at inference for v0.
4. **Ablation:** each of {superword, entropy-guard, length-bonus} shows a measured contribution;
   any that doesn't survive ablation is cut (rule 9).
