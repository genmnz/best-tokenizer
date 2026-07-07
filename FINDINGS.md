# FINDINGS — best-tokenizer lab

Canonical research log. Every technique studied, every benchmark run, every trend named.
If it isn't here, the research didn't happen. Newest material appended; dates in ISO.

Scope: **learned/compression tokenization for text & bytes**, plus the transferable
quantization idea from video. Six donor repos cloned into `best/`, nine papers in `papers/`.
Target: build **Ada** — one small, fast, CPU-friendly tokenizer that takes the best idea
from each donor and measurably beats plain BPE at equal vocab (see `ADA.md`).

Reference hardware for all numbers below: this container — 4 vCPU, 15 GiB RAM, CPU-only,
Python 3.11, no numpy/torch (pure-Python measurements).

---

## 1. Donor technique inventory (RESEARCH)

Each entry: the one breakthrough, the exact code path that implements it, the wall it hits.

### 1.1 BLT — entropy-adaptive byte patching  · `best/blt`  · arXiv:2412.09871
**Breakthrough.** No fixed vocabulary. A small byte-level "entropy model" predicts the
next-byte distribution; per-byte entropy `H = -Σ p·log p` drives patch boundaries — a new
(small) patch starts where the next byte is *hard to predict* (high entropy), predictable
runs get merged into big patches. Compute follows surprise.

**Code path.**
- `best/blt/bytelatent/data/patcher.py:48` `entropy(scores)` — natural-log entropy of the
  softmax over the 256-way byte distribution.
- `:111` `patch_start_mask_from_entropy_with_monotonicity(entropies, t)` — boundary where
  `entropy[i] - entropy[i-1] > t` (a *rise* in surprise starts a patch).
- `:137` `patch_start_mask_global_and_monotonicity(entropies, t, t_add)` — boundary where
  `(Δentropy > t_add) AND (entropy > t)`. Default global threshold `t = 1.3354` nats
  (`patcher.py:35`).
- Entropy model itself: `best/blt/bytelatent/entropy_model.py:13` — a small sliding-window
  (512) causal LMTransformer, frozen at inference.

**Wall.** Needs a separately-trained neural entropy model to run at all; the boundary rule
is a non-differentiable threshold (heuristic, not learned end-to-end). Heavy for CPU.

### 1.2 H-Net — differentiable dynamic chunking  · `best/hnet`  · arXiv:2507.07955
**Breakthrough.** Learns *where* to chunk end-to-end, no entropy model, no threshold.
A routing module scores each position by how *dissimilar* it is from its predecessor;
dissimilar = semantic change = a chunk boundary. Fully differentiable via a smoothing
("de-chunk") upsampler, so boundaries are trained jointly with the LM objective.

**Code path.**
- `best/hnet/hnet/modules/dc.py:69` `RoutingModule.forward` —
  `boundary_prob = (1 - cos_sim(q·h_t, k·h_{t+1})) / 2` (`dc.py:88-92`), with `q,k`
  linear projections initialised to **identity** (`dc.py:55-57`). Boundary = argmax over
  {no,yes}, i.e. `prob > 0.5` (`dc.py:104-106`).
- `best/hnet/hnet/modules/dc.py:167` `ChunkLayer` — gathers only boundary positions
  (downsample). `:213` `DeChunkLayer` — upsamples back with an **EMA smoother** implemented
  on the Mamba2 scan kernel, weighted by the boundary probability (`dc.py:277-295`), which
  is what makes the discrete selection differentiable (straight-through-like).
- Encoder/decoder are Mamba (SSM) layers — recurrent state is a natural compressor.

**Wall.** Needs Mamba2 CUDA kernels (`mamba_chunk_scan_combined`) for the differentiable
de-chunk; hierarchical stacking is training-heavy. No CPU story.

### 1.3 EvaByte — multibyte prediction + EVA linear attention  · `best/evabyte`  · HKU 2025
**Breakthrough (two).** (a) **Multibyte prediction**: one hidden state predicts `n>1`
future bytes in parallel, amortising the "bytes are ~4× longer than tokens" cost —
`best/evabyte/evabyte_hf/modeling_evabyte.py:753`
`lm_head = Linear(hidden, vocab_size * num_pred_heads)`. (b) **EVA attention**: local exact
attention inside a window + chunk-level random-feature (linear) approximation for far
context — `best/evabyte/evabyte_hf/eva.py:63` `EvaAttention`, `window_size=512`,
`chunk_size` landmarks (`eva.py:89-98`). Byte vocab = 320 (`configuration_evabyte.py:11`).

**Wall.** Still a full 6.5B byte LM; multibyte heads assume independence between the
predicted bytes (the arXiv:2511.11346 follow-up fixes this with probabilistic circuits).

### 1.4 SuperBPE — superword tokens (cross-whitespace merges)  · `best/superbpe`  · arXiv:2503.13423
**Breakthrough.** Drop BPE's sacred whitespace-pretokenization *after* a transition point.
Stage 1 learns ordinary subwords (whitespace enforced); Stage 2 resumes the *same* merge
process with pretokenization disabled, so merges span spaces to form "superword" tokens
("of the", "by the way"). Up to 33% fewer tokens at equal vocab. **Cheapest, most drop-in
win on the whole list — no neural net at all.**

**Code path.**
- Two-stage design documented `best/superbpe/README.md:45-50`; stage boundary is which
  pretokenization regex is active (`best/superbpe/utils.py:35` `get_pretokenization_regex`,
  `:57` builds the pre_tokenizer sequence).
- `best/superbpe/train_tokenizer.py:98` `train_or_extend_tokenizer(...)` — presence of an
  existing `merges.txt` switches from "train" to "extend" (= enter stage 2).
- Ships a released 128k english tokenizer (`best/superbpe/tokenizer_json/`, 265 MB).

**Wall.** Requires a custom fork of HF `tokenizers` (Rust) to disable pretokenization
mid-training. Greedy longest-match encode can mis-segment if superwords overlap.

### 1.5 MrT5 — learned mid-network token deletion  · `best/mrt5`  · arXiv:2410.20771
**Breakthrough.** Don't merge at the input — merge *inside* the network. After a few byte
encoder layers a learned "delete gate" removes redundant byte positions, shortening the
sequence for the deep layers. Orthogonal to input-level patching (§1.1/1.4).

**Code path.**
- `best/mrt5/models/modeling_mrt5.py:104` `ScaledSigmoid` = `scale · sigmoid(-logit)`;
  `:107` the gate is a feed-forward over hidden states producing per-position logits, with
  optional Gumbel noise (`:114`). Gate value ~0 ⇒ position deleted from later attention.
- `best/mrt5/models/shortening.py` (adapted from Nawrot et al. dynamic-pooling) —
  `downsample`/`common`/`final` do the actual boundary-weighted pool via `einsum`.
- A deletion-rate regulariser pushes the gate toward a target compression.

**Wall.** Gate is a soft mask trained with a regulariser — needs full training and careful
tuning of `sigmoid_mask_scale` and the deletion target; another full T5 to run.

### 1.6 Cosmos FSQ — codebook-free finite scalar quantization  · `best/cosmos-tokenizer`  · arXiv:2309.15505
**Breakthrough (transferable).** Not text, but the cleanest *bottleneck* on the list.
**FSQ** replaces the learned VQ-VAE codebook with plain per-channel rounding: bound each of
`d` latent channels with a `tanh`, round to one of `L` levels with a straight-through
estimator, done. Implicit codebook size = `∏ levels` (e.g. 6 channels × ~8 levels ≈ 64k)
with **zero embedding parameters and no codebook collapse**.

**Code path.**
- `best/cosmos-tokenizer/cosmos_tokenizer/modules/quantizers.py:71` `FSQuantizer`.
- `:136` `bound(z)` = `tanh(z + shift) · half_l - offset`; `:143` `quantize` =
  `round_ste(bound(z)) / half_width`. `codebook_size = ∏ levels` (`:129`).
- `:36` `ResidualFSQuantizer` stacks FSQ residually for more capacity.

**Wall.** Low per-code capacity leans hard on the decoder to fix things up (the CS-FSQ
follow-up splits channels to add capacity). Only relevant to *neural* (continuous-latent)
tokenizers — the v1 Ada upgrade, not the v0 statistical one.

---

## 2. Papers pool (`papers/`)

Backing the six donors + greenfield technique papers (latest, some not-yet-implemented):
`BLT 2412.09871`, `H-Net 2507.07955`, `H-Net++ 2508.05628`, `SuperBPE 2503.13423`,
`MrT5 2410.20771`, `Length-MAX 2511.20849`, `Multi-byte probabilistic circuits 2511.11346`,
`DeepSeek-OCR 2510.18234`, `FSQ 2309.15505`. All 9 downloaded as PDFs.

Greenfield (no open model yet — build targets):
- **Length-MAX** (2511.20849) — reframes tokenizer construction as *length maximisation*
  under a vocab budget; drop-in BPE-compatible. Directly usable at vocab-build time on CPU.
- **Multi-byte probabilistic circuits** (2511.11346) — relaxes the independence assumption
  between EvaByte's parallel byte predictions. Neural, v1+.

---

## 3. Trends (cross-repo convergence — outranks individual results)

- **T1 — "Dynamic boundary" is the whole field.** BLT (entropy heuristic) and H-Net
  (learned cosine-dissimilarity router) are the two answers to *where to compress*. Every
  other lane is re-deriving it: audio's FlexiCodec (content-adaptive frame rate),
  time-series' Kairos (mixture-of-size patching), MrT5 (learned mid-net deletion). The
  design axis for Ada is **which boundary signal** (entropy vs. dissimilarity vs. frequency).
- **T2 — "Compress before the model sees it."** SuperBPE (fewer tokens/text), BLT/H-Net
  (patch bytes), DeepSeek-OCR (text→pixels→few vision tokens) all shrink the sequence the
  transformer consumes. Tokenization *is* compression.
- **T3 — Kill the learned codebook.** FSQ (round to a grid) and WavTokenizer/Cosmos show
  fixed/implicit codebooks beat trained VQ embeddings (no collapse). For a discrete neural
  Ada, prefer FSQ over VQ-VAE.
- **T4 — Small beats big when the representation is right.** SuperBPE lifts MMLU +8.2% at
  fixed compute purely by tokenizing better; the win is in the front-end, not parameters.

---

## 4. Original ideas (rule 10 — measured, honest)

### OI-1 — Entropy-guarded superword BPE (Ada v0)
Synthesis: take SuperBPE's cross-whitespace merging (§1.4) but choose *which* pairs to merge
using BLT's principle (§1.1) — merge *predictable* spans, keep *surprising* boundaries —
computed from **corpus statistics, not a neural entropy model**, so it runs on CPU at
vocab-build time. Merge score = `count(L,R) · (1 + λ·(Hmax − H(R|L)))` where `H(R|L)` is the
branching entropy of the right symbol given the left. Plus a Length-MAX (§1.5) tie-break by
realised token-length reduction. Measured results in §5.

---

## 5. Benchmarks (RESULTS — numbers, not adjectives)

Harness: `lab/ada/bench.py`. Corpus: `lab/ada/corpus.py` synthetic multi-domain, 6000 lines,
seed 0, 85/15 train/test. Same corpus + same vocab (1024) + same encode code path for every
row (rule 6). Container: 4 vCPU CPU-only, pure Python. **2026-07-07.**

| config | vocab | merges | bytes/tok | vs BPE | encode MB/s | train s | lossless |
|---|---|---|---|---|---|---|---|
| raw-bytes (floor) | 256 | 0 | 1.000 | — | — | 0 | ✓ |
| bpe-baseline | 1024 | 768 | 4.936 | — | 1.32 | 4.4 | ✓ |
| +superword (SuperBPE) | 1024 | 768 | **7.077** | **+43.4%** | 1.30 | 7.1 | ✓ |
| +superword+entropy (BLT guard) | 1024 | 768 | 7.079 | +43.4% | 1.31 | 8.6 | ✓ |
| +length-bonus (Length-MAX, w=0.1) | 1024 | 768 | 6.952 | +40.9% | 1.32 | 8.8 | ✓ |

Ablation sweep (held-out bytes/tok, superword on):
`entropy λ = {0.25,0.5,1.0,2.0} → 7.081 / 7.079 / 7.086 / 7.080` (all ≥ 7.077 superword-only);
`length_weight = {0.02,0.05,0.10} → 7.054 / 7.022 / 6.923` (monotonically worse).

**Verdict (all axes must hold together — they do):**
- **Compression:** superword is the whole win, **+43.4% bytes/token** at equal vocab. This is
  the SuperBPE result reproduced from scratch on a fresh corpus.
- **Speed:** the O(n·log n) heap encoder (`ada_tokenizer.py:_encode_symbols`) makes superword
  encode **1.30 MB/s vs baseline 1.32 MB/s — statistically equal** (the naive O(n²) encoder was
  5× slower at 0.25 MB/s; that was the wall, now removed). Train < 9 s.
- **Memory:** identical merge table (768 merges), no auxiliary model at inference. ✓
- **Lossless:** every config round-trips exactly, incl. unicode/emoji (test suite). ✓

**Honest cuts (rule 9):**
- **Length-bonus is CUT (default off).** It monotonically *hurt* held-out compression —
  greedily building long tokens overfits the training corpus. A disproven idea, kept as a
  configurable knob for a natural-text re-test.
- **Entropy-guard is marginal here (+0.12% at λ=1.0), kept on.** Synthetic templated text has
  near-degenerate branching entropy, so this under-tests BLT's principle. Its real value should
  show on natural morphology (BLT/H-Net evidence); flagged for the v1 natural-text eval, not
  claimed as a win yet.

**Bottom line:** Ada v0 = **byte-level two-stage superword BPE with a statistical entropy
guard** — a pure-stdlib tokenizer that beats plain BPE by ~43% compression at equal vocab,
equal speed, equal memory, and full losslessness. It reproduces SuperBPE's headline result
without SuperBPE's Rust fork, and folds in BLT's boundary principle as a CPU-only corpus
statistic. The neural donors (H-Net router, MrT5 gate, FSQ bottleneck) are the v1 upgrade
(`lab/ada/ada_train.ipynb`) that must beat this v0 baseline in ablation to earn inclusion.

---

## 6. Does Ada need a dataset? (answered 2026-07-07)

Yes — **raw text, no labels.** A tokenizer is trained *unsupervised*: it learns merge rules
(v0) or byte-boundary weights (v1) purely from the statistics of a text corpus. There is no
annotation step. The synthetic `corpus.py` here is only a stand-in to honor "no downloads on
this box" (rule 11); on Colab the training notebook streams a real corpus. Corpus quality and
domain coverage directly set tokenizer quality, so the corpus choice below is load-bearing.

## 7. (reserved)

## 8. Component sweeps (rule 13 — dated, re-verify at 30 days)

### Sweep 2026-07-07 — training corpus for Ada
Query themes: best open web pretraining corpus 2026; multilingual FineWeb-2 vs CulturaX; code
dataset The Stack v2 streaming. Winners (all HuggingFace, streaming, permissive licenses):
- **`HuggingFaceFW/fineweb-edu`** (`sample-10BT`) — English, classifier-quality-filtered web;
  the current default-best small-model corpus (SmolLM2 lineage). ODC-By. **Default.**
- **`HuggingFaceFW/fineweb-2`** — 8 TB, ~3T words, 1000+ languages / 1893 language-script pairs;
  beats CC-100, mC4, CulturaX, HPLT on the multilingual eval suite. ODC-By. Use per-script
  configs (`arb_Arab`, `zho_Hans`, `fra_Latn`, …) to stress byte-level multilingual robustness.
- **`bigcode/the-stack-smol`** — content-bearing source-code sample, 600+ languages. (NB the
  full `the-stack-v2-train-full-ids` ships only file *IDs* → needs Software Heritage S3 creds;
  the smol/`the-stack-dedup` variants stream actual content — a gotcha, logged in §10.)
- **Recipe for a byte tokenizer:** MIX English + code + one non-Latin script — that is exactly
  where byte-level tokenization beats subword BPE, so it is the honest stress test for Ada.

Wired into `lab/ada/ada_data.py` (`REGISTRY` + `DataConfig.mix`) and `lab/ada/ada_train.ipynb`.
Dated links: FineWeb (arXiv:2406.17557), FineWeb-2 (arXiv:2506.20920), StarCoder2/Stack v2
(arXiv:2402.19173), SmolLM2 data recipe (arXiv:2502.02737).

### Sweep 2026-07-07b — go aggressive + fix the gated-dataset crash
User hit `DatasetNotFoundError: bigcode/the-stack-smol is a gated dataset ... must be authenticated`.
Re-swept for the *highest-quality, most-recent, and crucially UN-gated* corpora, and rebuilt the
loader into a quality-tiered registry + resilient blender (`ada_data.py`, `SOURCES`/`RECIPES`):
- **Code (the fix):** `bigcode/the-stack-smol`, `-dedup`, `starcoderdata`, and Stack-v2 are all
  gated. **`codeparrot/github-code-clean`** (and `github-code`) are **ungated**, streaming, field
  `code`, 30+ languages — now Ada's default code source. (The `default` recipe no longer touches
  any gated dataset.)
- **Premium English web added:** `mlfoundations/dclm-baseline-1.0-parquet` (DCLM, 3.8T, highest
  macro-avg baseline); `nvidia/Nemotron-CC-HQ` (+5.6 MMLU vs DCLM, **gated → HF_TOKEN**);
  `openbmb/Ultra-FineWeb` (verified re-filter of FineWeb, arXiv:2505.05427); `ontocord/MixtureVitae`
  (permissive-first + reasoning, arXiv:2509.25531).
- **Math/reasoning added:** `HuggingFaceFW/finemath` (finemath-4plus/3plus), `open-web-math`,
  `nvidia/Nemotron-CC-Math` (133B tokens, beats FineMath, arXiv:2508.15096, **gated**).
- **Knowledge added:** `wikimedia/wikipedia` (`20231101.en`), Stack-Exchange.
- **Resilience:** each source tries several candidate configs + text-field names; gated/missing/
  errored sources are warned-and-skipped, and an empty result falls back to synthetic. So a gated
  dataset can never again crash a run. Gated sources auto-skip unless `HF_TOKEN` is set.
- **Recipes:** `default` (all-ungated blend: edu+code+multilingual+math+wiki), `aggressive`
  (adds DCLM/Ultra-FineWeb/premium), `multilingual`, `code-heavy`, `english`.
- Quality gates in the loader: min/max line length, exact-line dedup (blake2b), `min_alnum_frac`
  to drop markup/punctuation noise, shuffle.
Dated links: DCLM (arXiv:2406.11794), Nemotron-CC (developer.nvidia.com/blog/announcing-nemotron-cc),
Ultra-FineWeb (arXiv:2505.05427), MixtureVitae (arXiv:2509.25531), Nemotron-CC-Math (arXiv:2508.15096).

## 10. Stack & gotcha ledger (rule 14)

- **2026-07-07 — The Stack v2 `-train-full-ids` has no content.** It stores Software-Heritage
  blob IDs, not source; fetching text needs SWH S3 credentials.
- **2026-07-07 — the whole `bigcode/` code line is GATED** (`the-stack`, `the-stack-smol`,
  `the-stack-dedup`, `starcoderdata`, Stack-v2): `DatasetNotFoundError: ... is a gated dataset ...
  must be authenticated`. **Fix:** use ungated `codeparrot/github-code-clean` (field `code`).
  Ada's loader now defaults to it and marks every gated source `needs_token`, auto-skipping unless
  `HF_TOKEN` is set. General rule: never let a single dataset be load-bearing — the loader tries
  candidate configs/fields and skips on any error.
- **2026-07-07 — this container has no `datasets`/`numpy`/`torch`.** Real-corpus training and all
  v1 work happen on Colab; `ada_data.load_hf_corpus` imports `datasets` lazily so the module
  still imports here for the synthetic path.
