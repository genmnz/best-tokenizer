# WORKPLAN — best-tokenizer

Evaluated work order. Follow it; skipping ahead requires a note in FINDINGS.

## Phase 0 — pool (DONE 2026-07-07)
- [x] Clone 6 donor repos into `best/` (code-only, shallow, LFS skipped): BLT, H-Net, EvaByte,
      SuperBPE, MrT5, Cosmos-Tokenizer.
- [x] Download 9 papers into `papers/` (donors + Length-MAX + multi-byte circuits + DeepSeek-OCR + FSQ).

## Phase 1 — research (DONE 2026-07-07)
- [x] Read the exact code path implementing each donor's breakthrough (FINDINGS §1, file:line).
- [x] Name the cross-repo trends (FINDINGS §3): dynamic boundaries, compress-before-model,
      kill-the-codebook, small-beats-big.

## Phase 2 — Ada v0 (DONE 2026-07-07)
- [x] Build the statistical tokenizer: byte 2-stage superword BPE + entropy guard + length term.
- [x] Synthetic corpus generator (no downloads). Config-driven (rule 10).
- [x] Test suite 7/7 (lossless, unicode, determinism, superword-crossing, save/load, compression).
- [x] Equal-footing ablation benchmark. **Result: +43.4% bytes/tok vs BPE, equal speed/memory,
      lossless** (FINDINGS §5).
- [x] Honest cuts: length-bonus CUT (hurt); entropy-guard kept but marginal on synthetic text.

## Phase 3 — Ada v0 hardening (NEXT)
- [ ] Re-evaluate entropy-guard on a *natural* corpus (Colab-side generator or allowed sample),
      since synthetic templated text under-tests branching entropy. Decide keep/tune/cut with numbers.
- [ ] Throughput: profile the Python heap encoder; prototype a batched/vectorized or Rust-port
      encode. Target ≥ baseline MB/s at the +43% compression.
- [ ] Vocab-size sweep (512 / 1024 / 4096 / 16k) — compression vs vocab curve vs plain BPE.
- [ ] Add a real BPE reference (HF `tokenizers`/`tiktoken`) as an external sanity baseline once
      a venv is available; confirm our from-scratch BPE matches it within tolerance.

## Phase 4 — Ada v1 neural (Colab)
- [ ] Run `lab/ada/ada_colab_train.ipynb` on Colab: H-Net router + FSQ bottleneck + MrT5 gate.
- [ ] Bring back checkpoint + report → `lab/ada/COLAB_RUNS.md` (newest first, verdict + next tweak).
- [ ] Ablate each neural donor; keep only what beats v0's bits-per-byte / bytes-per-token.
- [ ] Freshness re-sweep (rule 12) before the run: "what changed since 2026-07-07?" for each axis.

## Definition of done (Ada)
A single small tokenizer that, at equal vocab, beats plain BPE on compression **and** matches it
on encode speed **and** memory, with every retained technique surviving ablation.
