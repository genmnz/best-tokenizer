# best-tokenizer — lab rules (repurposed)

This workspace mines the best-in-class **learned/compression tokenization** repos and papers
(2024–2026) to build **one** small, fast, CPU-friendly tokenizer — **Ada** — that steals the
strongest idea from each donor and measurably beats plain BPE at equal vocab, on the axes
that matter together: **accuracy (compression fidelity), speed, and memory footprint.**

## Layout

| Path | Contents |
|---|---|
| `best/` | The 6 cloned donor repos (code-only, shallow, LFS skipped). Pristine — read-only. Not committed; reproduce with `scripts/fetch_pool.sh`. |
| `papers/` | The paper pool (latest arXiv PDFs) backing the donors + greenfield techniques. Not committed; `scripts/fetch_pool.sh`. |
| `FINDINGS.md` | The single canonical research log. Every technique, trend, benchmark. |
| `ADA.md` | The charter for our model: donor roster, decision matrix per axis, plan. |
| `WORKPLAN.md` | The evaluated work order. |
| `lab/ada/` | OUR code: the Ada tokenizer, synthetic corpus, tests, benchmark harness. |
| `bench/` | Shared benchmark scripts / results. |

## Rules (the ones that carry over)

1. **Research first.** Read the paper and the exact code path before using a technique.
   Web-search for newer results and failure modes. Never trust README claims unreproduced.
2. **Everything goes in FINDINGS.md**, with citations (paper, `file:line`, URL). Always current.
3. **Test everything.** A technique is "understood" only when a script in `lab/` exercises it
   and a benchmark on THIS CPU-only box records real numbers.
4. **`best/` is pristine upstream.** Never modify vendored repos. All our work lives in `lab/`.
5. **Extract techniques, don't copy code.** Re-implement ideas in our own stack; cite origin;
   track licenses in FINDINGS.
6. **Compare on equal footing.** Same corpus, same hardware, same measurement script. Baseline
   (plain BPE) and Ada run through the *same* code path with flags toggled.
7. **Bottlenecks & breakthroughs are first-class.** For each donor: its one advantage, its wall.
8. **Name the trends.** When unrelated teams converge on an idea, log it in FINDINGS §3 and
   weight it heavily.
9. **Play around — invent.** Novel combinations are the point; a disproven idea *with numbers*
   is still a finding (FINDINGS §4).
10. **Extremely configurable.** Every knob (vocab size, transition point, entropy λ, seed,
    paths) is a named config field — no magic numbers, no hardcoded paths.
11. **No training on this box, no dataset downloads.** The laptop/container is for inference,
    benchmarks, and tests. Neural upgrades (Ada v1) train on Colab via a self-contained
    notebook; corpora are synthetic generators here, never downloads. The v0 statistical
    tokenizer trains in seconds on CPU from a synthetic corpus — that is allowed (no gradients).
12. **Freshness is maintained.** A load-bearing fact older than 30 days is re-verified before it
    backs a new decision.

## Target (all three met together, not traded off)

- **Compression beast** — fewer tokens per byte than plain BPE at the *same* vocab size.
- **Speed beast** — fast encode (MB/s) and fast train, pure-Python-or-better on CPU.
- **Small beast** — tiny merge table / memory footprint; simple, elegant design. Every donor
  technique must pay for its complexity with a measured win that survives ablation.
