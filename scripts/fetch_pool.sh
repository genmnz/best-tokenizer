#!/usr/bin/env bash
# Reproduce the research pool: 6 donor repos in best/ and 9 papers in papers/.
# Repos are code-only (shallow, LFS skipped) — we study techniques, not weights.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p best papers

echo "== cloning donor repos into best/ =="
export GIT_LFS_SKIP_SMUDGE=1
clone() { [ -d "best/$2" ] || git clone --depth 1 --single-branch "$1" "best/$2"; }
clone https://github.com/facebookresearch/blt        blt
clone https://github.com/goombalab/hnet              hnet
clone https://github.com/OpenEvaByte/evabyte         evabyte
clone https://github.com/PythonNut/superbpe          superbpe
clone https://github.com/jkallini/mrt5               mrt5
clone https://github.com/NVIDIA/Cosmos-Tokenizer     cosmos-tokenizer

echo "== downloading papers into papers/ =="
dl() { [ -s "papers/$2" ] || curl -sSL --max-time 120 -o "papers/$2" "https://arxiv.org/pdf/$1"; }
dl 2412.09871 BLT_byte-latent-transformer.pdf
dl 2507.07955 HNet_dynamic-chunking.pdf
dl 2508.05628 HNetpp_persian-morphology.pdf
dl 2503.13423 SuperBPE_superword-tokens.pdf
dl 2410.20771 MrT5_dynamic-token-merging.pdf
dl 2511.20849 LengthMAX_tokenizer.pdf
dl 2511.11346 MultiByte_probabilistic-circuits.pdf
dl 2510.18234 DeepSeekOCR_optical-compression.pdf
dl 2309.15505 FSQ_finite-scalar-quantization.pdf

echo "== done =="
du -sh best/* papers 2>/dev/null || true
