#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/main.yaml}
CKPT=${2:?usage: launch_eval.sh CONFIG CKPT [OUT]}
OUT=${3:-./eval/$(basename "${CONFIG%.yaml}")}

mkdir -p "${OUT}"

python -m cartidt.driver.evaluate \
  --config "${CONFIG}" \
  --ckpt "${CKPT}" \
  --out "${OUT}"
