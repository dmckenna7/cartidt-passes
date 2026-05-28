#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/main.yaml}
OUT=${2:-./runs/$(date +%Y%m%d_%H%M%S)}

mkdir -p "${OUT}"

torchrun \
  --standalone \
  --nproc_per_node=4 \
  -m cartidt.driver.train \
  --config "${CONFIG}" \
  --out "${OUT}" \
  "${@:3}"
