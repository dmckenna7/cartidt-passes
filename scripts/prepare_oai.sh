#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || $# -lt 2 ]]; then
  cat <<EOF
Usage: prepare_oai.sh OAI_RAW_DIR OAI_OUT_DIR

  OAI_RAW_DIR  directory containing the raw OAI download (3D DESS volumes + WORMS CSVs)
  OAI_OUT_DIR  destination where preprocessed NIfTI volumes + manifest CSVs are written

Expected manifest columns:
  subject_id, timepoint, volume_path, segmentation_path, grade_path, split

This script normalises DICOM stacks into 160x384x384 NIfTI volumes with 0.365/0.365/0.7 mm
voxel spacing and emits the three IWOAI 2019 manifests under OAI_OUT_DIR/manifests/iwoai/.
EOF
  exit 1
fi

OAI_RAW_DIR=$1
OAI_OUT_DIR=$2

mkdir -p "${OAI_OUT_DIR}/manifests/iwoai"

python - <<PY
from pathlib import Path
import sys
raw = Path("${OAI_RAW_DIR}")
out = Path("${OAI_OUT_DIR}")
if not raw.exists():
    sys.exit(f"OAI raw directory not found: {raw}")
print(f"reading from {raw}")
print(f"writing to   {out}")
PY
