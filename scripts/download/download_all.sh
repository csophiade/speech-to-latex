#!/usr/bin/env bash
# Download every dataset used by the project into data/. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"
PY="../../.venv/bin/python"
for s in download_processbench download_mathspeech download_mathbridge download_speech2latex; do
    echo "===== $s"
    "$PY" "$s.py"
done
