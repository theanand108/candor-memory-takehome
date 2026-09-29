#!/bin/sh
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" exec python3 scripts/analyze_retrieval.py "$@"
