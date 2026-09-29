#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:-evals/memory_train.jsonl}"
OUTPUT="${2:-outputs/memory_train.answers.jsonl}"
DATA="${3:-data}"

PYTHONPATH=. python3 -m candor_memory.cli "$INPUT" "$OUTPUT" "$DATA"
printf 'Wrote %s\n' "$OUTPUT"
