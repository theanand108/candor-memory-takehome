from __future__ import annotations

import json
import sys
from pathlib import Path

from .answering import answer_from_evidence
from .hybrid import HybridIndex
from .ingestion import load_units, parse_dt
from .temporal import build_temporal_view


def run_memory(input_path: str, output_path: str, data_dir: str = "data") -> None:
    units = load_units(data_dir)
    questions = []
    with open(input_path, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    # Build one temporal index per as_of so all retrieval candidates are
    # guaranteed to respect the evaluator's time boundary.
    indexes: dict[str, tuple[list, HybridIndex]] = {}
    outputs = []
    for question in questions:
        as_of = parse_dt(question["as_of"])
        key = question["as_of"]
        if key not in indexes:
            visible = build_temporal_view(units, as_of)
            indexes[key] = (visible, HybridIndex(visible))
        visible, index = indexes[key]
        hits = index.search(question["question"], limit=20)
        ranked = [hit.unit for hit in hits]

        # The evaluator scores the top 10, but the answer writer should see
        # the complete retrieved evidence budget. Multi-record questions often
        # have one decisive passage just outside the first ten; exposing all
        # twenty improves grounded synthesis without changing retrieval scores.
        answer, sources, abstained = answer_from_evidence(question["question"], ranked)
        outputs.append({
            "id": question["id"],
            "answer": answer,
            "sources": sources,
            "retrieved": [u.id for u in ranked],
            "abstained": abstained,
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: python -m candor_memory.cli <input.jsonl> <output.jsonl> <data_dir>", file=sys.stderr)
        raise SystemExit(2)
    run_memory(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
