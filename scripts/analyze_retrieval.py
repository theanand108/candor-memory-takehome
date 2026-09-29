from __future__ import annotations

import argparse
import json
from pathlib import Path

from candor_memory.ingestion import load_units, parse_dt
from candor_memory.search import LexicalIndex
from candor_memory.temporal import build_temporal_view


def flatten_needed(value: list[list[str]]) -> set[str]:
    out: set[str] = set()
    for group in value:
        out.update(group)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="evals/memory_train.jsonl")
    parser.add_argument("--data", default="data")
    parser.add_argument("--ids", nargs="*", default=[])
    args = parser.parse_args()

    wanted = set(args.ids)
    units = load_units(args.data)
    by_id = {u.id: u for u in units}

    with open(args.gold, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    for q in questions:
        if wanted and q["id"] not in wanted:
            continue
        visible = build_temporal_view(units, parse_dt(q["as_of"]))
        index = LexicalIndex(visible)
        hits = index.search(q["question"], limit=20)
        ranked = [h.unit for h in hits]
        needed = flatten_needed(q.get("needed", []))
        found = needed & {u.id for u in ranked[:10]}
        if wanted or len(found) != len(needed):
            print("=" * 100)
            print(q["id"], q["category"])
            print("QUESTION:", q["question"])
            print("AS_OF:", q["as_of"])
            print("NEEDED:", sorted(needed))
            print("FOUND_TOP10:", sorted(found))
            print("MISSING:", sorted(needed - found))
            print("GOLD:", q.get("gold_answer"))
            print("TOP20:")
            for i, unit in enumerate(ranked, 1):
                marker = "<-- NEEDED" if unit.id in needed else ""
                print(f"{i:02d} {unit.id:35} {unit.source:12} {marker}")
                print("   ", unit.text.replace("\n", " ")[:240])


if __name__ == "__main__":
    main()
