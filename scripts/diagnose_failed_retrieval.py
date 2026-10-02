from __future__ import annotations

import json
from pathlib import Path

from candor_memory.hybrid import HybridIndex
from candor_memory.ingestion import load_units, parse_dt
from candor_memory.state_rerank import rerank
from candor_memory.temporal import build_temporal_view

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "evals" / "memory_train.jsonl"
TARGETS = {"MEM-TR-02", "MEM-TR-04", "MEM-TR-20", "MEM-TR-21", "MEM-TR-25"}

units = load_units(str(ROOT / "data"))
questions = [json.loads(line) for line in GOLD.read_text().splitlines() if line.strip()]

for q in questions:
    if q["id"] not in TARGETS:
        continue
    visible = build_temporal_view(units, parse_dt(q["as_of"]))
    index = HybridIndex(visible)
    hits = index.search(q["question"], limit=20)
    ranked = rerank(q["question"], [h.unit for h in hits])
    needed = {item for group in q.get("needed", []) for item in group}
    print(f"\n=== {q['id']} ===")
    print(q["question"])
    print("needed:", sorted(needed))
    print("top 20 after rerank:")
    hit_scores = {h.unit.id: h.score for h in hits}
    for i, unit in enumerate(ranked, 1):
        mark = " <== NEEDED" if unit.id in needed else ""
        print(f"{i:2d} {hit_scores.get(unit.id, 0.0):.6f} {unit.id} [{unit.source}] {unit.text[:180]!r}{mark}")
