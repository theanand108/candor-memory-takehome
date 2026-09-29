# Candor Memory Take-home

This repository contains the implementation for the Candor memory assessment.

## Current implementation

Milestone 1 establishes a deterministic temporal memory foundation:

- normalizes all supplied data sources into citable `MemoryUnit` records;
- preserves source, author/speaker, timestamps and stable IDs;
- applies the `as_of` visibility boundary before retrieval;
- handles Slack edits and deletions in the temporal view;
- performs deterministic BM25-style lexical retrieval;
- emits the required memory JSONL shape;
- applies a conservative baseline answer/abstention layer.

This is intentionally a baseline. Semantic retrieval, stronger reranking, temporal reasoning, LLM synthesis, and the action bonus will be added only after the baseline is evaluated against the supplied harness.

## Run

From the repository root:

```bash
./run.sh
```

Optional arguments:

```bash
./run.sh evals/memory_train.jsonl outputs/memory_train.answers.jsonl data
```

## Evaluation

After generating answers:

```bash
python3 eval_harness/score_retrieval.py \
  --gold evals/memory_train.jsonl \
  --answers outputs/memory_train.answers.jsonl

python3 eval_harness/score_memory.py \
  --gold evals/memory_train.jsonl \
  --answers outputs/memory_train.answers.jsonl \
  --judge none
```

## Design notes

The supplied brief defines retrieval as the primary memory concern. The implementation therefore treats temporal eligibility as a hard retrieval boundary rather than asking a language model to ignore future evidence after retrieval.

The challenge data is retained unchanged while development is in progress. Evaluation scripts under `eval_harness/` are treated as supplied assessment infrastructure.

## Status

Core ingestion and temporal retrieval baseline implemented. Evaluation-driven improvements are the next milestone.
