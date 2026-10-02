# Candor Memory Take-home

A temporal memory system for the Candor take-home assessment. It ingests the supplied work-life data, builds an as-of-safe evidence view, retrieves relevant records with hybrid lexical + semantic search, expands related evidence chains, and produces grounded JSONL answers with source IDs and abstention support.

## What the system is designed to solve

The assessment is primarily a **retrieval problem**: the answer writer can only be correct if the system retrieves the records needed to answer the question. The implementation therefore treats temporal correctness, retrieval quality, evidence grounding, and abstention as first-class concerns.

The system explicitly handles:

- temporal `as_of` boundaries;
- Slack edits and deletions;
- source/author/speaker metadata and stable record IDs;
- lexical + dense semantic retrieval;
- cross-record evidence chains;
- current-state questions where facts changed over time;
- commitment/delivery questions;
- cross-source questions such as dictation → email/Slack;
- prompt-injection-like content inside memory as untrusted data;
- grounded answers with cited source IDs;
- abstention when the requested fact is not available.

## Architecture

```text
                    data/
                      |
                 ingestion.py
                      |
              MemoryUnit records
                      |
             temporal.py
                      |
          as_of-filtered memory view
                      |
              +-------+-------+
              |               |
         lexical BM25     SentenceTransformer
              |               |
              +-------+-------+
                      |
                 hybrid.py
                      |
        RRF fusion + intent priors
        + structural state anchors
        + evidence-chain expansion
                      |
                state_rerank.py
                      |
              final evidence ranking
                      |
                answering.py
                      |
        grounded answer + sources + abstain
                      |
              memory_train.answers.jsonl
```

### 1. Ingestion and normalization

`candor_memory/ingestion.py` converts the supplied sources into a common `MemoryUnit` representation while preserving source type, stable IDs, timestamps, authors/speakers, record relationships, and searchable metadata.

### 2. Temporal memory

`candor_memory/temporal.py` constructs the visible memory for each question's `as_of` timestamp. Retrieval therefore never receives records that were not available at the requested point in time. Edits and deletions are handled in the temporal view rather than delegated to the answer model.

### 3. Hybrid retrieval

`candor_memory/hybrid.py` combines:

- deterministic lexical retrieval;
- dense semantic retrieval using `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`;
- reciprocal-rank fusion;
- small intent-specific priors;
- structural state anchors for patterns such as current launch state, delivery status, launch-slip causes, and travel-day calendar joins;
- bounded expansion of related passages within the same record and through explicit target relationships.

The semantic layer is optional. If `sentence-transformers` is unavailable, the deterministic lexical path remains usable.

### 4. Final reranking and answer generation

`candor_memory/state_rerank.py` applies deterministic state-aware reranking. `candor_memory/answering.py` then selects grounded evidence and produces the required JSONL answer shape. The answer layer is deliberately constrained rather than asking a model to reconstruct temporal truth from an unrestricted corpus.

### 5. Prompt-injection safety

Memory content is treated as **data, never instructions**. Retrieval and answer construction explicitly avoid evidence that contains known instruction-injection markers. Secrets in the supplied data are not reproduced.

## Key decisions

### Temporal filtering before retrieval

The most important correctness rule is that nothing after `as_of` exists. Instead of retrieving everything and asking an LLM to ignore future evidence, the implementation builds a temporal view first and indexes only that view.

### Hybrid retrieval instead of semantic search alone

The data contains exact names, dates, IDs, commitments, edits, and short messages. Lexical matching is valuable for these precise signals, while dense retrieval helps with paraphrased questions. RRF combines both without allowing either channel to completely replace the other.

### Deterministic state anchors

Generic similarity can rank related background chatter above the decisive state transition. The state-anchor layer adds small, explicit boosts for recognizable evidence structures without hard-coding the train-set answers. Examples include later launch-state records, actual delivery messages, launch-regression evidence, and the flight-date/calendar relationship.

### Evidence-chain expansion

Some questions require more than one record. The retriever therefore expands from strong candidates to bounded sibling passages and explicit target relationships while preserving the original ranking signal.

### Grounding over verbosity

The answer layer is designed to return concise, source-grounded answers rather than reproducing large portions of retrieved records. Source IDs remain part of the output contract.

## What did not work / known limitations

Development was evaluation-driven. Several approaches improved individual cases but were rejected when they caused broader regressions.

- A purely lexical baseline was insufficient for paraphrased and multi-source questions.
- Semantic retrieval alone could surface relevant-looking but non-decisive evidence.
- Generic reranking could bury the decisive state transition behind earlier planning chatter.
- A final answer-layer experiment intended to preserve decisive details in long evidence did **not** improve retrieval and reduced the offline answer score, so it was reverted. The submitted branch contains that experiment and an explicit revert in Git history; the implementation therefore returns to the stronger measured state before that experiment.
- The current offline answer scorer still exposes weaknesses on some temporal, commitment-status, reported-speech, disagreement, abstention, and prompt-injection cases. These are documented by the supplied harness rather than hidden.

## Running

The required interface is a single command from the repository root:

```bash
./run.sh
```

Defaults:

```text
evals/memory_train.jsonl -> outputs/memory_train.answers.jsonl
data/
```

Optional arguments:

```bash
./run.sh evals/memory_train.jsonl outputs/memory_train.answers.jsonl data
```

The script invokes the package with `PYTHONPATH=.` and writes one JSON object per question.

## Evaluation

Generate the train-set output first:

```bash
./run.sh
```

Then run the supplied scorers:

```bash
python3 eval_harness/score_retrieval.py \
  --gold evals/memory_train.jsonl \
  --answers outputs/memory_train.answers.jsonl

python3 eval_harness/score_memory.py \
  --gold evals/memory_train.jsonl \
  --answers outputs/memory_train.answers.jsonl
```

### Latest measured train-set result

The final submitted implementation was measured at:

| Metric | Result |
|---|---:|
| Retrieval score | **84.0%** |
| Retrieval MRR | **0.5782** |
| Exact passage: top 5 / 10 / 20 | **56% / 84% / 84%** |
| Whole record: top 5 / 10 / 20 | **64% / 88% / 88%** |
| Nothing needed in top 20 | **4%** |
| Forbidden records in top 10 | **0** |
| Forbidden records in top 20 | **0** |
| Answer strict score (`judge none`) | **51.8%** |
| Answer lenient score (`judge none`) | **59.3%** |
| Unverified answers | **2** |
| Hard failures | **0** |
| Source recall | **0.5133** |
| Source precision | **0.6667** |

The answer scorer's 95% intervals are reported by the supplied harness. With only 27 train questions, individual percentage-point changes should be interpreted cautiously.

### Remaining answer-level failures

The latest `score_memory.py --judge none` run flags several cases where the answer does not preserve a decisive evaluator term, including:

- launch dates and historical launch states;
- commitment/delivery status (`sent`);
- exact numeric details;
- reported-speech attribution;
- disagreement participants;
- temporal numeric details;
- abstaining when the requested fact is absent;
- exact technical terms such as `PostGIS`;
- the calendar/board detail for the Denver travel question;
- preserving a prompt-injection safety fact such as `not signed`.

The retrieval layer nevertheless reaches **84%** on the supplied train retrieval evaluation, with no forbidden records retrieved in the measured top-10/top-20 sets.

## Output contract

For each input question the system emits:

```json
{
  "id": "MEM-TR-01",
  "answer": "...",
  "sources": ["..."],
  "retrieved": ["..."],
  "abstained": false
}
```

`retrieved` contains the ranked evidence IDs handed to the answer layer; the evaluator scores the top 10. `sources` contains the records actually relied upon by the answer.

## Tools and models used

- Python 3
- `sentence-transformers>=5.0,<7`
- `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` for dense retrieval
- supplied Python evaluation harness
- Git/GitHub for versioning
- local development on macOS

No paid external API is required for the memory pipeline. The semantic model is downloaded from the Hugging Face Hub when it is not already cached.

## Bonus action system

The optional VoiceOS/TextOS action bonus was **not implemented**. The submission focuses on the required memory system and its retrieval/answer evaluation.

## Submission commit

The final submission commit is recorded in Git history. The last answer-layer experiment was explicitly reverted after regression testing; no further experimental changes are part of the submitted implementation.
