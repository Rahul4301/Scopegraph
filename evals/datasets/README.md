# Dataset placement

Place licensed or downloaded datasets under their named subdirectories. Do not commit benchmark files.

Phase 8 adapters accept local releases and validate them before replay:

```bash
make validate-external DATASET=longmemeval DATA_PATH=data/longmemeval/longmemeval_s_cleaned.json
make validate-external DATASET=locomo DATA_PATH=data/locomo/locomo10.json
make validate-external DATASET=memconflict DATA_PATH=data/memconflict/release.json
make validate-external DATASET=longmemeval_v2 DATA_PATH=data/longmemeval_v2/release.jsonl
make validate-external DATASET=memoryagentbench DATA_PATH=data/memoryagentbench/release.json
make validate-external DATASET=rhelm DATA_PATH=data/rhelm/normalized.jsonl
make validate-external DATASET=membench DATA_PATH=data/membench/release.json
make validate-external DATASET=mem2actbench DATA_PATH=data/mem2actbench/release.jsonl
make validate-external DATASET=time DATA_PATH=data/time/release.json
```

Acquire LongMemEval from its [official repository](https://github.com/xiaowu0162/longmemeval), LoCoMo from the [official repository](https://github.com/snap-research/locomo), and MemConflict from the [official repository](https://github.com/TaoZhen1110/MemConflict). The MemConflict adapter uses a documented local interchange shape because releases vary; it never fabricates records.

Additional supported releases are LongMemEval-V2, MemoryAgentBench, Microsoft
RHELM, MemBench, Mem2ActBench, and TIME. Some upstream releases split questions
from conversations, email, files, or trajectories. Join those source files into
one JSON/JSONL record per question before validation. The normalized interchange
accepts:

```json
{
  "id": "question-id",
  "question": "What changed?",
  "answer": "The current answer",
  "question_type": "knowledge_update",
  "question_date": "2026-01-03",
  "supporting_evidence": ["session-2:0"],
  "sessions": [
    {
      "id": "session-1",
      "date": "2026-01-01",
      "messages": [{"role": "user", "content": "The earlier fact."}]
    }
  ]
}
```

Accepted history aliases include `history`, `conversation`, `context`,
`timeline`, `events`, and `episodes`; accepted answer aliases include
`gold_answer`, `target`, and `reference_answer`. Validation rejects records that
omit the actual history rather than silently fabricating benchmark context.
