# Evaluation

Phase 7 provides an executable, credential-free harness around the shared `MemorySystem` interface. CrossScopeMem creates deterministic global, project, and session memories with scope interference, temporary overrides, and (at higher difficulty) updates and distractors. Phase 9 adds a reproducible smoke command and graph-backed demo/export scripts. The generator records structured gold answers and memory content; it does not fabricate external benchmark data.

Run the four architecture variants with the same histories, seed, top-k, token budget, and deterministic hash-bucket embedder:

```bash
make eval-all
make eval-report
make smoke
```

For a meaningful local comparison, increase the generated histories and keep the
same difficulty across systems:

```bash
make eval-all SCENARIOS=40 DIFFICULTY=3
make eval-report
```

`SCENARIOS` defaults to 1 for a fast smoke run. `SYSTEMS` can be narrowed to a
comma-separated subset, for example `SYSTEMS=scopegraph,vector_memory`.

The raw JSONL record preserves retrieved IDs, scopes, scores, status, trace paths, latency, token count, logical storage statistics, configuration hash, seed, and git commit. `evals.analysis.aggregate` scores raw records independently of execution; `tables` and `plots` write Markdown and SVG artifacts.

Implemented metrics include exact match, normalized token F1, Precision@K, Recall@K, Cross-Scope Contamination Rate, scope-classification accuracy, stale-memory rate, p50/p95 latency, token summaries, and logical storage counts. The correction runner measures error relapse after no correction, conversational correction, and direct graph correction at +1, +5, +10, and +20 sessions.

No external benchmark results are claimed. External adapters cover LongMemEval,
LongMemEval-V2, LoCoMo, MemConflict, MemoryAgentBench, RHELM, MemBench,
Mem2ActBench, and TIME. Releases remain local and must pass validation before replay.
