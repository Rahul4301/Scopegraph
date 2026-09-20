# Evaluation

Phase 7 provides an executable, credential-free harness around the shared `MemorySystem` interface. CrossScopeMem creates deterministic global, project, and session memories with scope interference, temporary overrides, and (at higher difficulty) updates and distractors. The generator records structured gold answers and memory content; it does not fabricate external benchmark data.

Run the four architecture variants with the same histories, seed, top-k, token budget, and deterministic hash-bucket embedder:

```bash
make eval-all
make eval-report
```

The raw JSONL record preserves retrieved IDs, scopes, scores, status, trace paths, latency, token count, logical storage statistics, configuration hash, seed, and git commit. `evals.analysis.aggregate` scores raw records independently of execution; `tables` and `plots` write Markdown and SVG artifacts.

Implemented metrics include exact match, normalized token F1, Precision@K, Recall@K, Cross-Scope Contamination Rate, scope-classification accuracy, stale-memory rate, p50/p95 latency, token summaries, and logical storage counts. The correction runner measures error relapse after no correction, conversational correction, and direct graph correction at +1, +5, +10, and +20 sessions.

No external benchmark results are claimed. Phase 8 owns LongMemEval, LoCoMo, and MemConflict acquisition/adapters.
