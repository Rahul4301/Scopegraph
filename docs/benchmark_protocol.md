# Benchmark Protocol

Every backend will receive identical histories, queries, embedding and answer models, prompts, temperature, top-k budget, token budget, and scoring. Configuration, random seed, model identifiers, timestamp, git commit, and configuration hash will be stored with each run. Generated benchmark data must not be presented under an external benchmark name. External datasets will require validation and documented acquisition steps.

Phase 7's offline runner uses a deterministic hash-bucket embedding provider and a transparent fact-value extractor so benchmark plumbing can be tested without API keys. These are harness defaults, not research claims about answer quality. External adapters normalize local LongMemEval, LongMemEval-V2, LoCoMo, MemConflict, MemoryAgentBench, RHELM, MemBench, Mem2ActBench, and TIME releases into the same session model; validation happens before replay and source files remain outside the repository. Raw execution and deterministic scoring are separate commands:

```text
run_eval -> results/raw/*.jsonl -> run_report -> processed/tables/figures
```

The correction-persistence runner uses the same in-memory ScopeGraph correction service as the API and records relapse at fixed future-session offsets. It does not require the UI or a human-subject study.
