# Scopegraph: faster ingestion for Mem0

Repo: `/Users/rahulsuthar/Downloads/Scopegraph` (`src/memstudy`). Status: the runner can ingest several histories at once (`--ingest_workers N`) for an arm that declares `concurrent_prepare`, but no arm does yet. Mem0 still ingests serially (shared local Qdrant client and instance-global cost sinks make concurrent `prepare` unsafe without a per-worker Memory pool).

## Problem
- `runner.py` ingests histories one at a time unless the arm declares `concurrent_prepare`.
- `--workers` only parallelizes questions inside a history, after `arm.prepare(history)` has finished.
- `Mem0Arm.prepare` (`arms/mem0_arm.py`) does serial `memory.add` calls (one LLM extraction each).

## Safe changes (results unchanged, no PREREG edit needed)
1. **Ingest different histories concurrently.** Each history has its own `user_id`, so nothing is shared.
   - The local Qdrant path store allows one client at a time per process. Either use one store dir per history, or share one client and serialize only the Qdrant calls with a lock. Slow LLM extraction still overlaps.
2. **Pipeline ingestion with querying.** Ingest history N+1 in a background thread while history N is answered and judged (the runner already does this for arms with `concurrent_prepare`).

## Changes that alter results (need a PREREG note, avoid by default)
- Mem0 adds within one history in parallel: extraction compares against already-stored memories, so order changes what is stored.
- Larger `turns_per_add` or `write_granularity: session`: different arm from the registered ten-turn default.

## Implementation plan for Mem0
1. Make `Mem0Arm` safe for concurrent `prepare` on different histories:
   - Guard shared mutable state with a lock or per-history keys.
   - `Meter.use_sink` and `ingest_sink.snapshot()/since()` assume one ingest at a time. Make the sink per-call or thread-local, otherwise per-history ingest cost is misattributed.
   - `Budget.guard/charge` must be thread-safe.
2. Set `concurrent_prepare = True` on the arm once the above holds.
3. Respect API rate limits: reuse the existing transient-error retry. Start at 4 workers and watch for 429s.

## Verify
- Run a 2-3 history subset with `--ingest_workers 1` and `4`; stored-unit counts and ingest cost per history should match (Mem0 may differ slightly from LLM nondeterminism).
- Check `results/ledger.jsonl` totals still equal the sum of per-history ingest costs.
- Compare wall-clock ingest seconds in the run summary.
