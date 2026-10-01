# Documentation index

Read in this order if you are new: `architecture.md`, then `evaluation.md`, then the root
[RESULTS.md](../RESULTS.md) to see which claims the current evidence does and does not support.

## Design

| Doc | One line |
| --- | --- |
| [architecture.md](architecture.md) | Memory hierarchy, write/read/correction paths, and the Phase 1–8 build map. |
| [schema.md](schema.md) | Neo4j node labels, relationships, and key properties. |
| [corrections.md](corrections.md) | Soft, revisioned, audited edit/move/archive/merge/restore semantics. |
| [ui.md](ui.md) | The React Memory Explorer: what it renders and which API routes it uses. |
| [literature.md](literature.md) | Prior systems ScopeGraph borrows from, and what it deliberately does not claim. |

## Evaluation and evidence

| Doc | One line |
| --- | --- |
| [evaluation.md](evaluation.md) | How to run CrossScopeMem, the correction experiment, and the external benchmarks; metric definitions. |
| [benchmark_protocol.md](benchmark_protocol.md) | The binding protocol: conditions, statistics, sign convention, and dated protocol amendments. |
| [audit.md](audit.md) | Dated implementation audits (append-only): known gaps and run-by-run addenda. |
| [../RESULTS.md](../RESULTS.md) | Claims ledger: every reported number, its producing file and command, and its caveats. |
| [../results/README.md](../results/README.md) | Table of every run on disk (date, commit, dataset, systems, seeds, judge, N, status). |
| [../results/SCHEMA.md](../results/SCHEMA.md) | Every JSONL field and metric, with formulas and what each does not measure. |
| [../evals/README.md](../evals/README.md) | Command-level quick reference for the evaluation package. |
| [../data/SOURCES.md](../data/SOURCES.md) | Official sources and checksums for the three external benchmarks. |

## Working in the repo

| Doc | One line |
| --- | --- |
| [../README.md](../README.md) | Project overview, setup, and API usage. |
| [../CLAUDE.md](../CLAUDE.md) / [../AGENTS.md](../AGENTS.md) | Identical files: repo map, make targets, working rules, definition of done. |

## Conventions

- Results under `results/` are immutable once written; new analyses go to new directories.
- Synthetic (CrossScopeMem) and external (LoCoMo, LongMemEval-S, MemoryAgentBench) numbers are
  never combined, and latencies are only compared within one dataset and protocol.
