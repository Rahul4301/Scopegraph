# Future Work: Extending "Memory Layer or Full Context?"

Proposed follow-on studies, to be scoped with the faculty mentor after the main study (arms A, B, C on LoCoMo and LongMemEval-S) is complete. Both extensions reuse the main study's harness, reader, grader, and pre-registration discipline, so each can be added without redesigning the core comparison.

## Extension 1: Add Supermemory (arm D) as a second memory system

**Status: deferred (backburner).** A Supermemory API key is not affordable at this time, so this extension is on hold until funding or credits are available. Nothing in the main study depends on it. The Supermemory code has been removed from the main codebase and lives only in git history (see below).

### Motivation

The main study tests one memory system, Mem0. Its stated limitation is that conclusions apply to Mem0 only. Supermemory is a different design: a hosted service with its own extraction, memory graph, and hybrid search, rather than an open-source library we run locally. If full context is non-inferior to Mem0 and also to Supermemory, the finding no longer depends on one vendor's architecture. If the two systems disagree, that is a finding in itself.

### Question and hypotheses

Does the headline result (H1, H3 from the main proposal) hold for a second, architecturally different memory system?

| ID | Hypothesis | Test | Falsified if |
|----|------------|------|--------------|
| F1 | No memory (arm A) is non-inferior to Supermemory on LoCoMo (categories 1 to 4) and LongMemEval-S. | Paired non-inferiority, 3 point margin, same as H1. | Lower 95% bound of (A minus Supermemory) falls below the margin, or Supermemory is significantly better. |
| F2 | Plain RAG (arm C) is within the margin of Supermemory. | Paired non-inferiority, same as H3. | Supermemory exceeds RAG by more than the margin. |
| F3 | Supermemory and Mem0 differ by less than the margin. | Paired equivalence test between the two memory arms. | The two systems differ by more than the margin in either direction. |

The margin and tests are fixed before data collection, as in the main study.

### Design

- Same reader (GPT-6 Luna), same grader (GPT-5 nano), same questions, same answer template.
- Same shared retrieval token budget as arms B and C (7,000 tokens, whole items in rank order from 200 candidates), so the comparison isolates how each system builds and ranks memories, not how much it is allowed to show the reader.
- One Supermemory container per history, hybrid search, memory-oriented ingestion, version-pinned SDK (`supermemory==3.62.0`).
- Ingestion cost and per-query cost recorded separately and fed to the same cost-crossover model used for H2, so Supermemory gets its own crossover estimate.

### Status of the code

The arm, its offline tests (against a fake service), its rate-card pricing and its `g2_supermemory` gate were removed from the main branch so the main study ships without it. They are recoverable from commit `f46e569`: `src/memstudy/arms/supermemory_arm.py`, `tests/test_supermemory.py`, the `services.supermemory` block in `configs/prices.yaml`, the arm D block in `configs/study.yaml`, and `SupermemoryPrice` in `src/memstudy/budget.py`. Reviving it means restoring those, adding the arm to the CLI and `SYSTEMS` in `src/memstudy/execute.py`, and extending it to MemoryAgentBench's single-document histories (one service document per chunk). It has never called the real service, so a small live check is needed before any paid run.

### Cost and scope decision

Estimated from the published rate card (not yet measured):

| Benchmark | Estimated Supermemory cost | Note |
|-----------|----------------------------|------|
| LoCoMo | about $1 | 10 histories, 10K to 21K tokens each |
| LongMemEval-S | about $260 | about 51.9M ingested tokens at $5 per 1M; every question has its own haystack, so nothing is shared |

LongMemEval-S dominates the cost because histories cannot be reused. Options to put to the mentor:

1. LoCoMo only (about $1). Cheap, but only ten histories, so intervals are wide.
2. LoCoMo plus a fixed random subset of LongMemEval-S (for example 100 of 500 questions, seed fixed in advance). Roughly $52 for the subset at the same rate.
3. Full LongMemEval-S, which needs a raised budget cap or outside funding.

Recommendation: option 2, with the subset size chosen from a power calculation on the pilot data so the non-inferiority test is not underpowered.

### Risks

- Hosted service behavior and pricing can change between pilot and final run. Pin versions and record the run date.
- Supermemory's own ingestion is asynchronous, so ingestion completion must be confirmed before querying, or results will reflect partial indexing.
- A deprecated or withdrawn extraction model upstream would invalidate comparability, so the model versions in use are recorded for every run.

## Extension 2: VibeMemBench (memory for coding agents)

### What it is

VibeMemBench (Fan et al., arXiv:2609.23570, September 2026) evaluates memory systems on real repository coding tasks. Per its abstract: 111 coding targets from 90 SWE-rebench V2 repositories, with 3,634 history trajectories from those repositories. Targets are bug fixes, feature requests, interface changes, and configuration work. An agent edits the codebase under a declared memory condition and executable tests decide whether the task is resolved. Each target is kept only if injecting its history experience improves the executable outcome in a reference setting.

Reported headline: injecting verified experience directly raises task resolution on four of five held-out solvers by 1.1 to 4.5 percentage points and lowers agent steps on all five. But when four existing memory systems must build and retrieve experience from the same history, eleven of twelve solver and system pairings fail to beat the memory-off baseline.

### Why it fits this project

The main study asks whether a memory layer is needed when history fits in context, using conversational benchmarks, and lists "no coding or repository workloads" as a limitation. VibeMemBench tests the same question on a different workload and measures a downstream outcome (tests pass) rather than answer accuracy judged by a model. It also has a clean place for the full-context arm: the history trajectories can be placed directly in the prompt when they fit, which the benchmark's own memory systems do not do.

### Question and hypotheses

Does a full-context, no-memory arm match or beat memory systems on repository coding tasks when the relevant history fits in context?

| ID | Hypothesis | Test | Falsified if |
|----|------------|------|--------------|
| V1 | Full-history context is non-inferior to a memory system on task resolution rate. | Paired non-inferiority on per-target resolved/unresolved outcomes, margin set from pilot variance. | Memory system significantly better, or lower bound below margin. |
| V2 | Plain retrieval over verbatim trajectory chunks is within the margin of an extraction-based memory system. | Paired non-inferiority. | Extraction exceeds verbatim retrieval by more than the margin. |
| V3 | Memory arms use fewer agent steps or tokens than full context at equal resolution rate. | Paired comparison of steps and measured cost per target. | No reduction in steps or cost. |

Arms would mirror the main study: A (full history in prompt where it fits), B (Mem0), C (verbatim RAG), plus the benchmark's memory-off baseline. Supermemory would be added only if Extension 1 is later funded.

### Open items to verify before committing

These could not be confirmed from the abstract alone and need a read of the full paper and repository:

- Whether the dataset, harness, and verified experiences are publicly released, and under what license.
- How large the trajectory history per target is in tokens, and how many targets fit in the reader's context window without truncation. Targets that do not fit would be reported as excluded, not cut.
- Which solver agents and scaffold the benchmark uses, and whether a single fixed solver can be run on our budget.
- Compute requirements: each target needs an executable repository environment and repeated agent runs, which is far more expensive per item than a conversational question.

### Cost and scope

Not yet estimated. Agent runs on 111 targets across several arms are likely to be the most expensive part of any future work in this project, so a cost estimate from a small pilot (a handful of targets, one arm) must come before any commitment. A reasonable fallback scope is one fixed solver, a fixed subset of targets, and two or three arms.

### Risks

- The benchmark is a very recent preprint and not yet peer reviewed as far as we could tell, so its construction details may change.
- Coding outcomes are noisy (one run per target is a weak estimate), so repeated runs or a larger margin may be needed, which raises cost.
- Execution environments (Docker images, repo dependencies) are a significant engineering burden, which is why the earlier coding benchmark work was removed from the main study.

## Suggested order

1. Finish the main study as pre-registered.
2. VibeMemBench pilot after reading the full paper and confirming data availability, with go or no-go decided on measured pilot cost. Run it with arms A, B, and C only, since Supermemory is deferred.
3. Supermemory (backburner, funding permitting): LoCoMo first (about $1) after one live smoke test, then the LongMemEval-S subset once its size is set by power analysis. Free options to explore before paying: vendor startup, research, or student credits, or a free tier if one covers LoCoMo.

## References

- Fan, L., Shi, Y., Li, Y., Sun, C., Chen, X., Xu, X., Wei, H., Ni, S., Yang, M., and Ye, J. (2026). VibeMemBench: Evaluating Memory Systems for Coding Agents on Real Repository Coding Tasks. arXiv:2609.23570. https://arxiv.org/abs/2609.23570
- Main study references (Mem0, LoCoMo, LongMemEval, and related work): see the research proposal.
