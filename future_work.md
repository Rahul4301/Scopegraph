# Future Work: Extending "Memory Layer or Full Context?"

A proposed follow-on study, to be scoped with the faculty mentor after the main study (arms A, B, C on LoCoMo and LongMemEval-S) is complete. The extension reuses the main study's harness, reader, grader, and pre-registration discipline, so it can be added without redesigning the core comparison.

## Extension: VibeMemBench (memory for coding agents)

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

Arms would mirror the main study: A (full history in prompt where it fits), B (Mem0), C (verbatim RAG), plus the benchmark's memory-off baseline.

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
2. VibeMemBench pilot after reading the full paper and confirming data availability, with go or no-go decided on measured pilot cost. Run it with arms A, B, and C.

## References

- Fan, L., Shi, Y., Li, Y., Sun, C., Chen, X., Xu, X., Wei, H., Ni, S., Yang, M., and Ye, J. (2026). VibeMemBench: Evaluating Memory Systems for Coding Agents on Real Repository Coding Tasks. arXiv:2609.23570. https://arxiv.org/abs/2609.23570
- Main study references (Mem0, LoCoMo, LongMemEval, and related work): see the research proposal.
