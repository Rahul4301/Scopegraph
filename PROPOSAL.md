# When Does a Memory Layer Pay for Itself?

## Accuracy and Cost Boundaries of Agent Memory Against Cached Full Context

Rahul Suthar | B.S. Computer Science, California State University, East Bay
suthar.rahul042@gmail.com

*Faculty Mentor: ______________________________ | Undergraduate Research Proposal | October 2026 (revised)*

**Keywords:** LLM agent memory; long context; prompt caching; Mem0; Supermemory; LoCoMo; LongMemEval; MemoryAgentBench; non-inferiority testing; cost model

---

## Abstract

Memory layers such as Mem0 and Supermemory extract and store distilled facts from a conversation history so that a language model need not read the whole history at question time. Frontier models now accept contexts of about one million tokens, and providers bill cached prompt prefixes at a fraction of the normal input price. Whether a memory layer is required therefore has no single answer. The answer depends on the length of the history and on the pattern of queries, namely their number and their spacing relative to the cache lifetime. This study measures that dependence. A no-memory baseline with prompt caching is compared with two extraction-based memory systems (Mem0 and a self-hosted Supermemory pipeline) and a verbatim retrieval control, under a fixed reader, a fixed retrieval budget, and a shared extraction model. Accuracy is tested on LongMemEval-S with pre-registered non-inferiority tests and a margin of 3 percentage points. LoCoMo and MemoryAgentBench serve as descriptive replications. Cost is computed from cached-token counts logged on every call, and the crossover query volume is estimated as a function of history length and query spacing. A stage attribution records whether each failure arises at extraction, at retrieval, or at reading. The harness, the pre-registration, and the item-level records are released so that later memory systems can be compared against the same baseline.

---

## Central claim

> Whether a language model agent requires an external memory layer cannot be settled by a single comparison. The answer depends on the length of the history, on the number of queries asked of it, and on the interval between queries relative to the cache lifetime of the provider. This study measures that dependence for a no-memory baseline with prompt caching, against extraction-based memory systems and a verbatim retrieval control, under a protocol that equates the reader, the retrieval budget, and the extraction model. The results locate the history length below which full context is sufficient and the query volume above which a memory layer is cheaper. They also show at which stage each approach loses information.

Two consequences follow for how results are reported. All accuracy measurements use one reader, so every boundary is a boundary for that reader. A verdict measured at one history length under one price card becomes outdated when either changes, so results are reported as curves and thresholds with confidence intervals. The protocol and the baseline are treated as the lasting contribution, because a new memory method can be evaluated against them without repeating the comparison.

---

## 1. Introduction

A language model agent that serves one user over weeks accumulates a conversation history that grows without limit. Two strategies exist for answering questions about that history. A memory layer ingests the history, extracts salient facts into a store, and retrieves a small subset at question time. The authors of Mem0 report better answer quality than several baselines, together with lower latency and token cost than passing the full conversation [1]. The alternative keeps the history in the prompt. This has become practical as context windows reach about one million tokens and as providers bill cached prompt prefixes at a fraction of the normal input price.

Comparisons between the two strategies usually end in a verdict: either memory layers help, or long context suffices. A verdict of this kind is a point estimate. It holds for one reader model, one price card, one history length, and one assumption about how often the history is queried, and it can change when any of these changes. Earlier cost analyses also assume a fixed cache discount [4], although the effective discount depends on whether successive queries arrive within the cache lifetime. A practitioner whose history fits in context therefore cannot easily tell whether a memory layer helps, hurts, or only adds cost.

This study treats the choice as a boundary to be located. Along the axes of history length and query pattern, a first boundary is expected to separate the region where full context is sufficient from the region where a memory layer lowers cost without loss of accuracy. A second boundary may separate the region where full-context accuracy holds from the region where it degrades. Both boundaries are expected to depend on the reader model. This study measures them for one reader, and the results are worded accordingly. The study is organized around four research questions (Table 1).

| ID | Research question | Status |
| --- | --- | --- |
| RQ1 | When the history fits in context, is full context with caching non-inferior in accuracy to a memory layer? | Confirmatory (H1, H3) |
| RQ2 | At what number of queries per history does a memory layer become cheaper per question, and how does that number depend on query spacing? | Estimand (H2) |
| RQ3 | How do accuracy and cost change as the history grows from about 100K tokens toward the limit of the context window? | Exploratory |
| RQ4 | At which stage does each approach lose information: extraction, retrieval, or reading? | Exploratory |

*Table 1. Research questions and their status. Only RQ1 is tested confirmatorily; the others are reported with intervals and without adjustment.*

The study makes four contributions.

1. A matched-condition protocol in which every arm shares the reader, the answer template, the questions, the retrieval token budget, and (for the memory systems) the extraction model and embedder.
2. A cost model computed from cached-token counts logged on every call, extended to model the interval between queries, which determines whether a cached prefix is still available.
3. A stage attribution that records, for each question, whether the gold fact was stored, retrieved, and used by the reader.
4. Released artifacts: the harness, the pre-registration, immutable item-level records, and the cost ledger.

---

## 2. Key Prior Work

**Paper.** Pollertlam, N. and Kornsuwannawit, W. (2026). *Beyond the Context Window: A Cost-Performance Analysis of Fact-Based Memory vs. Long-Context LLMs for Persistent Agents.* arXiv:2603.04814 [4]. This is a preprint that, as far as the author could tell, has not been peer reviewed.

**Background.** Persistent conversational agents must choose between passing the full history to a long-context model and maintaining a memory system that extracts and retrieves facts. The authors ask which choice is better in accuracy and in cumulative API cost.

**Contributions.** The authors build a Mem0-based memory system (open-source Mem0, GPT-5-nano for fact extraction, text-embedding-3-small embeddings, a pgvector store, top 20 facts retrieved) and compare it with passing the raw history to GPT-5-mini or GPT-OSS-120B. On LoCoMo the memory system scored 57.68% against 92.85% for long-context GPT-5-mini. On LongMemEval it scored 49.00% against 82.40%. They also build a cost model with a 90% discount on cached input tokens from the second turn onward. At a context length of 100 thousand tokens the memory system becomes cheaper after about ten interaction turns, and the break-even point falls as context grows.

**Limitations.** By the authors' own account, the study uses a single memory architecture (flat extracted facts), ingests all data in one static write phase, relies on an analytic cost model with a fixed cache discount instead of measured cache behavior, and uses a GPT-5-mini judge that may favor models of its own family. Its long-context reader is a small model, so the result may not transfer to a current reader with a window of one million tokens.

**Why this paper.** It is the closest prior work to the present question, and its numbers set the bar that this study must explain. The present study keeps the core comparison and changes four elements: the reader, the number of memory systems, the treatment of caching, and the range of history lengths. The cache discount is measured instead of assumed and is allowed to depend on query spacing. The fixed 100K-token regime is replaced by a length axis.

---

## 3. Related Work and Positioning

**Benchmarks.** LoCoMo evaluates very long-term conversational memory with ten long dialogues and question types that include single-hop, multi-hop, temporal, open-domain, and adversarial questions [2]. LongMemEval tests five long-term memory abilities over 500 curated questions. Its S setting embeds each question in its own haystack of about 115 thousand tokens as published, and 97K to 106K tokens as measured here with a proxy tokenizer [3]. MemoryAgentBench converts existing long-context datasets and two newly constructed datasets into incrementally presented multi-turn inputs and evaluates four competencies: accurate retrieval, test-time learning, long-range understanding, and selective forgetting [8]. Its inputs range from about 103 thousand to 1.44 million tokens.

**Memory versus long context.** An independent testbed on LoCoMo found Mem0, plain RAG, and full context in a 77% to 81% accuracy cluster, with RAG matching that cluster at 8.4 times lower total cost of ownership than Mem0 [5]. A controlled ablation reports that verbatim chunks beat LLM-extracted artifacts by 15.9 points on LoCoMo and 22.0 points on LongMemEval-S [6]. A pre-registered study found that raw turns selected by a typed decision model were non-inferior to an LLM-extraction memory at a tight budget, while extraction systems were more accurate at generous budgets [7]. Table 2 compares these studies with the present one.

| Study | Systems compared | Benchmarks | Cost treatment | Design |
| --- | --- | --- | --- | --- |
| Pollertlam and Kornsuwannawit [4] | Mem0-based memory vs. long-context GPT-5-mini | LongMemEval, LoCoMo, PersonaMemv2 | Analytic model with fixed cache discount | Accuracy and cumulative cost |
| Wolff and Bennati [5] | Mem0, Graphiti, cognee, RAG, full context | LoCoMo | Total cost of ownership | Reproducible testbed |
| An [6] | Extracted artifacts vs. verbatim chunks | LoCoMo, LongMemEval-S | Not the focus | Controlled ablation |
| Sharma and Lall [7] | LLM-extraction memory vs. selected raw turns | LoCoMo, LongMemEval | Write cost reported | Pre-registered, non-inferiority |
| This study | No memory with caching, Mem0, Supermemory (self-hosted), plain RAG | LoCoMo, LongMemEval-S, MemoryAgentBench (fitting subset), composite long histories | Measured cached tokens; crossover as a function of history length and query spacing | Pre-registered margin and falsification conditions; stage attribution |

*Table 2. Prior comparisons and this study. Entries for prior work come from the cited papers' abstracts and, for [4], the full text.*

To the author's knowledge, no single earlier study combines the following elements.

1. A no-memory baseline with a current reader whose window is about one million tokens, so that no history requires truncation.
2. Memory systems and a verbatim-retrieval control run through the same reader, retrieval budget, questions, judge, and (for the memory systems) extraction model.
3. Cost computed from cached-token counts logged on every call, with query spacing treated as a variable.
4. Hypotheses stated as non-inferiority with explicit margins and falsification conditions, registered before data collection.
5. A history-length axis and a stage attribution, so that the result is a boundary with a stated mechanism and not a single comparison.

---

## 4. Study Design

### 4.1 Hypotheses and Falsification

The margin and the tests are fixed before data collection. Only H1 and H3 are confirmatory.

| ID | Hypothesis | Primary test | Outcomes |
| --- | --- | --- | --- |
| H1 | On LongMemEval-S, the no-memory arm (A) is non-inferior to Mem0 (B). | Paired non-inferiority on per-question outcomes, 3 percentage point margin, history-clustered bootstrap interval for (A minus B). | Non-inferior if the lower limit exceeds -3 points. Inferiority demonstrated (H1 falsified) if the upper limit is below -3 points. Inconclusive otherwise. |
| H2 | Estimand, not a test: the number of queries per history beyond which Mem0 is cheaper per question than cached full context, as a function of history length and query spacing. | Cost model from measured token counts, with and without caching. Crossover query count with a bootstrap interval over histories. | Not falsifiable. The absence of a crossover in the tested range is a finding that full context is also cheaper there. |
| H3 | On LongMemEval-S, plain RAG (C) is non-inferior to Mem0 (B). | Same test and interval, for (C minus B). | Same three outcomes as H1. |

*Table 3. Hypotheses, tests, and outcomes.*

LoCoMo and MemoryAgentBench are registered as descriptive replications of H1 and H3. They contain too few independent histories for a confirmatory test, and their results can be consistent or inconsistent with the LongMemEval-S result without establishing non-inferiority. Arm D (Supermemory) is analyzed with the same tests as arm B but is labeled exploratory, so that the confirmatory family and its statistical power remain as registered.

The headline reading combines H1 and H2 (Table 4).

| H1 | H2 | Reading |
| --- | --- | --- |
| Non-inferior | No crossover up to 200 queries per history | Full context is sufficient: no accuracy or cost reason for a memory layer in this regime. |
| Non-inferior | Crossover at Q* of at most 200 | A memory layer is a cost optimization above Q* queries per history and gives no accuracy gain. |
| Falsified (memory layer more accurate by more than the margin) | Any | The headline claim fails in this regime; report where and why. |
| Inconclusive | Any | No accuracy claim; report the interval and the power analysis. |

*Table 4. Joint reading of H1 and H2. Here "realistic" means at most 200 queries per history, the largest per-history count in LoCoMo; the tested range is 1 to 1,000.*

### 4.2 Task and Data

The task is long-term conversational question answering. The input is a long multi-session history plus a question about it, for example a fact stated in an early session, a fact that changed later, or a question the history cannot answer. The output is a short free-text answer that is graded against a gold answer. The same task definition applies to all arms.

| Benchmark | Contents | Role | Grading |
| --- | --- | --- | --- |
| LongMemEval-S [3] | 500 questions, each with its own haystack of 97K to 106K tokens (proxy tokenizer), so histories are not shared and caching cannot amortize. 30 questions test abstention. | Confirmatory (H1, H3) | GPT-5 nano judge; substring match as diagnostic |
| LoCoMo [2] | 10 conversations of 10K to 21K tokens, 1,986 questions. Categories 1 to 4 (1,540 questions) are primary; the 446 adversarial questions are reported separately and graded for abstention. | Descriptive replication; the only benchmark where one history receives many questions, which makes it the primary source for measured crossover | GPT-5 nano judge; substring match as diagnostic |
| MemoryAgentBench [8] | 146 long contexts shared by many questions, 3,671 questions in total, pinned revision. 145 contexts fit the reader window; one context of 1.48M tokens is excluded and reported. Accurate retrieval and conflict resolution are the primary descriptive competencies. Five accurate-retrieval contexts repeat LongMemEval-S content and are excluded from the primary set, leaving 25 contexts and 2,500 questions. Test-time learning and long-range understanding are exploratory. | Descriptive replication | Substring match against accepted answers (primary); judge verdict (secondary) |
| Composite long histories (Section 4.4) | Histories constructed from LongMemEval-S sessions at target lengths between 200K tokens and the window limit | Exploratory (RQ3) | GPT-5 nano judge; substring match as diagnostic |

*Table 5. Benchmarks. All three published benchmarks are used as published, with checksums recorded in the repository.*

### 4.3 Arms

All arms share the reader, the answer template, and the questions. Only the way the history reaches the reader differs. Arms B, C, and D place retrieved items in the prompt under one shared token budget of 7,000 tokens, filled with whole items in rank order from a pool of 200 candidates, and no item is cut.

| Arm | Name | Definition | Role |
| --- | --- | --- | --- |
| A | No memory | Full history first, question last. The prefix is byte-identical across questions about one history. An explicit cache breakpoint follows the history when a history receives at least two questions (LoCoMo). For a single question (LongMemEval-S) no breakpoint is used, because a cache write without a later read is a surcharge. Both modes are priced in the cost model. | Baseline |
| B | Mem0 | Open-source Mem0, version pinned (2.2.1), one user identifier per history, local vector store, default library settings (threshold 0.1, no reranking), ten turns per ingestion call, session date written into each message. The default extraction model and embedder are recorded. If the extraction model is withdrawn, a pre-registered fallback applies and arm B is rerun in full for any benchmark that used the default. | System under test |
| C | Plain RAG | Verbatim chunks of whole turns (256 tokens), the same embedder as arm B, cosine ranking. No LLM extraction. | Control for H3 |
| D | Supermemory (self-hosted) | The open-source Supermemory server [9], one isolated space per history, configured with the same extraction model and the same embedder as arm B. | Second memory system (exploratory) |

*Table 6. Experimental arms.*

Arm D uses the self-hosted server because it removes the per-token ingestion fee of the hosted platform and allows the extraction model to be held equal to arm B. The documentation states that the hosted platform uses proprietary extraction models and that self-hosted extraction runs on a model supplied by the user [9]. Arm D is therefore reported as "Supermemory (self-hosted, shared extraction model)". Its results do not describe the hosted product. A small hosted run on LoCoMo may be added as a check on divergence if credits are available.

For both memory systems the study records what each system extracts by default. Mem0's default extraction reads only user messages. For LoCoMo, where the two speakers are peers, both are sent as user messages prefixed with their names. For LongMemEval the real user and assistant roles are kept, so facts stated only by the assistant are not extracted, which is Mem0 as released. A sensitivity analysis excludes the single-session-assistant category of LongMemEval-S and reports whether the conclusion changes.

### 4.4 History-Length Axis

LongMemEval-S histories of about 100K tokens are far below the window limit. In that range full context is expected to perform well, and a comparison confined to it cannot locate a boundary. The length axis therefore uses composite histories.

A composite history concatenates the evidence sessions of a set of LongMemEval-S questions with distractor sessions drawn from the haystacks of other questions. It is built to a series of target lengths between 200K tokens and the limit of the reader window, minus a fixed margin for the question and the output. Every question whose evidence sessions lie in a composite is asked of that history. This design has two practical effects. Ingestion cost is shared among many questions, and caching becomes applicable to arm A. It also introduces a risk: evidence sessions from different LongMemEval-S instances may address the same topic, so a distractor can contradict the gold answer of a question. A conflict screen is applied to each composite, and its procedure is fixed before the run. Questions are drawn as a fixed random subset with a seed registered in advance, and the subset size is set from pilot variance and the budget cap.

The reader price card includes a surcharge for inputs above 272K tokens, and the cost model applies it. Accuracy is reported as a function of history length for every arm, together with the position of the evidence within the history, which LongMemEval provides through evidence session identifiers.

### 4.5 Lifecycle: As-Of Queries and Write Granularity

The standard protocol ingests a whole history once and queries it at the end. Deployed agents instead accumulate history and are queried throughout. Two additions on LoCoMo address this difference, since LoCoMo is the cheapest benchmark and the one that gives many questions to one history.

**As-of querying.** Each question is asked after the session that contains its last evidence turn, and again at the end of the conversation. Memory arms ingest sessions incrementally, so the store at each checkpoint contains only past sessions. Arm A receives the history up to the checkpoint, with questions ordered by checkpoint so that cached prefixes grow. Cache behavior under growing prefixes is measured, not assumed. Accuracy and cumulative cost are reported as the history grows.

**Write granularity.** For arm B, ingestion is run per turn, per ten turns (the default), and per session. The comparison with arm A is reported for each setting, because the choice of granularity changes both extraction quality and ingestion cost.

### 4.6 Reader and Judge

The primary reader is GPT-6 Luna (window of about 1.05M tokens; GPT-5.6 Luna is the fallback) with reasoning effort `none`, temperature 0, a maximum output of 512 tokens, and one answer template. The judge and sole grader is GPT-5 nano with minimal reasoning effort, one fixed prompt, and a capped output. No second grader and no human grading are used. The served model snapshot is logged for every call because the provider lists only the alias.

No second reader is used. The reader is a strong model that is weaker than the largest current frontier models, so full-context accuracy is expected to degrade at shorter histories than it would for a larger reader, and any accuracy boundary found here is likely to lie at or below the boundary for a larger model. This direction is an expectation and is not tested. The cost model is a function of the price card, so cost results can be recomputed for other readers or providers. Accuracy results cannot. Because the judge and the reader are both OpenAI models, an advantage for same-family answers is possible. The deterministic substring metric is therefore reported next to the judge accuracy.

Judge checks are run in the pilot, before any full run, with thresholds fixed in advance.

| Check | Threshold | Consequence if violated |
| --- | --- | --- |
| Unparseable judge output | At most 1% of judged answers | The study stops and the owner decides, because those answers would be dropped silently. |
| Flip rate on 100 reruns (50 from arm A, 50 from arm B) | At most 5% | Every answer is graded by the majority of three judge runs, and the flip rate is reported with every accuracy figure. |
| Agreement with a deterministic containment test on short-answer, non-abstention questions | At least 85% | Containment accuracy is reported beside every judge accuracy, and a conclusion that differs between the two is labeled judge-dependent. |

*Table 7. Pre-registered judge checks.*

### 4.7 Cost Model

Cost per question is computed from token counts and cache fields logged on every call, not from an assumed discount. Let H be the history length in tokens, q the question length, o the output length, Q the number of queries per history, and g the interval between successive queries. Let p_in, p_w, p_r, and p_out be the input, cache-write, cache-read, and output prices, and T the cache lifetime. The reader price card is read from the repository configuration with its source and date.

For arm A, a query that arrives within T of the previous query finds the prefix warm, because the lifetime is refreshed on reuse. Let f denote the fraction of queries that find the prefix warm; it is measured from logged cached-token counts and simulated for synthetic arrival schedules. Then

C_A(Q) = H * p_w + (Q - 1) * [ f * H * p_r + (1 - f) * H * p_w ] + Q * (q * p_in + o * p_out)

where, for a history that receives a single question, the first term is replaced by H * p_in because no breakpoint is used.

For a memory arm, with I(H) the one-time ingestion cost (extraction tokens and embeddings), B the retrieval budget, and r the retrieval cost per query,

C_M(Q) = I(H) + Q * (r + (B + q) * p_in + o * p_out)

The crossover Q* is the smallest Q for which C_M(Q) is below C_A(Q). It is estimated with a bootstrap interval over histories, once with caching and once without, as a function of H and g. Inter-arrival schedules cover bursts (g well below T) and sparse traffic (g above T). Latency is reported separately, with ingestion time listed apart from per-query time.

### 4.8 Stage Attribution

For each question answered by a memory arm, three outcomes are recorded: whether the gold fact appears in the store (extraction), whether it appears in the retrieved context (retrieval), and whether the reader answers correctly given that it does (reading). Gold-in-context is already logged. Gold-in-store is added as a normalized substring test over stored memories. Because extracted facts paraphrase the source, this test underestimates storage for the extraction systems, and a judge-based check on a fixed random sample of 200 questions is used to estimate the size of that bias. For arm A, retrieval and storage are absent by construction, and the analysis reports reader accuracy as a function of evidence position and history length.

### 4.9 Statistical Analysis

H1 and H3 are tested on LongMemEval-S, where each of the 500 histories holds one question, so histories are independent and the question-level and history-clustered intervals coincide. Intervals are two-sided and use 10,000 bootstrap resamples of histories with seed 0, on paired per-question differences. The history-clustered interval is the primary interval for every benchmark. Holm's procedure adjusts across the two confirmatory tests, and the adjusted interval level is stated for each step in the registration.

The margin of 3 points is the largest accuracy loss accepted in exchange for dropping a memory layer. It was chosen before any data and is not derived from data. Whether each conclusion also holds at margins of 2 and 5 points is reported as exploratory.

With 500 paired questions, the power of a non-inferiority test depends on the discordance rate d, the share of questions on which exactly one of two arms is correct. When the true difference is zero, the lower confidence limit is approximately -z * sqrt(d / 500). To remain above -3 points, d must be at most about 12% with an unadjusted 95% interval and at most about 9% after the Holm adjustment. Larger true advantages for the baseline relax this requirement. The power curve over a plausible range of d is computed before the full run and reported. A non-significant result in an underpowered test is labeled inconclusive and is not presented as evidence against non-inferiority. A result that clears the margin stands in either case.

LoCoMo and MemoryAgentBench have few independent histories (10 and 25), and a percentile bootstrap undercovers with so few clusters. Their intervals are reported, and no non-inferiority claim rests on them. Conclusions are worded as consistent or inconsistent with the LongMemEval-S result. All other comparisons, which include categories, length-axis results, as-of results, arm D, and sensitivity margins, are exploratory and reported descriptively without adjustment.

### 4.10 Components and Status

| Component | Content | Confirmatory | Status |
| --- | --- | --- | --- |
| Core comparison | Arms A, B, C on LongMemEval-S, LoCoMo, MemoryAgentBench; H1 to H3 | H1, H3 | Harness built and tested offline; no benchmark run yet |
| Cost model | Measured cache tokens; crossover with and without caching | No | Built; extension to query spacing is analysis on logged tokens |
| Stage attribution | Gold-in-store logging, evidence-position analysis | No | Gold-in-context built; gold-in-store to add |
| Length axis | Composite histories (Section 4.4) | No | To build |
| Lifecycle | As-of queries and write granularity (Section 4.5) | No | To build |
| Arm D | Supermemory self-hosted (Section 4.3) | No | To build; earlier code recoverable from repository history |

*Table 8. Components. Everything outside the core comparison is exploratory unless it is added to the registration before the registration tag is created.*

---

## 5. Expected Results and Discussion

Prior work suggests, but does not guarantee, that full context performs at least as well as a memory system on conversational benchmarks at lengths near 100K tokens [4][5], and that verbatim retrieval comes close to extraction [5][6]. The study is designed so that each expectation can fail. Table 9 lists the possible patterns along the length axis.

| Pattern | Observation | Interpretation |
| --- | --- | --- |
| 1. Full context sufficient across the tested lengths | Arm A is non-inferior at every length, and no crossover appears within 200 queries per history. | In the tested range a memory layer adds cost and complexity without benefit. |
| 2. Cost boundary only | Arm A matches the memory arms on accuracy, and the memory arms become cheaper beyond a measured Q*, which shifts with query spacing. | A memory layer is a cost optimization with a quantified threshold that depends on traffic pattern. |
| 3. Accuracy boundary | Arm A degrades beyond a history length L*, while the memory arms or plain RAG do not. | A memory layer is justified by accuracy above L*, and the stage attribution shows whether the degradation arises in reading. |
| 4. Extraction adds little | Plain RAG is within the margin of both memory systems at lower ingestion cost. | LLM extraction adds no value over verbatim retrieval in this regime. |

*Table 9. Possible patterns and their interpretation.*

On MemoryAgentBench, selective forgetting is the competency where a memory layer with update operations is most likely to outperform full context. It is therefore the most informative descriptive test of the headline claim. If full context with caching is non-inferior whenever the history fits, many deployments may pay for a memory layer they do not need. If a crossover exists, the study supplies a concrete query volume, as a function of query spacing, above which a layer pays for itself.

The stage attribution is intended to be usable by developers of memory systems. If failures concentrate at extraction, work on extraction prompts is indicated. If they concentrate at retrieval, ranking and budget allocation are the targets. If they concentrate at reading, the evidence is present and the reader is the limiting component.

---

## 6. Limitations and Threats to Validity

- **Reader and seed.** The design uses one reader and one seed. Accuracy boundaries apply to that reader, and cost boundaries can be recomputed for other price cards. Whether the accuracy boundary moves for a larger or a smaller reader is not tested.
- **Reasoning effort.** The reader runs with reasoning effort `none`. This choice may lower accuracy on multi-hop and temporal questions for every arm and may compress or distort differences between arms. A small run with reasoning enabled is a useful check.
- **Grader.** A single LLM grader whose error and family-level bias are not independently measured. The reader and the judge are both OpenAI models in the core design. The registered thresholds in Table 7 bound but do not remove this risk.
- **Independent histories.** LoCoMo has ten histories, so history-level intervals are wide. LongMemEval-S supports confirmatory tests but is underpowered for equivalence when the discordance rate is high (Section 4.9).
- **Composite histories.** Composite histories are synthetic. They may contain contradictions between evidence sessions, and they differ in structure from the organic histories of deployed agents.
- **Systems.** Mem0 is the open-source library and not the hosted platform, so results are not comparable to published Mem0 numbers. Supermemory runs self-hosted with a shared extraction model and does not represent the hosted product. Results are tied to the tested versions.
- **Mem0 configuration.** Mem0 runs with library defaults. A tuned configuration might perform better, and the claim is limited to the default.
- **Contamination.** LoCoMo and LongMemEval are public and may appear in the training data of the reader.
- **Scope.** The benchmarks are conversational. No coding or repository workloads are included. MemoryAgentBench runs with one generic reader template and not with its own task prompts. Fit checks use a proxy tokenizer, and the usage field returned by the API is authoritative for cost.
- **Prices.** Price cards change. Every price is recorded with its source and date, and the cost model is a function of the price card, so conclusions can be recomputed.

---

## 7. Ethical Considerations and Risks

**Privacy.** Memory layers retain information about users, and sending histories to hosted APIs (the reader and the judge) exposes that content to third parties. This study uses only public benchmark conversations and no real user data. Any conclusion about deployment notes that a memory layer adds a persistent store of personal information that full-context prompting does not.

**Misleading comparisons.** Commercial memory systems change quickly, so a result tied to one version, one reader, and one judge could be over-generalized, either to dismiss memory layers in general or to discredit a vendor unfairly. Versions and configurations are pinned and reported. Conclusions are worded as applying to the tested regime under the stated settings, and the self-hosted status of arm D is stated wherever its results appear.

---

## 8. Timeline

| Week | Dates (2026) | Work |
| --- | --- | --- |
| 1 to 2 | Oct 5 to Oct 16 | Finish arms and the logging and grading pipeline; add gold-in-store logging; build the composite-history constructor and the reader abstraction; decide which extensions enter the registration; create the registration tag. |
| 3 | Oct 19 to Oct 23 | Pilot (30 LoCoMo questions from 3 conversations, 20 LongMemEval-S questions); judge checks; power curve; cost estimates for each extension; go or no-go decision per extension. |
| 4 to 5 | Oct 26 to Nov 6 | Full LoCoMo, all registered arms; as-of queries and write granularity. |
| 6 to 7 | Nov 9 to Nov 20 | Full LongMemEval-S, all registered arms. |
| 8 | Nov 23 to Nov 27 | Length-axis runs. MemoryAgentBench on the primary descriptive competencies only if schedule and budget allow; it is dropped before the length axis. |
| 9 | Nov 30 to Dec 4 | Analysis: tests, intervals, crossover surfaces, stage attribution. |
| 10 | Dec 7 to Dec 11 | Write-up and buffer. |

*Table 10. Ten-week timeline, fitted to a December 2026 graduation. The projected reader and ingestion cost of the core design is about $35 under current price cards, against a hard cap of $350; extensions are costed from the pilot. Each paid stage requires the owner's approval, and the command-line tool refuses a paid command until the registration tag exists and the matching approval gate is set.*

---

## 9. Released Artifacts

The harness and configuration files, the registration with its tag, the analysis scripts, the cost model as a function that accepts a price card, and the item-level records are released. Raw records are created exclusively and never overwritten, and each record carries the configuration hash, the served model snapshot, and the cache fields returned by the API. A new memory system can be added as an arm, run through the same reader and budget, and compared against the stored baseline without repeating the baseline runs.

---

## References

[1] Chhikara, P., Khant, D., Aryan, S., Singh, T., and Yadav, D. (2025). Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. arXiv:2504.19413. https://arxiv.org/abs/2504.19413

[2] Maharana, A., Lee, D.-H., Tulyakov, S., Bansal, M., Barbieri, F., and Fang, Y. (2024). Evaluating Very Long-Term Conversational Memory of LLM Agents (LoCoMo). Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics. arXiv:2402.17753. https://arxiv.org/abs/2402.17753

[3] Wu, D., Wang, H., Yu, W., Zhang, Y., Chang, K.-W., and Yu, D. (2025). LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. International Conference on Learning Representations (ICLR) 2025. arXiv:2410.10813. https://arxiv.org/abs/2410.10813

[4] Pollertlam, N., and Kornsuwannawit, W. (2026). Beyond the Context Window: A Cost-Performance Analysis of Fact-Based Memory vs. Long-Context LLMs for Persistent Agents. arXiv preprint arXiv:2603.04814. https://arxiv.org/abs/2603.04814

[5] Wolff, B., and Bennati, J. (2026). Cost and Accuracy of Long-Term Memory in Distributed Multi-Agent Systems Based on Large Language Models. IEEE COMPSAC 2026. arXiv:2601.07978. https://arxiv.org/abs/2601.07978

[6] An, T. (2026). Fidelity Before Structure: Verbatim Chunks Beat Lossy Artifact Extraction in Long-Conversation LLM Memory. arXiv preprint arXiv:2601.00821. https://arxiv.org/abs/2601.00821

[7] Sharma, R., and Lall, R. (2026). When Does Selection Replace Extraction? A Pre-Registered Test of Agent Memory with a Typed Decision Model. arXiv preprint arXiv:2609.34227. https://arxiv.org/abs/2609.34227

[8] Hu, Y., Wang, Y., and McAuley, J. (2026). Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions. International Conference on Learning Representations (ICLR) 2026. arXiv:2507.05257. https://arxiv.org/abs/2507.05257

[9] Supermemory. Self-hosting overview (Supermemory local). Documentation, accessed October 2026. https://supermemory.ai/docs/self-hosting/overview
