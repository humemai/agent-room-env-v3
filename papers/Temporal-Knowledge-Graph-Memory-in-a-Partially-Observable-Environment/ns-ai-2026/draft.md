# A Benchmark for Temporal Knowledge-Graph Memory in Partially Observable Environments

Figures are already saved in `/mnt/ssd2/repos/agent-room-env-v3/papers/Temporal-Knowledge-Graph-Memory-in-a-Partially-Observable-Environment/ns-ai-2026/figures/`.

again, feel free to use what we already have in our axiv paper `/mnt/ssd2/repos/agent-room-env-v3/papers/Temporal-Knowledge-Graph-Memory-in-a-Partially-Observable-Environment/arxiv-2026/main.tex`

also, time to time visit `/mnt/ssd2/repos/agent-room-env-v3/papers/Temporal-Knowledge-Graph-Memory-in-a-Partially-Observable-Environment/ns-ai-2026/cfp.md` for the call for papers and make sure the draft is aligned with the CFP requirements.

## Abstract

- Start with the benchmark gap in neurosymbolic AI.
- Introduce the benchmark artifact and what makes it distinctive: graph-shaped hidden state, graph-shaped observations, explicit temporal memory.
- Define the main task at a high level: answering object-location queries while navigating under partial observability.
- Summarize the evaluation protocol: configurable layouts, train/test split, memory-capacity sweep, multiple baseline families.
- Summarize the main benchmark findings without overselling a single method.
- End with availability, reproducibility, and benchmark use by future work.
- Target length: about 140 to 180 words.
- Abstract text budget: 0.20 page.
- Cumulative manuscript budget: 0.80 page including title, authors, affiliations, and keywords.

## Introduction

- Motivate why memory is central in partially observable environments.
- Motivate why neurosymbolic systems need benchmarks where memory structure is explicit and inspectable.
- Explain the gap in current benchmark families: language-heavy or agent-centric KG use, but not benchmark-native temporal KG memory.
- Introduce Room Environment v3 as a benchmark rather than as only an environment or method testbed.
- End with a short list of benchmark-first contributions.
- No figure or table here.
- Page budget: 1.00 page.
- Cumulative manuscript budget: 1.80 pages.

## Related Work

- Benchmark environments for partially observable agents: TextWorld, Jericho.
- KG-based or graph-based agents in interactive settings.
- Neurosymbolic benchmark papers and papers on benchmarking principles.
- Clarify the nearest comparisons and the exact novelty gap this benchmark addresses.
- Keep this section focused on positioning, not a broad survey.
- No figure or table here.
- Page budget: 0.70 page.
- Cumulative manuscript budget: 2.50 pages.

## Benchmark Overview

- Define the benchmark at a conceptual level.
- Explain the hidden state as an RDF knowledge graph.
- Explain observations as RDF graph fragments under local visibility.
- Explain why the benchmark is suitable for studying temporal memory in neurosymbolic agents.
- State the intended benchmark scope and target capabilities.
- Figure 1: reuse the environment overview figure from the arXiv paper, ideally the combined bird's-eye and graph view.
- Caption focus: what the benchmark world looks like and how graph-shaped state differs from graph-shaped observation.
- Page budget: 1.00 page.
- Cumulative manuscript budget: 3.50 pages.

## Benchmark Construction and Statistics

- Describe environment construction and configurable parameters.
- Describe rooms, objects, moving objects, walls, and deterministic dynamics.
- Describe how training and test layouts or question orders are constructed.
- Provide key benchmark statistics: number of rooms, objects, questions, triples, and split properties.
- Include representative examples of hidden state, observation, and query instances.
- Table 1: benchmark summary table with official configuration, rooms, objects, questions, episode length, and split information.
- This is the main artifact-definition section and one of the most important sections in the paper.
- Page budget: 1.60 pages.
- Cumulative manuscript budget: 5.10 pages.

## Tasks, Metrics, and Evaluation Protocol

- Formalize the agent task and question-answer loop.
- Define the evaluation setting: navigation plus QA under partial observability.
- Define metrics such as QA accuracy, room coverage, and triple coverage.
- Describe the memory-capacity sweep and seed protocol.
- Make clear what counts as the official evaluation protocol for this benchmark.
- Table 2: compact official protocol table with train/test setup, seeds, capacities, and metrics.
- Keep the prose procedural and concise.
- Page budget: 1.00 page.
- Cumulative manuscript budget: 6.10 pages.

## Baselines

- Present the symbolic baselines: RDF and RDF-star.
- Present the neural baselines: LSTM and Transformer.
- Explain the shared interface so that comparisons are meaningful.
- Explain the memory representations and main policy differences at a high level.
- Frame them as baseline implementations for characterizing the benchmark.
- No figure by default.
- If space permits, add a very small comparison table; otherwise keep all of this in prose.
- Page budget: 0.90 page.
- Cumulative manuscript budget: 7.00 pages.

## Benchmark Evaluation and Characterization

- Report the main benchmark results across capacities and splits.
- Characterize what the benchmark reveals about symbolic versus neural memory behavior.
- Discuss which benchmark properties are diagnostic: generalization, interpretability, capacity sensitivity, exploration difficulty.
- Use coverage and memory-state analyses to show what the benchmark measures beyond top-line QA accuracy.
- Keep the emphasis on understanding the benchmark, not proving one model family is universally superior.
- Figure 2: reuse the QA accuracy figure across memory capacities.
- Figure 3: reuse the coverage metrics figure.
- Optional Figure 4: reuse the memory-state evolution figure only if space remains after the first full draft.
- If Figure 4 is included, present it as qualitative interpretability evidence rather than as a core result.
- Page budget: 3.40 pages with Figures 2 and 3 as core items; treat Figure 4 as the first cut if space gets tight.
- Cumulative manuscript budget: 10.40 pages.

## Discussion and Limitations

- State clearly what this benchmark is good for and what it is not designed to test.
- Discuss limitations of deterministic dynamics, fixed question format, and environment simplicity.
- Discuss limitations of the current baseline suite without promising more experiments.
- Explain why the benchmark still fills an important evaluation gap despite those limitations.
- Mention natural future extensions such as richer tasks, more stochasticity, or additional baseline families.
- No figure or table here.
- Page budget: 1.00 page.
- Cumulative manuscript budget: 11.40 pages.

## Conclusion

- Restate the benchmark contribution in one compact paragraph.
- Summarize what the baseline study shows about the benchmark’s usefulness.
- Emphasize availability, reproducibility, and relevance to future neurosymbolic memory research.
- No figure or table here.
- Page budget: 0.40 page.
- Cumulative manuscript budget: 11.80 pages.

## Layout and Budget Notes

- Current format assumption: `Afour` with `\documentclass[Afour,sageh,times]{sagej}`.
- Target main-manuscript length: about 12.00 pages excluding references.
- Working assumption for front matter: about 0.80 page total for title, authors, affiliations, abstract, and keywords.
- Leave about 0.20 page of slack in the main body for caption growth, line-wrap changes, or slightly longer transitions.
- Reserve 1.00 page for references.
- The section budgets above therefore sum to 11.80 pages of planned content plus 0.20 page of slack.
- After drafting the next 1--2 sections in `Afour`, remeasure the actual page density and recalibrate only if the real layout diverges materially from this plan.
- Safe plan: keep Figures 1 to 3 and Tables 1 to 2 as core items, and treat Figure 4 as optional.
- Core visual plan:
- Figure 1 in Benchmark Overview.
- Table 1 in Benchmark Construction and Statistics.
- Table 2 in Tasks, Metrics, and Evaluation Protocol.
- Figure 2 and Figure 3 in Benchmark Evaluation and Characterization.
- Optional Figure 4 in Benchmark Evaluation and Characterization.
- If the draft exceeds 12 pages, cut Figure 4 first, then compress Related Work, then compress Baselines.
- The cumulative budgets above are manuscript totals, not section-only totals.
