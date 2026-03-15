# ESWC 2026 Rebuttal (Research Track, Full Paper)

Thank you for the thoughtful reviews. We address the main concerns below, focusing on clarifications and evidence already present in the submission and its anonymized reproducibility repository.

## R1: Competitor baselines / novelty

We intentionally focus on a *controlled* benchmark to isolate the effect of **explicit temporal KG memory** under identical observations and query conditions.

**Why no direct “competitor” comparisons?** Many related works are not directly comparable because they differ in (i) observation modality (text/pixels/latent features vs. RDF triples), (ii) task definition (navigation-only vs. joint navigation + per-step object-location QA), and/or (iii) environment dynamics and evaluation protocol.

Rooms provides *symbolic* local RDF subgraphs (not text) and an explicit KG hidden state; adding cross-domain baselines would conflate memory effects with modality/protocol effects.

In the camera-ready we will make these non-comparabilities explicit in Related Work (what is comparable vs. what is not), e.g., differences in observation modality (KG triples vs. text/pixels), task and action space (per-step QA + navigation vs. navigation-only), and evaluation protocol/dynamics (fixed query schedule and deterministic periodic dynamics vs. other regimes). Our main experimental axis is *memory mechanism* under a fixed, transparent protocol.

**What baselines are already included?** We compare (a) a plain symbolic RDF agent (no temporal qualifiers), (b) an RDF-star temporal-KG agent with qualifier-driven QA/exploration/eviction, and (c) two neural sequence baselines (LSTM and Transformer DQN) that receive the same symbolic observations and learn a joint (answer × move) policy.

The “test split” keeps the same dynamics but permutes question order (see submitted train/test configs’ `question_objects` lists), targeting *strategy* generalization rather than memorization of a fixed QA schedule.

**Conceptual contribution (beyond engineering).** The contribution is not only implementation: (1) a deterministic, configurable environment whose hidden state and observations are explicit KGs; (2) a lightweight, inspectable temporal memory model using statement-level qualifiers (recency, last access, usage); and (3) a systematic capacity/generalization analysis under a fixed, reproducible protocol. We will sharpen this framing in the final version and make explicit that the benchmark is designed to separate “memory mechanism” effects from modality effects.

## R1/R2: Figure readability and formatting

We agree. In the camera-ready, we will (i) increase font sizes/line widths, (ii) reduce clutter in the KG visualizations by focusing on salient subgraphs and moving dense versions to supplementary material, (iii) sharpen captions to state the intended takeaway, and (iv) fix line-break and layout issues throughout.

## R2: Motivation and application context

We will strengthen motivation with concrete application analogues: agents that maintain an evolving **semantic world model** under partial observability (e.g., household assistants tracking object locations over time; digital-twin style state tracking; interactive QA over evolving knowledge). We will also reframe Rooms as a *unit-test style* environment for temporal memory policies, where deterministic replay enables diagnosis of why a memory strategy succeeds/fails.

## R2: Why RDF / RDF-star (vs. tabular structures)

R2 asks why not represent the “grid-based map” with a tabular structure. The key clarification is that the environment is *not grid-based* in its representation: Figure 1(a) is a schematic visualization, while Figure 1(b) shows the KG view (also what the agent observes). Both the hidden state and observations are RDF triples; thus modeling the agent memory as RDF / RDF-star graphs is the natural, minimal choice.

Using a separate table/coordinate representation would add an extra transformation layer (RDF → table and back) and introduce different inductive biases.

RDF enables transparent querying/inspection; RDF-star enables *explicit qualifiers* (time added, last accessed, recall count), which support time/usage-aware selection and eviction without opaque custom structures. The TKG is not a black box: behavior is driven by explicit rules/queries over stored statements and qualifiers.

## R2: Determinism, periodic schedules, and the periodicity discussion

Determinism and periodic wall patterns are deliberate design choices for replayability and controlled comparison. In the submitted code, walls follow fixed periodic patterns and moving objects execute the first feasible action in a fixed preference list, making transitions deterministic.

We will clarify the periodicity discussion: each wall schedule and each object may have its own period/cycle length, but because the transition is deterministic and the state space is finite, the *overall* environment dynamics become periodic after a transient, with a single global period given by the LCM of the component periods.

## R2: Baseline design choices (e.g., random eviction for plain RDF)

R2 asks why the plain RDF baseline uses random eviction instead of LRU. The plain RDF baseline is intentionally *minimal* facts-only memory and does not store RDF-star qualifiers (timestamps/usage). Without such metadata it cannot implement time-aware FIFO/LRU/LFU within the same memory model; random eviction is the simplest unbiased baseline.

We will also clarify memory scarcity: as stated in the paper, the agent observes ~5–6 triples/step, so over 100 steps it sees hundreds of triples; given a capacity budget, one can directly infer how often eviction must occur (512 triples yields rare eviction, smaller capacities require frequent pruning).

Finally, as stated in the paper, we did evaluate multiple TKG pruning heuristics (FIFO/LRU/LFU) but, due to space, reported the best; full combinations are in the anonymized repo (e.g., `qa-accuracy-all.md`).

## R2: Training details and reproducibility

R2 asks whether the neural agents are new and why they are a natural choice. Our intent is not a novel RL algorithm, but representative neural baselines for sequential decision making under partial observability. RL is a natural fit (POMDP; exploration affects future QA), and we use reward only from question answering (no shaped internal rewards).

We chose DQN with a joint (answer × move) action space to avoid “fancier” multi-policy setups and keep comparisons balanced. The anonymized repository includes training settings, parameter counts, and raw QA/coverage values; in the camera-ready we will add a short related-work pointer and additional training detail (without introducing new experimental claims).

We agree RL training involves many design choices (algorithm, discount, steps, etc.); our intent was to keep the baseline straightforward and avoid over-tuning. We will clarify this and report the key settings needed for faithful reproduction.
