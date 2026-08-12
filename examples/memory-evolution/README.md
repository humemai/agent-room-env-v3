# Memory-state evolution (TKG agent)

The TKG agent's long-term memory rendered at every timestep of one `large-02`
episode, `memory_state_000.pdf` through `memory_state_099.pdf` (capacity 512,
QA/explore = MRU, eviction = LFU, seed 0). This is the same episode and the same
configuration as the serialized snapshots in `../memory-snapshots/`.

Figure 5 of the paper shows three of these frames ($t=0$, $t=50$, $t=99$); the
full series is here so the accumulation can be followed step by step rather than
sampled.

Node colours follow the paper: rooms yellow, agent purple, static objects blue,
moving objects green, walls grey. Edge labels carry an occurrence count in
parentheses when several annotated memories share one base triple, which is why
the counts grow while the set of distinct triples does not.
