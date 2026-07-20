# Serialized memory snapshots (Turtle)

Turtle serializations of the TKG agent's memory at steps 0, 50, and 99 of a
`large-02` episode (capacity 512, QA/explore = MRU, eviction = LFU, seed 0),
produced with `Humemai.save_to_ttl()`.

Each memory is one reified statement (`rdf:Statement` with `rdf:subject`,
`rdf:predicate`, `rdf:object`) carrying a unique `humemai:memory_id` and its
temporal annotations (`humemai:time_added`, `humemai:last_accessed`,
`humemai:num_recalled`; short-term entries carry `humemai:current_time`).
This reification is the concrete serialization of the RDF 1.2-style annotation
model described in the paper: one reified statement corresponds to one annotated
occurrence of a base triple, so multiple annotation sets for the same base triple
can coexist.
