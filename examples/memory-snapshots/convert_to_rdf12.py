"""Convert reified memory snapshots to RDF 1.2 annotation syntax.

Reads the released reification Turtle (parseable by rdflib) and emits the
equivalent RDF 1.2 Turtle using named reifiers and annotation blocks
(one reifier per memory occurrence). Emission is plain text generation;
no RDF 1.2 parser is required.
"""

import sys
from pathlib import Path

from rdflib import Graph, RDF, URIRef

HUMEMAI = "https://humem.ai/ontology#"
BASE = "http://roomkg.local/"
XSD = "http://www.w3.org/2001/XMLSchema#"


def qname(term):
    s = str(term)
    if s.startswith(HUMEMAI):
        return "humemai:" + s[len(HUMEMAI):]
    if s.startswith(BASE):
        return f"<{s[len(BASE):]}>"
    return f"<{s}>"


def literal_repr(lit):
    if lit.datatype is not None:
        dt = str(lit.datatype)
        if dt.startswith(XSD):
            return f'"{lit}"^^xsd:{dt[len(XSD):]}'
        return f'"{lit}"^^<{dt}>'
    return f'"{lit}"'


def convert(path: Path) -> str:
    g = Graph()
    g.parse(path, format="turtle", publicID=BASE)

    out = [
        "@prefix humemai: <https://humem.ai/ontology#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
        "# RDF 1.2 annotation serialization (named reifiers), converted from",
        f"# the reification serialization in {path.name}.",
        "",
    ]

    statements = sorted(
        g.subjects(RDF.type, RDF.Statement),
        key=lambda st: int(g.value(st, URIRef(HUMEMAI + "memory_id"))),
    )
    reified = set()
    for st in statements:
        s = g.value(st, RDF.subject)
        p = g.value(st, RDF.predicate)
        o = g.value(st, RDF.object)
        mem_id = int(g.value(st, URIRef(HUMEMAI + "memory_id")))
        reified.add((s, p, o))
        anns = []
        for q, v in sorted(g.predicate_objects(st)):
            if q in (RDF.type, RDF.subject, RDF.predicate, RDF.object):
                continue
            if str(q) == HUMEMAI + "memory_id":
                continue  # the reifier name carries the id
            anns.append(f"       {qname(q)} {literal_repr(v)}")
        block = " ;\n".join(anns)
        out.append(
            f"{qname(s)} {qname(p)} {qname(o)} ~ humemai:m{mem_id}\n"
            f"    {{|\n{block} |}} ."
        )
        out.append("")

    # plain (non-reified) asserted triples, e.g. adjacency without annotations
    plain = [
        t for t in g
        if t[0] not in statements
        and t[1] not in (RDF.type, RDF.subject, RDF.predicate, RDF.object)
        and (t[0], t[1], t[2]) not in reified
        and not str(t[1]).startswith(HUMEMAI)
    ]
    if plain:
        out.append("# asserted triples without their own annotations")
        for s, p, o in sorted(plain):
            out.append(f"{qname(s)} {qname(p)} {qname(o)} .")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        src = Path(arg)
        dst = src.with_name(src.stem + "_rdf12.ttl")
        dst.write_text(convert(src))
        print(f"{src.name} -> {dst.name}")
