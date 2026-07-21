"""Verify that the RDF 1.2 snapshots carry exactly the same information as
the reification snapshots (structural one-to-one check).

Parses the reified Turtle with rdflib and the *_rdf12.ttl files with a small
independent parser for the emitted subset, then compares:
  1. the set of memory occurrences (base triple, memory id, annotation set),
  2. the set of asserted base triples.
"""

import re
import sys
from pathlib import Path

from rdflib import Graph, RDF, URIRef

HUMEMAI = "https://humem.ai/ontology#"
BASE = "http://roomkg.local/"


def memories_from_reified(path: Path):
    g = Graph()
    g.parse(path, format="turtle", publicID=BASE)
    mems, asserted = set(), set()
    stmts = set(g.subjects(RDF.type, RDF.Statement))
    for st in stmts:
        s = str(g.value(st, RDF.subject)).removeprefix(BASE)
        p = str(g.value(st, RDF.predicate)).removeprefix(BASE)
        o = str(g.value(st, RDF.object)).removeprefix(BASE)
        mem_id = None
        anns = set()
        for q, v in g.predicate_objects(st):
            qs = str(q)
            if q in (RDF.type, RDF.subject, RDF.predicate, RDF.object):
                continue
            if qs == HUMEMAI + "memory_id":
                mem_id = int(v)
            else:
                anns.add((qs.removeprefix(HUMEMAI), str(v)))
        mems.add((s, p, o, mem_id, frozenset(anns)))
    for s, p, o in g:
        if s in stmts or str(p).startswith(str(RDF)) or str(p).startswith(HUMEMAI):
            continue
        asserted.add((str(s).removeprefix(BASE), str(p).removeprefix(BASE),
                      str(o).removeprefix(BASE)))
    return mems, asserted


ANN_RE = re.compile(
    r"<(?P<s>[^>]+)> <(?P<p>[^>]+)> <(?P<o>[^>]+)> ~ humemai:m(?P<id>\d+)\s*"
    r"\{\|(?P<anns>.*?)\|\}\s*\.",
    re.S,
)
PLAIN_RE = re.compile(r"^<(?P<s>[^>]+)> <(?P<p>[^>]+)> <(?P<o>[^>]+)> \.\s*$", re.M)
ANN_ITEM_RE = re.compile(r'humemai:(\w+) "([^"]*)"')


def memories_from_rdf12(path: Path):
    text = path.read_text()
    mems, asserted = set(), set()
    for m in ANN_RE.finditer(text):
        anns = frozenset(ANN_ITEM_RE.findall(m.group("anns")))
        mems.add((m.group("s"), m.group("p"), m.group("o"),
                  int(m.group("id")), anns))
        asserted.add((m.group("s"), m.group("p"), m.group("o")))
    for m in PLAIN_RE.finditer(text):
        asserted.add((m.group("s"), m.group("p"), m.group("o")))
    return mems, asserted


if __name__ == "__main__":
    ok = True
    for arg in sys.argv[1:]:
        reified = Path(arg)
        rdf12 = reified.with_name(reified.stem + "_rdf12.ttl")
        m1, a1 = memories_from_reified(reified)
        m2, a2 = memories_from_rdf12(rdf12)
        same_m, same_a = m1 == m2, a1 == a2
        ok &= same_m and same_a
        print(f"{reified.name}: memories {len(m1)}=={len(m2)} "
              f"{'MATCH' if same_m else 'MISMATCH'}; "
              f"asserted triples {len(a1)}=={len(a2)} "
              f"{'MATCH' if same_a else 'MISMATCH'}")
        if not same_m:
            for x in list(m1 - m2)[:3]: print("  only in reified:", x)
            for x in list(m2 - m1)[:3]: print("  only in rdf12:  ", x)
        if not same_a:
            for x in list(a1 - a2)[:3]: print("  only in reified:", x)
            for x in list(a2 - a1)[:3]: print("  only in rdf12:  ", x)
    sys.exit(0 if ok else 1)
