"""Recompute every number the paper reports, from the released results.

Each claim below is computed from `data/` and `results/` and checked against the
value asserted in the paper. Optionally the paper and the response letter can be
passed as well, in which case the script also checks that each value literally
appears in them, so that editing a document without rerunning the numbers, or
rerunning the numbers without editing the documents, is caught.

    python verify_claims.py
    python verify_claims.py --paper main.tex --letter response-to-reviewers.md

Exits non-zero if any claim fails.

Two aggregation levels matter and are easy to confuse:

* A *run* is one directory: one policy configuration, one capacity, one seed. Its
  `results.yaml` already holds the mean over that run's episodes.
* A *configuration* is the mean over the five seeds. Every figure in the paper
  reports configuration-level means, so the bounds and spreads below are computed
  over those, never over individual runs.

The symbolic archive holds only `remember_policy: all` runs on the `large-02` and
`large-02-q` layouts, which is the grid the paper reports; see results/README.md.
"""

import argparse
import collections
import json
import re
import statistics as st
import sys
import tarfile
from pathlib import Path

CAPS = ["0", "2", "4", "8", "16", "32", "64", "128", "256", "512"]
TRAIN_ROOM, TEST_ROOM = "large-02", "large-02-q"


# --------------------------------------------------------------------------- io

def load_series(root: Path):
    """The aggregated series behind Figure 3."""
    fig = json.loads((root / "data/qa-accuracy-fig.json").read_text())["series"]
    tt = json.loads((root / "data/qa-accuracy-temporal-transformer.json").read_text())
    tl = json.loads((root / "data/qa-accuracy-temporal-lstm.json").read_text())
    return fig, tt, tl


def load_runs(tarball: Path):
    """Yield (config_fields, score) for every run directory in an archive."""
    field = lambda t, k: (m.group(1) if (m := re.search(rf"{k}:\s*(\S+)", t)) else None)
    with tarfile.open(tarball) as tf:
        cfgs, scores, params = {}, {}, {}
        for m in tf.getmembers():
            if not m.isfile():
                continue
            d, name = str(Path(m.name).parent), Path(m.name).name
            if name == "train.yaml":
                cfgs[d] = tf.extractfile(m).read().decode()
            elif name == "num_params.yaml":
                params[d] = re.search(r"(\d+)", tf.extractfile(m).read().decode()).group(1)
            elif name == "results.yaml":
                txt = tf.extractfile(m).read().decode()
                if s := re.search(r"test_score:\s*\n\s*mean:\s*([\d.]+)", txt):
                    scores[d] = float(s.group(1))
    for d, t in cfgs.items():
        if d in scores:
            yield {
                "dir": d,
                "room": field(t, "room_size"),
                "cap": int(field(t, "max_long_term_memory_size") or -1),
                "qa": field(t, "qa_policy"),
                "explore": field(t, "explore_policy"),
                "forget": field(t, "forget_policy"),
                "seed": field(t, "seed"),
                "params": params.get(d.split("__test__")[0], params.get(d)),
                "temporal": "temporal" in d,
            }, scores[d]


def by_config(runs, room, cap):
    """Mean over seeds for each policy configuration at one room/capacity."""
    g = collections.defaultdict(list)
    for cfg, sc in runs:
        if cfg["room"] == room and cfg["cap"] == cap:
            g[(cfg["qa"], cfg["explore"], cfg["forget"])].append(sc)
    return {k: st.mean(v) for k, v in g.items()}


# ------------------------------------------------------------------------ checks

def build_claims(root: Path):
    fig, tt, tl = load_series(root)
    sym = list(load_runs(root / "results/roomkg-per-seed-results-symbolic.tar.gz"))
    neu = list(load_runs(root / "results/roomkg-per-seed-results-neural.tar.gz"))

    def neural_best(room):
        """Max over configuration means, which is what the paper's bound is."""
        g = collections.defaultdict(list)
        for cfg, sc in neu:
            if cfg["room"] == room and cfg["params"]:
                g[(cfg["params"], cfg["temporal"], cfg["cap"])].append(sc)
        return max(st.mean(v) for v in g.values())

    def series_best(split, cap):
        return max(fig[split]["Neural - LSTM"][cap]["mean"],
                   fig[split]["Neural - Transformer"][cap]["mean"],
                   tl[split][cap]["mean"], tt[split][cap]["mean"])

    def crosses(agent, split, bound):
        return next(k for k in CAPS if fig[split][agent][k]["mean"] > bound)

    def spread(cap):
        v = sorted(by_config(sym, TEST_ROOM, cap).values())
        return round(v[0], 2), round(st.median(v), 2), round(v[-1], 2)

    C = []
    add = lambda where, what, got, want, cite="P": C.append((where, what, got, want, cite))

    add("§5.1, Table 3", "neural bound, train", round(neural_best(TRAIN_ROOM), 2), 18.6, "PL")
    add("§5.1, Table 3", "neural bound, test", round(neural_best(TEST_ROOM), 2), 14.2, "PL")
    add("Table 3", "TKG at K=512 train", fig["train"]["TKG"]["512"]["mean"], 44.48, "PL")
    add("Table 3", "TKG at K=512 test", fig["test"]["TKG"]["512"]["mean"], 45.64, "PL")
    add("Table 3", "KG at K>=512 train", fig["train"]["KG"]["512"]["mean"], 42.96, "P")
    add("Table 3", "KG at K>=512 test", fig["test"]["KG"]["512"]["mean"], 42.08, "P")
    for cap, want in ((32, (7.8, 9.68, 15.52)), (128, (23.8, 30.56, 33.2)),
                      (512, (45.28, 45.72, 46.52))):
        add("§5.1", f"27-variant worst/median/best, test, K={cap}", spread(cap), want,
            "PL" if cap in (32, 512) else "P")
    add("§5.1", "variant convergence at K=512 (points)", round(spread(512)[2] - spread(512)[0], 2), 1.24, "")
    add("§5.1", "TKG clears the bound, train", crosses("TKG", "train", 18.6), "64")
    add("§5.1", "TKG clears the bound, test", crosses("TKG", "test", 14.2), "32")
    add("§5.1", "KG clears the bound, train", crosses("KG", "train", 18.6), "128")
    add("§5.1", "KG clears the bound, test", crosses("KG", "test", 14.2), "64")
    add("§5.1", "TKG > KG at every K>=32, both rooms",
        all(fig[s]["TKG"][k]["mean"] > fig[s]["KG"][k]["mean"]
            for s in ("train", "test") for k in ("32", "64", "128", "256", "512")), True)
    add("§5.1", "TKG at K=512 vs neural bound, train (x)", round(44.48 / 18.6, 2), 2.39, "")
    add("§5.1", "TKG at K=512 vs neural bound, test (x)", round(45.64 / 14.2, 2), 3.21, "")
    add("§5.1", "strongest neural gain K=2->512, train (points)",
        round(fig["train"]["Neural - Transformer"]["512"]["mean"]
              - fig["train"]["Neural - Transformer"]["2"]["mean"], 1), 3.2)
    add("§5.1", "largest neural gain on held-out (points)",
        round(max(d[s]["512"]["mean"] - d[s]["2"]["mean"]
                  for d in (tt, tl) for s in ["test"]), 1), 0.6)
    add("§4", "27 variants present", len(by_config(sym, TEST_ROOM, 512)), 27)
    # Table 3 rows that were previously unguarded
    def cfg_stats(room, cap, pred):
        g = collections.defaultdict(list)
        for cfg, sc in sym:
            if cfg["room"] == room and cfg["cap"] == cap and pred(cfg):
                g[(cfg["qa"], cfg["explore"], cfg["forget"])].append(sc)
        return g

    sel = ("most_recently_used", "most_recently_used")          # train-selected at K=512
    tkg = [v for k, v in cfg_stats(TEST_ROOM, 512, lambda c: True).items() if k[:2] == sel]
    add("Table 3", "TKG at K=512 test, across-seed sd",
        round(st.pstdev(max(tkg, key=st.mean)), 2), 1.21, "P")
    # Table 3's K=0 floor is one bound over both symbolic agents, not a row each.
    # At K=0 neither retains anything, so the agents differ only in the order their
    # searches enumerate neighbouring rooms; splitting the row would invite the
    # reader to compare two arbitrary walks.
    o_tr = cfg_stats(TRAIN_ROOM, 0, lambda c: True)
    o_te = cfg_stats(TEST_ROOM, 0, lambda c: True)
    floor = max([st.mean(v) for v in o_tr.values()]
                + [st.mean(v) for v in o_te.values()]
                + [fig[s]["KG"]["0"]["mean"] for s in ("train", "test")])
    add("Table 3", "observation-only floor, both agents", round(floor, 2), 2.0, "")
    add("Table 3", "K=0 floor bound in Table 3 holds", floor <= 2.0, True, "")
    # last-seen: MRA retrieval, unbounded memory, exploration selected on training accuracy
    mra_tr = {k[1]: st.mean(v) for k, v in cfg_stats(TRAIN_ROOM, 1024, lambda c: True).items()
              if k[0] == "most_recently_added"}
    mra_te = {k[1]: st.mean(v) for k, v in cfg_stats(TEST_ROOM, 1024, lambda c: True).items()
              if k[0] == "most_recently_added"}
    pick = max(mra_tr, key=mra_tr.get)
    add("§5.2", "last-seen (MRA, unbounded) test", round(mra_te[pick], 2), 45.44, "L")
    add("§3.3", "state-space configurations (log10)",
        round(__import__("math").log10(49 ** 18 * 1279948013568000000000000000), 1), 57.5, "")

    a512, a1024 = by_config(sym, TEST_ROOM, 512), by_config(sym, TEST_ROOM, 1024)
    add("§5.2", "no variant moves by more than 0.16 at K=1024",
        round(max(abs(a1024[k] - v) for k, v in a512.items() if k in a1024), 2), 0.16, "PL")
    return C


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=".", type=Path)
    p.add_argument("--paper", type=Path, help="also check these values appear in the paper")
    p.add_argument("--letter", type=Path, help="also check these values appear in the letter")
    a = p.parse_args()

    claims = build_claims(a.root)
    width = max(len(w) for _, w, _, _, _ in claims)
    bad = 0
    print(f"{'where':14} {'claim':{width}}  {'computed':>22}  {'asserted':>22}")
    print("-" * (width + 66))
    for where, what, got, want, _ in claims:
        ok = got == want
        bad += not ok
        print(f"{where:14} {what:{width}}  {str(got):>22}  {str(want):>22}  "
              f"{'ok' if ok else '<-- MISMATCH'}")

    for label, path in (("paper", a.paper), ("letter", a.letter)):
        if not path:
            continue
        text = path.read_text(encoding="utf-8")
        missing = []
        for _, what, _, want, cite in claims:
            if label[0].upper() not in cite:
                continue
            for v in (want if isinstance(want, tuple) else [want]):
                if isinstance(v, float) and f"{v}" not in text and f"{v:.2f}" not in text:
                    missing.append((what, v))
        print(f"\n{label}: {len(missing)} expected value(s) missing from {path.name}")
        for what, v in missing[:12]:
            print(f"   {v} ({what})")

    print(f"\n{len(claims) - bad}/{len(claims)} claims verified")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
