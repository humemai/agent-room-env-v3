"""Render the train/test QA-accuracy figure (paper Figure 3).

Reads data/qa-accuracy-fig.json and produces a two-panel (train | test) line
chart with the worst/best range over the 27 TKG policy variants. The y-axis is
scaled to the plotted series; the analytic hidden-state oracle (100) and the
capacity-saturation point are reported in the text rather than drawn, since a
0-100 axis leaves both panels empty above the data.

Future series (e.g., temporally-informed neural agents) can be added without
touching this script:

    python plot-qa-accuracy.py --extra-series "Neural - Transformer (temporal)" extra.json

where extra.json maps {"train": {"<capacity>": {"mean": .., "std": ..}, ...},
"test": {...}}.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Colorblind-validated categorical palette (CVD-checked, fixed assignment)
STYLE = {
    "TKG": {"color": "#2a78d6", "marker": "o"},
    "KG": {"color": "#eb6834", "marker": "s"},
    "Neural - Transformer": {"color": "#1baf7a", "marker": "^"},
    "Neural - LSTM": {"color": "#eda100", "marker": "D"},
}
EXTRA_STYLE = [
    {"color": "#e87ba4", "marker": "v"},
    {"color": "#4a3aa7", "marker": "P"},
]


def load_series(data, env, agent, capacities):
    entries = data["series"][env][agent]
    mean = np.array([entries[str(k)]["mean"] for k in capacities], dtype=float)
    std = np.array([entries[str(k)]["std"] for k in capacities], dtype=float)
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default="data/qa-accuracy-fig.json", help="Input data JSON"
    )
    parser.add_argument(
        "--out", default="figures/agent_train_test_qa_accuracy.pdf", help="Output PDF"
    )
    parser.add_argument(
        "--extra-series",
        nargs=2,
        action="append",
        metavar=("NAME", "JSON"),
        default=[],
        help="Additional agent series to overlay (name + JSON path)",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.data).read_text())
    caps = data["memory_capacities"]
    pos = np.arange(len(caps), dtype=float)

    extras = []
    for name, path in args.extra_series:
        extras.append((name, json.loads(Path(path).read_text())))

    # Shared upper limit across both panels: the largest plotted value (series mean
    # plus its shading, and the TKG variant band), rounded up to the next multiple of 5.
    peaks = []
    for env in ("train", "test"):
        peaks += [max(data["tkg_variant_range"][env][str(k)][1] for k in caps)]
        for agent in STYLE:
            mean, std = load_series(data, env, agent, caps)
            peaks.append(float(np.max(mean + std)))
        for _, extra in extras:
            entries = extra[env]
            peaks += [entries[k]["mean"] + entries[k]["std"] for k in entries]
    ylim_top = 5.0 * np.ceil(max(peaks) / 5.0) + 2.0

    fig, axes = plt.subplots(
        1, 2, figsize=(9.6, 4.3), sharey=True, facecolor="white"
    )

    for ax, env, title in zip(axes, ("train", "test"), ("Train", "Test (held-out)")):
        ax.set_facecolor("white")

        # Worst/best range over the 27 TKG policy variants (dotted boundary
        # lines; a translucent fill is indistinguishable from the std shading)
        rng = data["tkg_variant_range"][env]
        lo = np.array([rng[str(k)][0] for k in caps])
        hi = np.array([rng[str(k)][1] for k in caps])
        for bound in (lo, hi):
            ax.plot(
                pos, bound, color=STYLE["TKG"]["color"],
                linewidth=1.3, linestyle=(0, (1.5, 2.0)), alpha=0.95,
            )

        for agent, style in STYLE.items():
            mean, std = load_series(data, env, agent, caps)
            ax.plot(
                pos, mean,
                color=style["color"], marker=style["marker"],
                markersize=5.5, linewidth=2.0,
            )
            ax.fill_between(
                pos, mean - std, mean + std, color=style["color"], alpha=0.14,
                linewidth=0,
            )

        for idx, (name, extra) in enumerate(extras):
            style = EXTRA_STYLE[idx % len(EXTRA_STYLE)]
            entries = extra[env]
            xs = [caps.index(int(k)) for k in sorted(entries, key=int) if int(k) in caps]
            mean = np.array([entries[str(caps[x])]["mean"] for x in xs])
            std = np.array([entries[str(caps[x])]["std"] for x in xs])
            ax.errorbar(
                np.array(xs, dtype=float), mean, yerr=std,
                color=style["color"], marker=style["marker"],
                markersize=5, linewidth=1.6, capsize=3, linestyle="--",
            )

        ax.set_title(title, fontsize=15)
        ax.set_xlabel("Long-term memory capacity $K$", fontsize=14)
        ax.set_xticks(pos)
        ax.set_xticklabels([str(c) for c in caps], fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        # Scale to the plotted data. The analytic oracle at 100 is no longer drawn,
        # so a 0-100 axis would leave the top half of both panels empty.
        ax.set_ylim(-1, ylim_top)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("QA accuracy (out of 100)", fontsize=14)

    handles = [
        Line2D([0], [0], color=s["color"], marker=s["marker"], markersize=5,
               linewidth=1.8, label=name)
        for name, s in STYLE.items()
    ]
    for idx, (name, _) in enumerate(extras):
        s = EXTRA_STYLE[idx % len(EXTRA_STYLE)]
        handles.append(Line2D([0], [0], color=s["color"], marker=s["marker"],
                              markersize=5, linewidth=1.6, linestyle="--", label=name))
    handles.append(
        Line2D([0], [0], color=STYLE["TKG"]["color"], linewidth=1.3,
               linestyle=(0, (1.5, 2.0)),
               label="TKG best/worst of 27 policy variants")
    )
    # Legend rows grow with the number of series; reserve bottom space
    # proportionally so it never rides up over the x-axis labels.
    nrows = -(-len(handles) // 3)
    fig.legend(
        handles=handles, loc="lower center", ncol=3, fontsize=11.5,
        frameon=False, bbox_to_anchor=(0.5, -0.045 * nrows),
    )
    fig.tight_layout(rect=(0, 0.02 + 0.02 * (nrows - 1), 1, 1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
