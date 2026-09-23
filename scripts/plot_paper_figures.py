#!/usr/bin/env python3
"""Paper figures from logged EXPs: sensitivity heatmap + shortlist table/bars."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SENS = ROOT / "results/20260923-phase2-sens-w4-full-fr1desk.json"

SHORTLIST = [
    # label, mean ATE, vs FP16 pct, note
    ("FP16", 0.0295, 0.0, "baseline"),
    ("FP8 e4m3", 0.0294, -0.4, "accuracy champ"),
    ("W8A8", 0.0315, 6.8, "uniform"),
    ("W4 geom-protect K=7", 0.0316, 7.1, "method wrinkle"),
    ("W4 mag-protect K=7", 0.0328, 10.9, "H3 proxy"),
    ("W4 uniform", 0.0333, 12.8, "uniform"),
]


def load_deltas():
    rec = json.loads(SENS.read_text())
    return {k: float(v) for k, v in rec["metrics"]["ate_rel_delta_per_unit"].items()}


def plot_sensitivity_heatmap(deltas: dict[str, float], out: Path):
    # group into grids
    def block_grid(prefix: str, n: int, cols: int):
        vals = np.full(n, np.nan)
        for i in range(n):
            key = f"{prefix}.{i}"
            if key in deltas:
                vals[i] = deltas[key] * 100.0  # percent
        rows = int(np.ceil(n / cols))
        grid = np.full((rows, cols), np.nan)
        for i, v in enumerate(vals):
            grid[i // cols, i % cols] = v
        return grid

    enc = block_grid("enc_blocks", 24, 8)
    dec = block_grid("dec_blocks", 12, 6)
    dec2 = block_grid("dec_blocks2", 12, 6)
    heads = np.array(
        [
            [
                deltas.get("downstream_head1", np.nan) * 100,
                deltas.get("downstream_head2", np.nan) * 100,
                deltas.get("decoder_embed", np.nan) * 100,
            ]
        ]
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2), constrained_layout=True)
    vmax = max(15.0, np.nanmax(list(deltas.values())) * 100)
    vmin = min(-5.0, np.nanmin(list(deltas.values())) * 100)
    cmap = "RdYlGn_r"

    panels = [
        (axes[0, 0], enc, "Encoder blocks (leave-one W4, ΔATE %)", 8),
        (axes[0, 1], dec, "Decoder blocks", 6),
        (axes[1, 0], dec2, "Decoder2 blocks", 6),
        (axes[1, 1], heads, "Heads / decoder_embed", 3),
    ]
    im = None
    for ax, grid, title, cols in panels:
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(cols))
        if grid.shape[0] == 1:
            ax.set_yticks([0])
            ax.set_yticklabels([""])
            labels = ["head1", "head2", "dec_embed"]
            ax.set_xticklabels(labels, rotation=20, ha="right")
        else:
            ax.set_yticks(range(grid.shape[0]))
            ax.set_yticklabels([f"r{r}" for r in range(grid.shape[0])])
            ax.set_xticklabels([str(i) for i in range(cols)])
        # annotate
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                v = grid[r, c]
                if np.isnan(v):
                    continue
                ax.text(
                    c,
                    r,
                    f"{v:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if abs(v) < 0.55 * vmax else "white",
                )

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="ΔATE vs FP16 (%)")
    fig.suptitle(
        "§6.3 Desk sensitivity — fake INT4 WO on one unit (rest FP16)",
        fontsize=12,
    )
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out} and {out.with_suffix('.pdf')}")


def plot_shortlist_bars(out: Path):
    labels = [r[0] for r in SHORTLIST]
    means = [r[1] for r in SHORTLIST]
    deltas = [r[2] for r in SHORTLIST]
    colors = []
    for lab, _, d, _ in SHORTLIST:
        if "FP16" in lab:
            colors.append("#9e9ac8")
        elif "FP8" in lab:
            colors.append("#2ca02c")
        elif "geom" in lab:
            colors.append("#1f77b4")
        elif "mag" in lab:
            colors.append("#ff7f0e")
        else:
            colors.append("#7f7f7f")

    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    x = np.arange(len(labels))
    bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0.0295, color="gray", ls="--", lw=1, label="FP16 mean")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("TUM fr1 mean ATE RMSE (m)")
    ax.set_title("Track-A shortlist (calib, subsample 2)")
    for b, d in zip(bars, deltas):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.00025,
            f"{d:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0.028, 0.035)
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out}")


def write_shortlist_md(out: Path):
    lines = [
        "# Track-A shortlist (TUM fr1 mean ATE)",
        "",
        "Source EXPs under `results/20260922-*.json` and `results/20260923-*.json`.",
        "",
        "| config | mean ATE (m) | vs FP16 | role |",
        "|--------|-------------:|-------:|------|",
    ]
    for lab, mean, d, note in SHORTLIST:
        lines.append(f"| {lab} | {mean:.4f} | {d:+.1f}% | {note} |")
    lines += [
        "",
        "## H3 (matched K=7 W4 protect)",
        "",
        "| allocator | mean ATE | vs FP16 | vs uniform W4 |",
        "|-----------|---------:|--------:|--------------:|",
        "| Geometry greedy | 0.0316 | +7.1% | −5.1% |",
        "| Magnitude L1 | 0.0328 | +10.9% | −1.7% |",
        "| Uniform W4 | 0.0333 | +12.8% | — |",
        "",
        "Geometry beats magnitude by 3.6% relative mean ATE at equal unit budget.",
        "",
    ]
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def main():
    deltas = load_deltas()
    plot_sensitivity_heatmap(deltas, FIG / "sensitivity_heatmap.png")
    plot_shortlist_bars(FIG / "shortlist_pareto_bars.png")
    write_shortlist_md(FIG / "shortlist_table.md")


if __name__ == "__main__":
    main()
