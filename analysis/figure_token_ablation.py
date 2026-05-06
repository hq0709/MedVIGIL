"""Token-ablation figure: progressive removal of key tokens vs. confidence.

Layout (matches the user's spec: token viz on the left, one line plot on the
right):
  Left  panel — the example question (MVB-0031) rendered four times with
                progressively more clinically informative tokens struck
                through (laterality -> anatomy -> finding).
  Right panel — a single confidence trajectory plot per model. Marker SIZE
                at each step is proportional to the letter-stability rate
                (fraction of cases where the step-k modal letter equals
                the step-0 letter). A model that stays high-confidence
                with large markers is keeping the same answer despite the
                clinical content being removed --- the failure mode.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/medvlm_bench_v1/token_ablation.csv"
OUT_DIR = ROOT / "paper/MedVIGIL_NeurIPS2026/figures"
OUT_BASE = OUT_DIR / "fig_token_ablation"

PROVIDER_COLOR = {
    "GPT-4o":               "#3D7B40",
    "Claude Opus 4.7":      "#C57A2D",
    "Gemini 3.1 Flash-Lite":"#1A6FB0",
    "Qwen3.5-397B":         "#6E2A8C",
}


def strike(word: str) -> str:
    return "".join(c + "̶" for c in word)


def _load_rows():
    rows = list(csv.DictReader(CSV_PATH.open()))
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append({
            "step": int(r["step"]),
            "step_label": r["step_label"],
            "conf": float(r["mean_confidence_pct"]),
            "stability": float(r["letter_stability_pct"]),
        })
    for m in by_model:
        by_model[m].sort(key=lambda r: r["step"])
    return by_model


def _render_text_panel(ax, fig):
    """Per-token rendering with explicit strikethrough line for removed tokens.

    Each token is drawn separately so we can colour-code (light gray for
    removed, dark for kept) and overlay a horizontal red strikethrough
    line on the removed tokens. This is visually unambiguous regardless
    of font / DPI, unlike Unicode combining-strikethrough characters.
    """
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Question variants  ·  MVB-0031", loc="left",
                 fontsize=10.5, fontweight="bold", color="#1f2937", pad=4)

    tokens = ["Is", "there", "evidence", "of", "a",
              "right", "apical", "pneumothorax?"]
    strike_steps = [
        ([],          "Step 0",  "full"),
        ([5],         "Step 1",  "− laterality"),
        ([5, 6],      "Step 2",  "− anatomy"),
        ([5, 6, 7],   "Step 3",  "skeleton"),
    ]
    KEPT = "#1f2937"
    REMOVED = "#a3a8b0"
    STRIKE_COLOUR = "#B82C2C"

    n_steps = len(strike_steps)
    margin_top = 0.96
    row_h = 0.92 / n_steps

    fig.canvas.draw()
    inv = ax.transAxes.inverted()
    renderer = fig.canvas.get_renderer()

    for i, (idxs, label, sub) in enumerate(strike_steps):
        y_label = margin_top - i * row_h - 0.02
        y_text  = y_label - 0.085

        ax.text(0.0,   y_label, label, fontsize=9, fontweight="bold",
                color=KEPT, transform=ax.transAxes)
        ax.text(0.165, y_label, sub, fontsize=8.6, color="#6b7280",
                style="italic", transform=ax.transAxes)

        # Render each token separately and track its bbox in axes coords
        x_cursor = 0.0
        for j, tok in enumerate(tokens):
            removed = j in idxs
            colour = REMOVED if removed else KEPT
            t = ax.text(x_cursor, y_text, tok,
                        fontsize=10.6, color=colour,
                        transform=ax.transAxes,
                        fontfamily="DejaVu Sans")
            # Measure the rendered glyph extent in axes coordinates.
            bb = t.get_window_extent(renderer=renderer).transformed(inv)
            # Overlay a strikethrough line on removed tokens.
            if removed:
                y_mid = (bb.y0 + bb.y1) / 2.0
                ax.plot([bb.x0, bb.x1], [y_mid, y_mid],
                        color=STRIKE_COLOUR, linewidth=1.8,
                        solid_capstyle="round",
                        transform=ax.transAxes, zorder=5)
            x_cursor = bb.x1 + 0.012

        if i < n_steps - 1:
            ax.plot([0.0, 1.0], [y_text - 0.045, y_text - 0.045],
                    color="0.85", linewidth=0.6, transform=ax.transAxes)


def _render_curve_panel(ax, by_model):
    """Single line plot. y = confidence (%); marker size ∝ letter-stability."""
    for s in ax.spines.values():
        s.set_color("0.35"); s.set_linewidth(0.9)
    ax.tick_params(length=4, width=0.8, color="0.35", labelsize=8.8)
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)

    step_labels = ["full", "− laterality", "− anatomy", "skeleton"]
    xs = [0, 1, 2, 3]

    # Marker size scaling: stability 0-100% -> marker area 18-200 (sqrt of size)
    def msize(stab):
        return max(20, 20 + 1.8 * stab)  # area in points^2

    for model, recs in by_model.items():
        ys = [r["conf"] for r in recs]
        stabs = [r["stability"] for r in recs]
        color = PROVIDER_COLOR.get(model, "0.3")
        ax.plot(xs, ys, color=color, linewidth=1.6, alpha=0.85, zorder=2)
        # scatter on top with size encoding
        ax.scatter(xs, ys, s=[msize(s) for s in stabs],
                   color=color, edgecolor="white", linewidth=0.9,
                   zorder=3, label=model)

    ax.set_ylabel("Mean answer confidence  (%)", fontsize=9.5)
    ax.set_xlabel("Token-removal step", fontsize=9.5, labelpad=2)
    ax.set_xticks(xs); ax.set_xticklabels(step_labels, fontsize=9)
    ax.set_ylim(50, 105)
    ax.set_yticks([60, 70, 80, 90, 100])
    ax.set_xlim(-0.25, 3.4)

    ax.set_title("Confidence trajectory  ·  marker size $\\propto$ letter-stability rate",
                 loc="left", fontsize=10.5, fontweight="bold",
                 color="#1f2937", pad=6)

    # Model legend below the plot, left side
    leg1 = ax.legend(loc="upper left", fontsize=8.6, frameon=False,
                     bbox_to_anchor=(-0.02, -0.30), ncol=3,
                     handlelength=1.3, columnspacing=1.0, handletextpad=0.5)
    ax.add_artist(leg1)

    # Stability-size legend below the plot, right side
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0],[0], marker='o', linestyle='None', markerfacecolor='0.55',
               markeredgecolor='white', markersize=v**0.5,
               label=f"{int(s)}%")
        for s, v in [(25, msize(25)), (50, msize(50)),
                     (75, msize(75)), (100, msize(100))]
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              title="letter-stability:", fontsize=7.8, title_fontsize=8.0,
              frameon=False, handlelength=0.5, ncol=4,
              columnspacing=0.5, handletextpad=0.25,
              bbox_to_anchor=(1.02, -0.30))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_model = _load_rows()

    fig = plt.figure(figsize=(11.0, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0],
                          wspace=0.22,
                          left=0.03, right=0.985, top=0.93, bottom=0.27)
    ax_text = fig.add_subplot(gs[0, 0])
    ax_curve = fig.add_subplot(gs[0, 1])

    _render_text_panel(ax_text, fig)
    _render_curve_panel(ax_curve, by_model)

    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{OUT_BASE}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {OUT_BASE}.{{pdf,png,svg}}")


if __name__ == "__main__":
    main()
