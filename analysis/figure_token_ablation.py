"""Token-ablation figure: progressive removal of key tokens vs. model confidence."""
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


def _render_text_panel(ax):
    """Render four progressive question variants stacked vertically.

    Uses one ax.text() call per row so character spacing matches the
    rendered glyph metrics; struck-through tokens are rendered with the
    Unicode combining-strikethrough character and a muted colour.
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

    # Compose each row as a single string.
    # We render the line with rich-text-style coloring by drawing
    # struck-through tokens in a separate ax.text call, layered on top
    # of the same baseline. This avoids per-glyph layout entirely.

    n_steps = len(strike_steps)
    margin_top = 0.96
    row_h = 0.92 / n_steps

    for i, (idxs, label, sub) in enumerate(strike_steps):
        y_label = margin_top - i * row_h - 0.02
        y_text  = y_label - 0.085

        ax.text(0.0,   y_label, label, fontsize=9, fontweight="bold",
                color="#1f2937", transform=ax.transAxes)
        ax.text(0.165, y_label, sub, fontsize=8.6, color="#6b7280",
                style="italic", transform=ax.transAxes)

        # Build a single-line rendering by inserting struck-through tokens
        # at the right positions.
        line_tokens = []
        for j, tok in enumerate(tokens):
            line_tokens.append(strike(tok) if j in idxs else tok)
        line = " ".join(line_tokens)
        ax.text(0.0, y_text, line,
                fontsize=10.4, color="#1f2937",
                transform=ax.transAxes,
                fontfamily="DejaVu Sans")

        if i < n_steps - 1:
            ax.plot([0.0, 1.0], [y_text - 0.045, y_text - 0.045],
                    color="0.85", linewidth=0.6, transform=ax.transAxes)


def _render_curves_panel(ax_conf, ax_stab, by_model):
    for ax in (ax_conf, ax_stab):
        for s in ax.spines.values():
            s.set_color("0.35"); s.set_linewidth(0.9)
        ax.tick_params(length=4, width=0.8, color="0.35", labelsize=8.5)
        ax.grid(axis="y", alpha=0.18, linewidth=0.8)

    step_labels = ["full", "− laterality", "− anatomy", "skeleton"]
    xs = [0, 1, 2, 3]

    for model, recs in by_model.items():
        ys = [r["conf"] for r in recs]
        ax_conf.plot(xs, ys, marker="o", linewidth=1.7, markersize=4.5,
                     color=PROVIDER_COLOR.get(model, "0.3"), label=model)
    ax_conf.set_ylabel("Mean answer\nconfidence  (%)", fontsize=9)
    ax_conf.set_ylim(45, 95)
    ax_conf.set_yticks([50, 60, 70, 80, 90])
    ax_conf.set_xticks(xs); ax_conf.set_xticklabels([""] * len(xs))
    ax_conf.set_title("Confidence  ·  letter-stability rate vs. token removal",
                      loc="left", fontsize=10.5, fontweight="bold",
                      color="#1f2937", pad=4)
    ax_conf.legend(loc="lower left", fontsize=8.0, frameon=False, ncol=2,
                   handlelength=1.2, columnspacing=0.8)

    for model, recs in by_model.items():
        ys = [r["stability"] for r in recs]
        ax_stab.plot(xs, ys, marker="s", linewidth=1.5, markersize=4.0,
                     color=PROVIDER_COLOR.get(model, "0.3"), linestyle="--")
    ax_stab.set_ylabel("Letter-stability\nrate vs. step 0  (%)", fontsize=9)
    ax_stab.set_ylim(15, 105)
    ax_stab.set_yticks([25, 50, 75, 100])
    ax_stab.set_xticks(xs); ax_stab.set_xticklabels(step_labels, fontsize=8.5)
    ax_stab.set_xlabel("Token removal step", fontsize=9, labelpad=4)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_model = _load_rows()

    fig = plt.figure(figsize=(11.0, 4.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 0.85],
                          wspace=0.32, hspace=0.05,
                          left=0.03, right=0.985, top=0.93, bottom=0.12)
    ax_text = fig.add_subplot(gs[:, 0])
    ax_conf = fig.add_subplot(gs[0, 1])
    ax_stab = fig.add_subplot(gs[1, 1])

    _render_text_panel(ax_text)
    _render_curves_panel(ax_conf, ax_stab, by_model)

    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{OUT_BASE}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {OUT_BASE}.{{pdf,png,svg}}")


if __name__ == "__main__":
    main()
