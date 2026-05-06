"""Visual-information-decay ablation figure + crossover table.

Reads results/blur_sweep/aggregate.csv (produced by aggregate.py) and writes:
  paper/MedVIGIL/figures/fig_visual_decay.{pdf,png,svg}
  paper/MedVIGIL/figures/visual_decay_crossover.csv
  paper/MedVIGIL/figures/visual_decay_crossover.tex
  paper/MedVIGIL/figures/fig_visual_decay_per_tier.{pdf,png}

Main figure: 1 row x 4 columns, one panel per model. x = blur sigma (log
scale), y = MCQ exact-letter accuracy. Two solid lines per panel:
  * image-required cases (text_only_answerable=False)  -- "needs vision"
  * text-answerable  cases (text_only_answerable=True) -- vision-irrelevant control
Plus a horizontal dashed line at the no-image (sigma=inf) accuracy on the
image-required group, marking the language-prior floor.

Crossover L*: the smallest sigma at which image-required accuracy is within
TOLERANCE percentage points of the no-image floor, i.e. visual information
no longer adds anything beyond what the language prior already supplies.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AGG_CSV = ROOT / "results" / "blur_sweep" / "aggregate.csv"
OUT_DIR = ROOT / "paper" / "MedVIGIL" / "figures"
FONT_DIR = ROOT / "paper" / "MedVIGIL" / "fonts"

# Look in either MedVIGIL/ or MedVIGIL_NeurIPS2026/ depending on local layout.
for cand in (ROOT / "paper" / "MedVIGIL_NeurIPS2026" / "figures",
             ROOT / "paper" / "MedVIGIL" / "figures"):
    if cand.parent.exists():
        OUT_DIR = cand
        FONT_DIR = cand.parent / "fonts"
        break

OUT_DIR.mkdir(parents=True, exist_ok=True)

for fp in (FONT_DIR / "mscorefonts" / "Comic.TTF",
           FONT_DIR / "mscorefonts" / "Comicbd.TTF"):
    if fp.exists():
        font_manager.fontManager.addfont(str(fp))

# Overall figure uses Comic Sans MS (matching other paper figures).
# A few specific labels (sigma=0, no-img, L*) are switched per-call to
# DejaVu Sans because Comic Sans MS lacks Greek glyphs and would mix fonts
# within the same string.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Comic Sans MS", "Arial", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.size"] = 10
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

ANNOT_FONT = {"family": "DejaVu Sans"}


# Display labels and colour for each model
MODEL_DISPLAY = {
    "gpt-4o":                       ("GPT-4o",            "#10a37f"),
    "claude-sonnet-4-6":            ("Claude Sonnet 4.6", "#cc785c"),
    "Qwen--Qwen3.5-397B-A17B":      ("Qwen3.5-397B",      "#615ced"),
    "huatuogpt-vision-7b":          ("HuatuoGPT-V 7B",    "#d62728"),
}
MODELS = list(MODEL_DISPLAY.keys())

SIGMA_ORDER = ["0", "2", "4", "8", "16", "32", "64", "inf"]
# L*: smallest finite sigma at which the model has lost (1 - LOSS_FRACTION_THRESHOLD)
# of its total visual contribution, i.e. (acc(sigma) - acc(inf)) / (acc(0) - acc(inf)) <= LOSS_FRACTION_THRESHOLD.
# Smaller L* => model is dominated by language priors at lower blur.
LOSS_FRACTION_THRESHOLD = 0.20


def sigma_to_x(s: str, x_inf: float) -> float:
    """Map sigma string to plot x. inf -> x_inf marker; 0 -> 1 (so log scale works)."""
    if s == "inf":
        return x_inf
    v = float(s)
    return max(v, 1.0)


def load_aggregate() -> dict:
    """Returns d[model][sigma][group][tier] = dict(n, acc)."""
    d: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    with AGG_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            d[r["model_id"]][r["sigma"]][r["group"]][r["tier"]] = {
                "n": int(r["n"]),
                "acc": float(r["acc"]) * 100,
            }
    return d


def compute_crossover(curves_acc: dict[str, float], floor_acc: float,
                       baseline_acc: float, frac: float = LOSS_FRACTION_THRESHOLD) -> str | None:
    """Smallest finite sigma at which the residual visual contribution is at
    most `frac` of the original visual contribution. Returns None if no
    finite sigma satisfies the condition (model still uses vision even at
    the largest sigma in the sweep)."""
    span = baseline_acc - floor_acc
    if span <= 0:
        return None  # baseline already at or below floor, ill-defined
    for s in SIGMA_ORDER:
        if s in ("0", "inf"):
            continue
        if s not in curves_acc:
            continue
        residual = (curves_acc[s] - floor_acc) / span
        if residual <= frac:
            return s
    return None


def main() -> None:
    if not AGG_CSV.exists():
        raise SystemExit(f"missing {AGG_CSV}; run aggregate.py first")
    data = load_aggregate()

    # -------- main figure: 1 x N panels --------
    n_panels = sum(1 for m in MODELS if m in data)
    if n_panels == 0:
        raise SystemExit("no model data found")

    fig, axes = plt.subplots(1, n_panels, figsize=(3.6 * n_panels, 3.5),
                              sharey=True, constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    # x positions: log-spaced for finite sigmas; the "inf" gets its own slot
    finite_sigmas = [s for s in SIGMA_ORDER if s != "inf" and s != "0"]
    x_inf = float(max(int(s) for s in finite_sigmas)) * 4  # one log-step beyond max finite

    crossovers: dict[str, str | None] = {}
    table_rows: list[dict] = []

    panel_i = 0
    for model in MODELS:
        if model not in data:
            continue
        ax = axes[panel_i]; panel_i += 1
        label, color = MODEL_DISPLAY[model]

        # Image-required curve (group=image_required, tier=all)
        ir_acc = {s: data[model].get(s, {}).get("image_required", {}).get("all", {}).get("acc", math.nan)
                  for s in SIGMA_ORDER}
        ta_acc = {s: data[model].get(s, {}).get("text_answerable",  {}).get("all", {}).get("acc", math.nan)
                  for s in SIGMA_ORDER}

        # Build full curves including sigma=0 (plotted at x=1 on log axis)
        all_x = [1.0] + [sigma_to_x(s, x_inf) for s in finite_sigmas]
        all_y_ir = [ir_acc["0"]] + [ir_acc[s] for s in finite_sigmas]
        all_y_ta = [ta_acc["0"]] + [ta_acc[s] for s in finite_sigmas]
        valid = [(x, y, y2) for x, y, y2 in zip(all_x, all_y_ir, all_y_ta) if not math.isnan(y)]
        finite_x = [t[0] for t in valid]
        finite_y = [t[1] for t in valid]
        finite_y_ta = [t[2] for t in valid]

        # Compute crossover first so we can shade
        baseline = ir_acc.get("0", math.nan)
        floor = ir_acc.get("inf", math.nan)
        ls = compute_crossover({s: ir_acc[s] for s in finite_sigmas if not math.isnan(ir_acc[s])},
                                floor_acc=floor, baseline_acc=baseline)
        crossovers[model] = ls

        # Subtle "language-prior dominated" shading right of L*
        if ls is not None:
            xs = sigma_to_x(ls, x_inf)
            ax.axvspan(xs, x_inf * 1.5, color=color, alpha=0.06, zorder=0)

        # No-image floor band (sits behind curves)
        if not math.isnan(floor):
            ax.axhline(floor, color="0.5", ls=":", lw=1.0, alpha=0.6, zorder=1)

        # Plot text-answerable control (background)
        if any(not math.isnan(v) for v in finite_y_ta):
            ax.plot(finite_x, finite_y_ta, "s--", color=color, lw=1.2, ms=4,
                    alpha=0.45, zorder=2, label="text-answerable")

        # Plot image-required (load-bearing curve)
        ax.plot(finite_x, finite_y, "o-", color=color, lw=2.4, ms=6.5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4, label="needs vision")

        # Sigma=0 tag — placed clearly above-right so it does not overlap the marker
        if not math.isnan(ir_acc["0"]):
            ax.annotate(f"σ=0: {ir_acc['0']:.0f}",
                        (1.0, ir_acc["0"]),
                        xytext=(8, 10), textcoords="offset points",
                        fontsize=9, color=color, fontweight="bold", zorder=6,
                        **ANNOT_FONT)

        # No-image floor marker + tag (left of marker so they don't overlap)
        if not math.isnan(floor):
            ax.scatter([x_inf], [floor], marker="X", color="0.30", s=70,
                        zorder=5, edgecolor="white", lw=0.8)
            ax.annotate(f"no-img: {floor:.0f}",
                        (x_inf, floor),
                        xytext=(-8, -1), textcoords="offset points",
                        fontsize=8, color="0.25", ha="right", va="center",
                        zorder=6, **ANNOT_FONT)

        # L* axvline + label as a small badge near top
        if ls is not None:
            xs = sigma_to_x(ls, x_inf)
            ax.axvline(xs, color=color, ls="--", lw=1.1, alpha=0.55, zorder=3)
            ax.annotate(rf"$L^{{*}}\!=\!{ls}$",
                        (xs, 92), color=color, fontsize=11.5,
                        ha="center", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.28",
                                   fc="white", ec=color, lw=0.8, alpha=0.95))

        ax.set_xscale("log")
        ax.set_xlim(0.7, x_inf * 1.6)
        ax.set_ylim(0, 100)
        # custom xticks: sigma=0 at x=1, then 2,4,8,16,32,64, then inf
        xticks = [1.0] + [float(s) for s in finite_sigmas] + [x_inf]
        xticklabels = ["0"] + finite_sigmas + [r"$\infty$"]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=9, **ANNOT_FONT)
        ax.set_yticks([0, 25, 50, 75, 100])
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(9)
            lbl.set_fontfamily(ANNOT_FONT["family"])
        ax.set_xlabel(r"blur $\sigma$ (px)", fontsize=10.5, **ANNOT_FONT)
        if panel_i == 1:
            ax.set_ylabel("MCQ accuracy (%)", fontsize=10.5, **ANNOT_FONT)
        # Tick mark style
        ax.tick_params(axis="both", which="major", length=4.5, width=0.8,
                        color="0.35", direction="out", pad=3)
        # Per-panel header (replaces suptitle) — Comic Sans for brand consistency
        ax.text(0.5, 1.02, label, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=12,
                color=color, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.30, lw=0.4)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("0.35"); ax.spines["left"].set_linewidth(0.9)
        ax.spines["bottom"].set_color("0.35"); ax.spines["bottom"].set_linewidth(0.9)
        leg = ax.legend(loc="lower left", fontsize=8.5, frameon=True,
                          framealpha=0.95, handlelength=1.6, handletextpad=0.5,
                          borderpad=0.45, borderaxespad=0.35)
        for txt in leg.get_texts():
            txt.set_fontfamily(ANNOT_FONT["family"])
        leg.get_frame().set_edgecolor("0.7")
        leg.get_frame().set_linewidth(0.4)

        # gather table rows
        for tier in ("all", "L1", "L2", "L3", "L4", "L5"):
            tier_curve = {s: data[model].get(s, {}).get("image_required", {}).get(tier, {}).get("acc", math.nan)
                          for s in SIGMA_ORDER}
            tier_floor = tier_curve.get("inf", math.nan)
            row = {
                "model": model, "model_label": label, "tier": tier,
                "acc_sigma0": tier_curve.get("0", math.nan),
                "acc_noimg": tier_floor,
                "drop": (tier_curve.get("0", math.nan) - tier_floor) if not math.isnan(tier_floor) else math.nan,
            }
            for s in finite_sigmas:
                row[f"acc_sigma{s}"] = tier_curve.get(s, math.nan)
            row["L_star"] = compute_crossover(
                {s: tier_curve[s] for s in finite_sigmas if not math.isnan(tier_curve[s])},
                floor_acc=tier_floor, baseline_acc=tier_curve.get("0", math.nan),
            )
            table_rows.append(row)

    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"fig_visual_decay.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -------- per-tier figure --------
    # Sequential blue->red palette: low risk = cool, high risk = warm.
    # Hand-tuned for white background + colour-blind-friendly hue separation.
    tier_colors = {
        "L1": "#1A6FB0",  # deep blue
        "L2": "#2E9F8F",  # teal
        "L3": "#A1A12B",  # olive (replaces washed-out yellow)
        "L4": "#D17A2B",  # orange
        "L5": "#B82C2C",  # deep red
    }
    tier_labels = {
        "L1": "L1 (modality / meta)",
        "L2": "L2 (low-stakes)",
        "L3": "L3 (mid)",
        "L4": "L4 (high-stakes)",
        "L5": "L5 (don't-miss)",
    }

    fig2, axes2 = plt.subplots(1, n_panels, figsize=(3.7 * n_panels, 4.05),
                                sharey=True)
    if n_panels == 1:
        axes2 = [axes2]
    fig2.subplots_adjust(top=0.78, bottom=0.16, left=0.05, right=0.985,
                          wspace=0.10)

    # X-marker x-position spread — render each tier's no-image floor at a
    # slightly different x so they don't pile up.
    inf_dx = {"L1": 0.85, "L2": 0.95, "L3": 1.05, "L4": 1.15, "L5": 1.25}

    panel_i = 0
    for model in MODELS:
        if model not in data:
            continue
        ax = axes2[panel_i]; panel_i += 1
        label, model_color = MODEL_DISPLAY[model]

        for tier in ("L1", "L2", "L3", "L4", "L5"):
            curve = {s: data[model].get(s, {}).get("image_required", {}).get(tier, {}).get("acc", math.nan)
                     for s in SIGMA_ORDER}

            xs_full = [1.0] + [sigma_to_x(s, x_inf) for s in finite_sigmas]
            ys_full = [curve["0"]] + [curve[s] for s in finite_sigmas]
            valid = [(x, y) for x, y in zip(xs_full, ys_full) if not math.isnan(y)]
            if not valid:
                continue
            xs, ys = zip(*valid)

            ax.plot(xs, ys, "-", color=tier_colors[tier], lw=1.9, alpha=0.95,
                    zorder=4, label=tier)
            ax.plot(xs, ys, "o", color=tier_colors[tier], ms=4.3,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=5)

            # No-image floor X markers, staggered horizontally per tier
            floor_t = curve.get("inf", math.nan)
            if not math.isnan(floor_t):
                ax.scatter([x_inf * inf_dx[tier]], [floor_t], marker="X",
                            color=tier_colors[tier], s=58, edgecolor="white",
                            lw=0.7, zorder=6)

        # Visual divider separating finite sigmas from the noimg group
        ax.axvline(x_inf * 0.7, color="0.85", lw=0.7, ls="-", zorder=1)

        ax.set_xscale("log")
        ax.set_xlim(0.7, x_inf * 1.55)
        ax.set_ylim(-2, 100)
        xticks = [1.0] + [float(s) for s in finite_sigmas] + [x_inf * 1.05]
        xticklabels = ["0"] + finite_sigmas + [r"$\infty$"]
        ax.set_xticks(xticks); ax.set_xticklabels(xticklabels, fontsize=9, **ANNOT_FONT)
        ax.set_yticks([0, 25, 50, 75, 100])
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(9); lbl.set_fontfamily(ANNOT_FONT["family"])
        ax.set_xlabel(r"blur $\sigma$ (px)", fontsize=10.5, **ANNOT_FONT)
        if panel_i == 1:
            ax.set_ylabel("MCQ accuracy (%)", fontsize=10.5, **ANNOT_FONT)
        ax.tick_params(axis="both", which="major", length=4.5, width=0.8,
                        color="0.35", direction="out", pad=3)
        ax.text(0.5, 1.02, label, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=12,
                color=model_color, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25, lw=0.4)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("0.35"); ax.spines["left"].set_linewidth(0.9)
        ax.spines["bottom"].set_color("0.35"); ax.spines["bottom"].set_linewidth(0.9)

    # Shared legend ABOVE the panels (with padding so it doesn't collide with
    # per-panel headers); the headers and legend share vertical real estate
    # because we widened top margin above.
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color=tier_colors[t], lw=2.0, ms=5.5,
                    markeredgecolor="white", markeredgewidth=0.6,
                    label={"L1": "L1 modality / meta",
                           "L2": "L2 low-stakes",
                           "L3": "L3 mid",
                           "L4": "L4 high-stakes",
                           "L5": "L5 don't-miss"}[t])
        for t in ("L1", "L2", "L3", "L4", "L5")
    ]
    inf_handle = plt.Line2D([0], [0], marker="X", color="0.4", lw=0,
                              ms=7.0, markeredgecolor="white",
                              markeredgewidth=0.6, label=r"$\sigma=\infty$ (no image)")
    leg = fig2.legend(handles=legend_handles + [inf_handle], loc="upper center",
                       ncols=6, fontsize=9.5, frameon=False,
                       bbox_to_anchor=(0.5, 0.965),
                       handletextpad=0.4, columnspacing=1.5)
    for txt in leg.get_texts():
        txt.set_fontfamily(ANNOT_FONT["family"])
    for ext in ("pdf", "png"):
        fig2.savefig(OUT_DIR / f"fig_visual_decay_per_tier.{ext}", dpi=200)
    plt.close(fig2)

    # -------- crossover table CSV --------
    csv_out = OUT_DIR / "visual_decay_crossover.csv"
    fieldnames = ["model", "model_label", "tier", "acc_sigma0",
                  *[f"acc_sigma{s}" for s in finite_sigmas],
                  "acc_noimg", "drop", "L_star"]
    with csv_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in table_rows:
            w.writerow({k: ("" if (isinstance(r.get(k), float) and math.isnan(r[k])) else r.get(k, "")) for k in fieldnames})

    # console summary (the LaTeX table is now hand-edited directly in
    # neurips_2026.tex; we no longer auto-write a visual_decay_crossover.tex)
    print(f"\n[ok] main figure  -> {OUT_DIR / 'fig_visual_decay.pdf'}")
    print(f"[ok] per-tier fig -> {OUT_DIR / 'fig_visual_decay_per_tier.pdf'}")
    print(f"[ok] csv table    -> {csv_out}")
    print("\nCrossover L* per model (overall, image-required cases):")
    for m, ls in crossovers.items():
        label = MODEL_DISPLAY[m][0]
        print(f"  {label:<22} L* = {'>=64' if ls is None else ls}")


if __name__ == "__main__":
    main()
