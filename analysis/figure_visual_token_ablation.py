"""Visual-token ablation figure (paper-style, matching figure_visual_decay)."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]

FONT_DIR = ROOT / "paper" / "MedVIGIL_NeurIPS2026" / "fonts"
if not FONT_DIR.exists():
    for cand in (ROOT / "paper").glob("*"):
        if (cand / "fonts").exists():
            FONT_DIR = cand / "fonts"
            break
for fp in (FONT_DIR / "mscorefonts" / "Comic.TTF",
            FONT_DIR / "mscorefonts" / "Comicbd.TTF"):
    if fp.exists():
        font_manager.fontManager.addfont(str(fp))

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Comic Sans MS", "Arial", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.size"] = 10
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
ANNOT_FONT = {"family": "DejaVu Sans"}

BENCH = ROOT / "data/medvlm_bench_v1"
CSV_PATH = BENCH / "visual_token_ablation.csv"
RAW_PATH = BENCH / "_visual_token_raw.jsonl"
IMG_DIR  = BENCH / "images_visualtoken"
GROUNDING = BENCH / "grounding.csv"
OUT_DIR = ROOT / "paper/MedVIGIL_NeurIPS2026/figures"
OUT_BASE = OUT_DIR / "fig_visual_token_ablation"

EXAMPLE_CASE = "MVB-0031"
PANEL_LETTERS = ["(a)", "(b)", "(c)", "(d)"]
STEP_LABELS = ["full",
               "33% ROI masked",
               "67% ROI masked",
               "100% ROI masked"]

MODEL_DISPLAY = {
    "GPT-5.5":         "#10a37f",
    "Claude Opus 4.7": "#cc785c",
    "Gemini 3 Flash":  "#615ced",
}
# DejaVu Serif for the inline letter annotations: visually distinct
# from the Comic Sans body labels and available without external fonts.
LETTER_FONT = {"family": "DejaVu Serif"}
MASK_COLOR = "#d62728"
ROI_BOX    = "#1f6dc4"


def _wilson_ci(p_pct: float, n: int, z: float = 1.96):
    """Wilson 95% CI for a proportion. p_pct in 0..100, n integer.
    Returns (lo_pct, hi_pct)."""
    if n <= 0:
        return p_pct, p_pct
    p = p_pct / 100.0
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return (centre - half) * 100.0, (centre + half) * 100.0


def _load_rows():
    rows = list(csv.DictReader(CSV_PATH.open()))
    by_model = defaultdict(list)
    for r in rows:
        n = int(r.get("n_cases", 0) or 0)
        ref = float(r.get("refusal_pct", 0.0))
        sw = float(r.get("switch_from_s0_pct", 0.0))
        ref_lo, ref_hi = _wilson_ci(ref, n)
        sw_lo,  sw_hi  = _wilson_ci(sw,  n)
        by_model[r["model"]].append({
            "step": int(r["step"]),
            "n": n,
            "refusal": ref, "refusal_lo": ref_lo, "refusal_hi": ref_hi,
            "switch":  sw,  "switch_lo":  sw_lo,  "switch_hi":  sw_hi,
            "stability": float(r.get("letter_stability_pct", 100.0)),
        })
    for m in by_model:
        by_model[m].sort(key=lambda r: r["step"])
    return by_model


def _load_example_letters():
    out = {}
    if RAW_PATH.exists():
        for line in RAW_PATH.open():
            r = json.loads(line)
            if r.get("case_id") == EXAMPLE_CASE:
                out[(r["model"], int(r["step"]))] = (r["letter"], float(r["confidence"]))
    return out


def _load_roi_bbox(case_id):
    for r in csv.DictReader(GROUNDING.open()):
        if r["case_id"] == case_id:
            try:
                return [float(x) for x in json.loads(r.get("roi_bbox_norm", "null"))]
            except Exception:
                return None
    return None


def _draw_thumbnail(ax, k, bbox_norm):
    path = IMG_DIR / f"{EXAMPLE_CASE}_step{k}.jpg"
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.40"); s.set_linewidth(0.7)
    if not path.exists():
        ax.text(0.5, 0.5, "(missing)", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="0.55")
        return
    img = mpimg.imread(path)
    H, W = img.shape[:2]
    ax.imshow(img, aspect="equal")
    if bbox_norm is not None:
        x0n, y0n, x1n, y1n = bbox_norm
        if x1n <= x0n or y1n <= y0n:
            xn, yn, wn, hn = bbox_norm
            x0n, y0n, x1n, y1n = xn, yn, xn + wn, yn + hn
        x0, y0 = x0n * W, y0n * H
        roi_w, roi_h = (x1n - x0n) * W, (y1n - y0n) * H
        ax.add_patch(Rectangle((x0, y0), roi_w, roi_h, fill=False,
                                edgecolor=ROI_BOX, linewidth=1.6))
        if k > 0:
            frac = k / 3.0
            mask_h = frac * roi_h
            ax.add_patch(Rectangle((x0, y0), roi_w, mask_h, fill=True,
                                    facecolor=MASK_COLOR, alpha=0.30,
                                    edgecolor=MASK_COLOR, linewidth=1.6))
    ax.set_title(f"{PANEL_LETTERS[k]} {STEP_LABELS[k]}",
                 loc="left", fontsize=9.6, color="#1f2937", pad=4)


def _style_axis(ax):
    for s in ax.spines.values():
        s.set_color("0.35"); s.set_linewidth(0.9)
    ax.tick_params(axis="both", which="major",
                    length=4.5, width=0.8, color="0.35",
                    direction="out", pad=3)
    ax.grid(axis="y", color="0.85", linewidth=0.6, alpha=0.6)


def _render_curves(ax_ref, ax_sw, by_model):
    _style_axis(ax_ref); _style_axis(ax_sw)
    xs = [0, 1, 2, 3]
    example_letters = _load_example_letters()

    # --- top: refusal rate (% of cases that pick option E) ---
    # Stagger letter-annotation x offsets per model so labels do not stack on
    # top of each other when two models pick the same letter at the same step.
    model_xoff = {m: dx for m, dx in zip(by_model.keys(), [-0.12, 0.0, 0.12])}
    for model, recs in by_model.items():
        ys   = [r["refusal"]    for r in recs]
        ylo  = [r["refusal_lo"] for r in recs]
        yhi  = [r["refusal_hi"] for r in recs]
        color = MODEL_DISPLAY.get(model, "0.3")
        ax_ref.fill_between(xs, ylo, yhi, color=color, alpha=0.10, zorder=1,
                              linewidth=0)
        ax_ref.plot(xs, ys, color=color, linewidth=2.2, alpha=0.95, zorder=2)
        ax_ref.scatter(xs, ys, s=70, color=color,
                        edgecolor="white", linewidth=1.2, zorder=3, label=model)
        dx = model_xoff.get(model, 0.0)
        for x_i, y_i in zip(xs, ys):
            letter, _ = example_letters.get((model, x_i), ("", 0.0))
            if letter:
                ax_ref.annotate(letter, xy=(x_i + dx, y_i),
                                xytext=(0, 9), textcoords="offset points",
                                ha="center", va="bottom",
                                fontsize=11.5, fontweight="bold",
                                color=color, zorder=4, **LETTER_FONT)

    ax_ref.set_ylabel("Refusal rate\n(% picking option E)", fontsize=10.2, **ANNOT_FONT)
    ax_ref.set_ylim(-3, 75); ax_ref.set_yticks([0, 25, 50, 75])
    ax_ref.set_xlim(-0.3, 3.4)
    ax_ref.set_xticks(xs); ax_ref.set_xticklabels([""] * len(xs))
    for lbl in ax_ref.get_yticklabels():
        lbl.set_fontfamily(ANNOT_FONT["family"])

    # --- bottom: switch rate from step 0, with 95% Wilson CI band ---
    for model, recs in by_model.items():
        ys   = [r["switch"]    for r in recs]
        ylo  = [r["switch_lo"] for r in recs]
        yhi  = [r["switch_hi"] for r in recs]
        color = MODEL_DISPLAY.get(model, "0.3")
        ax_sw.fill_between(xs, ylo, yhi, color=color, alpha=0.10, zorder=1,
                             linewidth=0)
        ax_sw.plot(xs, ys, color=color, linewidth=2.0,
                    linestyle="--", alpha=0.95, zorder=2)
        ax_sw.scatter(xs, ys, s=55, color=color,
                        edgecolor="white", linewidth=1.0, zorder=3,
                        marker="s")

    ax_sw.set_ylabel("Letter-switch rate\n(% changed vs. step 0)",
                      fontsize=10.2, **ANNOT_FONT)
    ax_sw.set_xlabel("ROI-mask step (letters above markers = "
                      "modal letter on MVB-0031)",
                      fontsize=10.0, labelpad=4, **ANNOT_FONT)
    ax_sw.set_ylim(-3, 75); ax_sw.set_yticks([0, 25, 50, 75])
    ax_sw.set_xlim(-0.3, 3.4)
    ax_sw.set_xticks(xs)
    ax_sw.set_xticklabels(["(a) full", "(b) 33%", "(c) 67%", "(d) 100%"],
                            fontsize=9.6)
    for lbl in ax_sw.get_xticklabels():
        lbl.set_fontfamily(ANNOT_FONT["family"])
    for lbl in ax_sw.get_yticklabels():
        lbl.set_fontfamily(ANNOT_FONT["family"])

    handles = [Line2D([0],[0], marker='o', linestyle='-',
                       color=MODEL_DISPLAY[m], markersize=9,
                       markerfacecolor=MODEL_DISPLAY[m],
                       markeredgecolor='white', markeredgewidth=1.2,
                       label=m, linewidth=2.0)
                for m in by_model.keys()]
    ax_sw.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, -0.34), ncol=3, frameon=False,
                  handlelength=1.6, handletextpad=0.5, columnspacing=1.6,
                  fontsize=9.8)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_model = _load_rows()
    bbox = _load_roi_bbox(EXAMPLE_CASE)

    fig = plt.figure(figsize=(11.6, 5.4))
    outer = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.30],
                              wspace=0.18,
                              left=0.045, right=0.99, top=0.97, bottom=0.20)
    left_gs = outer[0, 0].subgridspec(2, 2, wspace=0.10, hspace=0.20)
    img_axes = [[fig.add_subplot(left_gs[i, j]) for j in range(2)]
                  for i in range(2)]

    right_gs = outer[0, 1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.12)
    ax_ref = fig.add_subplot(right_gs[0, 0])
    ax_sw  = fig.add_subplot(right_gs[1, 0])

    for k in range(4):
        _draw_thumbnail(img_axes[k // 2][k % 2], k, bbox)
    _render_curves(ax_ref, ax_sw, by_model)

    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{OUT_BASE}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {OUT_BASE}.{{pdf,png,svg}}")


if __name__ == "__main__":
    main()
