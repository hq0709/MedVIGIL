"""Visual-token ablation figure (paper-style, matching figure_visual_decay)."""
from __future__ import annotations

import csv
import json
import math
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

# Match the typography used elsewhere in the paper figures.
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
    "GPT-4o":               "#10a37f",
    "Claude Opus 4.7":      "#cc785c",
    "Gemini 3.1 Flash-Lite":"#615ced",
}
MASK_COLOR = "#d62728"
ROI_BOX    = "#1f6dc4"


def _load_rows():
    rows = list(csv.DictReader(CSV_PATH.open()))
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append({
            "step": int(r["step"]),
            "conf": float(r["mean_confidence_pct"]),
            "stability": float(r["letter_stability_pct"]),
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
                                edgecolor=ROI_BOX, linewidth=1.6,
                                linestyle="-"))
        if k > 0:
            frac = k / 3.0
            mask_h = frac * roi_h
            ax.add_patch(Rectangle((x0, y0), roi_w, mask_h, fill=True,
                                    facecolor=MASK_COLOR, alpha=0.30,
                                    edgecolor=MASK_COLOR, linewidth=1.6))
    ax.set_title(f"{PANEL_LETTERS[k]} {STEP_LABELS[k]}",
                 loc="left", fontsize=9.6, color="#1f2937", pad=4)


def _style_curve_axis(ax):
    for s in ax.spines.values():
        s.set_color("0.35"); s.set_linewidth(0.9)
    ax.tick_params(axis="both", which="major",
                    length=4.5, width=0.8, color="0.35",
                    direction="out", pad=3)


def _render_curve_panel(ax, by_model):
    _style_curve_axis(ax)
    ax.grid(axis="y", color="0.85", linewidth=0.6, alpha=0.6)

    xs = [0, 1, 2, 3]

    def msize(stab):
        # Marker AREA (s argument of scatter) — gives a wide, visible range.
        return max(45, 45 + 4.0 * stab)

    example_letters = _load_example_letters()

    for model, recs in by_model.items():
        ys = [r["conf"] for r in recs]
        stabs = [r["stability"] for r in recs]
        color = MODEL_DISPLAY.get(model, "0.3")
        ax.plot(xs, ys, color=color, linewidth=2.2, alpha=0.95, zorder=2)
        ax.scatter(xs, ys, s=[msize(s) for s in stabs],
                   color=color, edgecolor="white", linewidth=1.2,
                   zorder=3, label=model)
        # Annotate each marker with the example case's modal letter
        for x_i, y_i, st in zip(xs, ys, stabs):
            letter, _ = example_letters.get((model, x_i), ("", 0.0))
            if letter:
                offset = 14 + 0.06 * st
                ax.annotate(letter, xy=(x_i, y_i),
                            xytext=(0, offset), textcoords="offset points",
                            ha="center", va="bottom",
                            fontsize=10, fontweight="bold",
                            color=color, zorder=4, **ANNOT_FONT)

    ax.set_ylabel("Mean answer confidence  (%)", fontsize=10.5, **ANNOT_FONT)
    ax.set_xlabel("ROI-mask step (example: MVB-0031)", fontsize=10.5, labelpad=4, **ANNOT_FONT)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"(a) full", "(b) 33%", "(c) 67%", "(d) 100%"],
                        fontsize=9.6)
    for lbl in ax.get_yticklabels():
        lbl.set_fontfamily(ANNOT_FONT["family"])
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily(ANNOT_FONT["family"])

    ax.set_ylim(50, 112); ax.set_yticks([60, 70, 80, 90, 100])
    ax.set_xlim(-0.3, 3.4)

    # legends below the plot (one row each, well separated vertically)
    model_handles = [Line2D([0],[0], marker='o', linestyle='-',
                              color=MODEL_DISPLAY[m], markersize=10,
                              markerfacecolor=MODEL_DISPLAY[m],
                              markeredgecolor='white', markeredgewidth=1.2,
                              label=m, linewidth=2.0)
                      for m in by_model.keys()]
    leg1 = ax.legend(handles=model_handles, loc="upper left",
                     bbox_to_anchor=(0.0, -0.16), ncol=3, frameon=False,
                     handlelength=1.6, handletextpad=0.5, columnspacing=1.4,
                     fontsize=9.6)
    ax.add_artist(leg1)

    size_handles = [Line2D([0],[0], marker='o', linestyle='None',
                              markerfacecolor='0.55', markeredgecolor='white',
                              markeredgewidth=1.0, markersize=v**0.5,
                              label=f"{int(s)}%")
                     for s, v in [(25, msize(25)), (50, msize(50)),
                                   (75, msize(75)), (100, msize(100))]]
    ax.legend(handles=size_handles, loc="upper right",
              bbox_to_anchor=(1.0, -0.16), ncol=4, frameon=False,
              handlelength=0.7, handletextpad=0.4, columnspacing=0.7,
              title="letter-stability rate", title_fontsize=8.8,
              fontsize=8.6)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_model = _load_rows()
    bbox = _load_roi_bbox(EXAMPLE_CASE)

    # Wider figure; left = 2x2 thumbnail grid; right = curve panel.
    fig = plt.figure(figsize=(11.6, 5.4))
    outer = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.30],
                              wspace=0.18,
                              left=0.045, right=0.99, top=0.94, bottom=0.18)
    left_gs = outer[0, 0].subgridspec(2, 2, wspace=0.10, hspace=0.20)
    img_axes = [[fig.add_subplot(left_gs[i, j]) for j in range(2)]
                  for i in range(2)]
    ax_curve = fig.add_subplot(outer[0, 1])

    for k in range(4):
        _draw_thumbnail(img_axes[k // 2][k % 2], k, bbox)
    _render_curve_panel(ax_curve, by_model)

    fig.suptitle(f"Visual-token ablation on {EXAMPLE_CASE}  (blue = ROI;"
                  " red = portion of ROI masked).  "
                  "Right: pilot $n=13$ stratified cases, three API models, "
                  "five self-consistency samples per cell. "
                  "Bold letter above each marker = modal letter on the example case.",
                  x=0.045, y=0.985, ha="left", fontsize=10.2,
                  fontweight="bold", color="#1f2937")

    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{OUT_BASE}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {OUT_BASE}.{{pdf,png,svg}}")


if __name__ == "__main__":
    main()
