"""Evidence-decay curves with text-only language-prior floor.

Information NOT in tab:probe_breakdown:
  - Trajectories from full evidence -> ROI removed -> no image at all.
  - Horizontal "language-prior floor" = DeepSeek text-only baseline mean,
    showing what the LM alone can do at each comparable stage.
  - Highlight inversions where ROI-masked > ROI-only (negative VGR == bad).
  - Shaded "language-prior takeover" region marking the rebound between
    ROI-masked dip and Knowledge-only peak.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper" / "MedVIGIL_NeurIPS2026" / "figures"
SUMMARY_CSV = ROOT / "results" / "metrics_summary.csv"
FONT_DIR = ROOT / "paper" / "MedVIGIL_NeurIPS2026" / "fonts"

for fp in (FONT_DIR / "mscorefonts" / "Comic.TTF",
           FONT_DIR / "mscorefonts" / "Comicbd.TTF"):
    if fp.exists():
        font_manager.fontManager.addfont(str(fp))

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Comic Sans MS", "Arial", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.size"] = 8
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

PROVIDER_COLOR = {
    "OpenAI":   "#10a37f",
    "Claude":   "#d97757",
    "Gemini":   "#4285f4",
    "Qwen":     "#915dff",
    "Moonshot": "#1a73e8",
    "LLaVA-Med": "#a83279",
    "Huatuo":   "#c0392b",
    "DeepSeek": "#444",
}

MODEL_META = [
    ("<MODEL>", "OpenAI", "<MODEL>"),
    ("gpt-5.4", "OpenAI", "GPT-5.4"),
    ("gpt-4o", "OpenAI", "GPT-4o"),
    ("gpt-5.4-mini", "OpenAI", "GPT-5.4-mini"),
    ("gpt-5.4-nano", "OpenAI", "GPT-5.4-nano"),
    ("claude-opus-4-7", "Claude", "Opus-4.7"),
    ("claude-sonnet-4-6", "Claude", "Sonnet-4.6"),
    ("claude-haiku-4-5-20251001", "Claude", "Haiku-4.5"),
    ("gemini-3-flash-preview", "Gemini", "Gemini 3-Flash"),
    ("gemini-3.1-flash-lite-preview", "Gemini", "Gemini 3.1-FL"),
    ("Qwen--Qwen3.5-9B", "Qwen", "Qwen3.5-9B"),
    ("Qwen--Qwen3.5-397B-A17B", "Qwen", "Qwen3.5-397B"),
    ("moonshotai--Kimi-K2.5", "Moonshot", "Kimi-K2.5"),
    ("moonshotai--Kimi-K2.6", "Moonshot", "Kimi-K2.6"),
    ("llava-med", "LLaVA-Med", "LLaVA-Med-7B"),
    ("huatuogpt-vision-7b", "Huatuo", "HuatuoGPT-V-7B"),
]
DEEPSEEK_KEYS = [("deepseek-v4-flash", "DS-v4-flash"),
                 ("deepseek-v4-pro", "DS-v4-pro")]

STAGES = [
    ("acc_original",      "Original",   "image + question"),
    ("acc_tcf",           "TCF",        "+text paraphrase"),
    ("acc_roi_only",      "ROI-only",   "ROI cropped"),
    ("acc_roi_masked",    "ROI-masked", "ROI removed"),
    ("acc_knowledge_only","Knowledge",  "no-image text"),
]


def load_curves():
    summary = {}
    with SUMMARY_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            summary[r["model"]] = r
    out, ds_curves = [], []
    for key, prov, lbl in MODEL_META:
        r = summary[key]
        ys = [100 * float(r[col]) for col, _, _ in STAGES]
        out.append((lbl, prov, ys))
    for key, lbl in DEEPSEEK_KEYS:
        r = summary[key]
        ys = [100 * float(r[col]) for col, _, _ in STAGES]
        ds_curves.append((lbl, ys))
    return out, ds_curves


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curves, ds_curves = load_curves()

    fig = plt.figure(figsize=(11.0, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1.0], wspace=0.22)
    ax = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])

    x = np.arange(len(STAGES))

    # Shaded "language-prior takeover" zone (TCF -> Knowledge transition)
    ax.axvspan(2.5, 4.0, color="#fff4d8", alpha=0.55, zorder=0)
    ax.text(3.25, 102, "language-prior takeover zone",
            fontsize=7.2, color="#aa8800", ha="center", va="top", style="italic")

    # DeepSeek text-only floor: average over the two configs at each stage
    ds_arr = np.array([c[1] for c in ds_curves])
    ds_mean = ds_arr.mean(axis=0)
    ax.fill_between(x, 0, ds_mean, color="#e6e6e6", alpha=0.6, zorder=0,
                    label="text-only LM ceiling (DeepSeek mean)")
    ax.plot(x, ds_mean, color="#444", lw=1.5, ls=":", zorder=2)
    for xi, yi in zip(x, ds_mean):
        ax.text(xi, yi - 4, f"{yi:.0f}", fontsize=6.2, color="#444",
                ha="center")
    ax.text(0.05, ds_mean[0] + 2, "text-only floor",
            fontsize=7, color="#444", style="italic")

    # All vision-capable curves, faint by family
    for lbl, prov, ys in curves:
        c = PROVIDER_COLOR.get(prov, "#888")
        ax.plot(x, ys, color=c, lw=1.0, alpha=0.40, marker="o",
                markersize=3.5, markeredgecolor="white", markeredgewidth=0.4,
                zorder=2)

    # Highlight medical-specific anchors (Huatuo + LLaVA-Med) thick
    for lbl, prov, ys in curves:
        if prov in ("LLaVA-Med", "Huatuo"):
            c = PROVIDER_COLOR[prov]
            ax.plot(x, ys, color=c, lw=2.2, alpha=1.0, marker="o",
                    markersize=5.5, markeredgecolor="white",
                    markeredgewidth=0.8, zorder=5,
                    label=f"{lbl} (medical-specific)")
            ax.text(len(STAGES) - 1 + 0.06, ys[-1], lbl,
                    fontsize=7.0, color=c, va="center", fontweight="bold")

    # Highlight Opus-4.7 (top) and GPT-4o (inversion case) as named exemplars
    spotlight = {"Claude Opus-4.7": "Opus-4.7 (top MCS)",
                 "GPT-4o": "GPT-4o (inverted: ROI-masked > ROI-only)"}
    for lbl, prov, ys in curves:
        if lbl in spotlight:
            c = PROVIDER_COLOR.get(prov, "#888")
            ax.plot(x, ys, color=c, lw=2.0, alpha=1.0, marker="o",
                    markersize=5.0, markeredgecolor="white",
                    markeredgewidth=0.8, zorder=5,
                    label=spotlight[lbl])
            ax.text(len(STAGES) - 1 + 0.06, ys[-1], lbl,
                    fontsize=7.0, color=c, va="center")

    # Mark inversions (ROI-masked > ROI-only) with a red downward arrow at that segment
    inverted_count = 0
    for lbl, prov, ys in curves:
        if ys[3] > ys[2]:  # ROI-masked > ROI-only
            inverted_count += 1
            x_mid = 2.5
            y_mid = (ys[2] + ys[3]) / 2
            ax.annotate("", xy=(3, ys[3]), xytext=(2, ys[2]),
                        arrowprops=dict(arrowstyle="->", color="#c1272d",
                                        lw=0.7, alpha=0.6), zorder=4)
    ax.text(2.5, 4, f"red arrows: ROI-masked > ROI-only inversions (negative VGR) — {inverted_count}/{len(curves)} models",
            fontsize=7.0, color="#c1272d", ha="center", style="italic")

    # X axis with stage labels (sub-label below the main tick)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s[1]}\n({s[2]})" for s in STAGES], fontsize=7.8)
    ax.set_ylim(-2, 105)
    ax.set_xlim(-0.4, len(STAGES) - 1 + 1.2)
    ax.set_ylabel("Accuracy (%)", fontsize=9.5)
    ax.set_title("Evidence-decay trajectories with text-only language-prior reference",
                 fontsize=10.5, pad=8)
    ax.grid(axis="y", lw=0.3, alpha=0.4)

    # Vertical separator at "ROI evidence removed" boundary
    ax.axvline(2.5, color="#bbb", lw=0.6, ls=":", zorder=0)

    # Provider legend
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=1.6, marker="o", markersize=4,
                      markeredgecolor="white", markeredgewidth=0.6, label=k)
               for k, c in PROVIDER_COLOR.items() if k != "DeepSeek"]
    handles.append(Line2D([], [], color="#444", lw=1.6, ls=":",
                          label="text-only DeepSeek mean"))
    ax.legend(handles=handles, loc="upper left", fontsize=6.8, ncol=2,
              labelspacing=0.3, columnspacing=0.8, handletextpad=0.3,
              frameon=False, bbox_to_anchor=(0.0, 0.99))

    # ------- Right panel: 3 archetype curves on a single shared axis -------
    archetypes = [
        ("Capable + grounded (Opus-4.7)", "claude-opus-4-7"),
        ("Strong VGR (Gemini 3.1-FL)", "gemini-3.1-flash-lite-preview"),
        ("Inverted: ROI-masked > ROI-only (GPT-4o)", "gpt-4o"),
        ("Medical 7B (HuatuoGPT-V-7B)", "huatuogpt-vision-7b"),
        ("Medical 7B (LLaVA-Med-7B)", "llava-med"),
    ]
    summary_dict = {key: (lbl, prov, ys) for (key, prov, lbl), (lbl2, prov2, ys)
                    in zip(MODEL_META, curves)}

    ax_r.set_title("Archetype trajectories", fontsize=9.5, pad=6, loc="left")
    # DeepSeek floor on right panel
    ax_r.fill_between(x, 0, ds_mean, color="#e6e6e6", alpha=0.55, zorder=0)
    ax_r.plot(x, ds_mean, color="#444", lw=1.0, ls=":", zorder=2)

    for title, key in archetypes:
        for_lbl, for_prov, ys = summary_dict[key]
        c = PROVIDER_COLOR.get(for_prov, "#888")
        ax_r.plot(x, ys, color=c, lw=1.7, marker="o", markersize=4.5,
                  markeredgecolor="white", markeredgewidth=0.7,
                  zorder=3)
        # Right-end label
        ax_r.text(len(STAGES) - 1 + 0.1, ys[-1], title.split("(")[1].rstrip(")"),
                  fontsize=6.6, color=c, va="center", fontweight="bold")

    # Annotate the inversion at GPT-4o
    gpt4o_ys = summary_dict["gpt-4o"][2]
    ax_r.annotate("inversion",
                  xy=(3, gpt4o_ys[3]), xytext=(2.0, gpt4o_ys[3] + 18),
                  fontsize=6.6, color="#c1272d",
                  arrowprops=dict(arrowstyle="->", color="#c1272d", lw=0.7))

    ax_r.set_xticks(x)
    ax_r.set_xticklabels([s[1] for s in STAGES], fontsize=7.5, rotation=30, ha="right")
    ax_r.set_ylim(-2, 105)
    ax_r.set_xlim(-0.4, len(STAGES) - 1 + 1.6)
    ax_r.set_ylabel("Accuracy (%)", fontsize=8.5)
    ax_r.grid(axis="y", lw=0.3, alpha=0.4)
    for s in ("right", "top"):
        ax_r.spines[s].set_visible(False)

    base = OUT_DIR / "fig_evidence_decay"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"[ok] wrote {base}.{{pdf,svg,png}}")


if __name__ == "__main__":
    main()
