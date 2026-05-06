"""Risk-tier SFR heatmap — augmented with cross-model strip + capability sidebar.

The figure shows information not reducible to the tab:risk_tier_results table:
  (a) per-tier cross-model SFR distribution (top strip) — reveals that L4/L5
      are universally hard, not a per-model artifact;
  (b) original-probe accuracy paired alongside trap-SFR (left sidebar) —
      visualises the capability-safety tension within each model;
  (c) sort order by SFR_w on the right SFR_w bar — the safety summary.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper" / "MedVIGIL_NeurIPS2026" / "figures"
SFR_TIER_CSV = ROOT / "results" / "sfr_by_tier.csv"
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

MODEL_META = [
    ("<MODEL>",                   "OpenAI",   "<MODEL>"),
    ("gpt-5.4",                   "OpenAI",   "GPT-5.4"),
    ("gpt-4o",                    "OpenAI",   "GPT-4o"),
    ("gpt-5.4-mini",              "OpenAI",   "GPT-5.4-mini"),
    ("gpt-5.4-nano",              "OpenAI",   "GPT-5.4-nano"),
    ("claude-opus-4-7",           "Claude",   "Claude Opus-4.7"),
    ("claude-sonnet-4-6",         "Claude",   "Claude Sonnet-4.6"),
    ("claude-haiku-4-5-20251001", "Claude",   "Claude Haiku-4.5"),
    ("gemini-3-flash-preview",    "Gemini",   "Gemini 3-Flash"),
    ("gemini-3.1-flash-lite-preview","Gemini","Gemini 3.1-FL"),
    ("Qwen--Qwen3.5-9B",          "Qwen",     "Qwen3.5-9B"),
    ("Qwen--Qwen3.5-397B-A17B",   "Qwen",     "Qwen3.5-397B"),
    ("moonshotai--Kimi-K2.5",     "Moonshot", "Kimi-K2.5"),
    ("moonshotai--Kimi-K2.6",     "Moonshot", "Kimi-K2.6"),
    ("llava-med",                 "LLaVA-Med","LLaVA-Med-7B"),
    ("huatuogpt-vision-7b",       "Huatuo",   "HuatuoGPT-V-7B"),
]
ICON_PATHS = {
    "OpenAI":    ROOT / "paper" / "MedVIGIL_NeurIPS2026" / "icons" / "openai.png",
    "Claude":    ROOT / "paper" / "MedVIGIL_NeurIPS2026" / "icons" / "claude.png",
    "Gemini":    ROOT / "paper" / "MedVIGIL_NeurIPS2026" / "icons" / "gemini.png",
    "Qwen":      ROOT / "paper" / "MedVIGIL_NeurIPS2026" / "icons" / "qwen.png",
    "Moonshot":  ROOT / "paper" / "MedVIGIL_NeurIPS2026" / "icons" / "moonshot.png",
    "LLaVA-Med": ROOT / "paper" / "MedVIGIL_NeurIPS2026" / "icons" / "llava-med.png",
    "Huatuo":    ROOT / "paper" / "MedVIGIL_NeurIPS2026" / "icons" / "huggingface.png",
}
TIERS = ["L1", "L2", "L3", "L4", "L5"]
WEIGHTS = {"L1": 1, "L2": 2, "L3": 3, "L4": 5, "L5": 8}


def load_data():
    sfr = {}
    with SFR_TIER_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            sfr.setdefault(r["model"], {})[r["risk_tier"]] = float(r["sfr"]) * 100
    summary = {}
    with SUMMARY_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            summary[r["model"]] = r
    rows, sfrw_vals, orig_acc, provs = [], [], [], []
    for key, prov, label in MODEL_META:
        vals = [sfr[key].get(t, 0) for t in TIERS]
        rows.append(vals)
        sfrw = sum(WEIGHTS[t] * sfr[key][t] for t in TIERS) / sum(WEIGHTS.values())
        sfrw_vals.append(sfrw)
        orig_acc.append(float(summary[key]["acc_original"]) * 100)
        provs.append(prov)
    return np.array(rows), np.array(sfrw_vals), np.array(orig_acc), provs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sfr_mat, sfrw, orig_acc, provs = load_data()
    labels = [m[2] for m in MODEL_META]

    # Sort: best SFR_w (safest) at top
    order = np.argsort(sfrw)
    sfr_mat = sfr_mat[order]
    sfrw = sfrw[order]
    orig_acc = orig_acc[order]
    labels = [labels[i] for i in order]
    provs = [provs[i] for i in order]
    n = len(labels)

    fig = plt.figure(figsize=(9.4, 6.3))
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1.5, 1.0, 5.0, 1.2],   # [model+icon, orig-acc bar, heatmap, sfrw bar]
        height_ratios=[0.85, 5.0],
        wspace=0.04, hspace=0.04,
    )
    ax_top = fig.add_subplot(gs[0, 2])           # per-tier cross-model strip
    ax_models = fig.add_subplot(gs[1, 0])        # model name + provider logo column
    ax_left = fig.add_subplot(gs[1, 1])          # original-accuracy bar
    ax = fig.add_subplot(gs[1, 2])               # main heatmap
    ax_right = fig.add_subplot(gs[1, 3])         # SFR_w bar

    cmap = LinearSegmentedColormap.from_list(
        "sfr", ["#2c8c4d", "#f7f7c4", "#c1272d"], N=256
    )

    # ------- Main heatmap -------
    im = ax.imshow(sfr_mat, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(len(TIERS)))
    ax.set_xticklabels([f"{t}\n(harm w={WEIGHTS[t]})" for t in TIERS], fontsize=8)
    ax.set_yticks([])
    for i in range(sfr_mat.shape[0]):
        for j in range(sfr_mat.shape[1]):
            v = sfr_mat[i, j]
            txt_color = "white" if v > 55 or v < 8 else "#222222"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=7.6, color=txt_color)
    ax.tick_params(axis="x", which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Clinical Risk Tier", fontsize=9, labelpad=6)

    # ------- Top strip: per-tier cross-model distribution -------
    # For each tier (column), show min-max range as horizontal bar; mean as marker.
    for j in range(sfr_mat.shape[1]):
        col = sfr_mat[:, j]
        lo, hi, mean, med = col.min(), col.max(), col.mean(), np.median(col)
        # Range bar
        ax_top.plot([j, j], [lo, hi], color="#888", lw=1.4, solid_capstyle="round")
        # Mean marker
        ax_top.scatter(j, mean, marker="*", color="#272727", s=42, zorder=3,
                       edgecolor="white", linewidth=0.6)
        # Median bar
        ax_top.plot([j - 0.18, j + 0.18], [med, med], color="#c1272d", lw=1.2)
        # numeric annotations
        ax_top.text(j, hi + 4, f"max {hi:.0f}", fontsize=6, ha="center", color="#666")
        ax_top.text(j, lo - 7, f"min {lo:.0f}", fontsize=6, ha="center", color="#666")
    ax_top.set_xlim(-0.5, len(TIERS) - 0.5)
    ax_top.set_ylim(-15, 110)
    ax_top.set_xticks([])
    ax_top.tick_params(axis="y", labelsize=7)
    ax_top.set_yticks([0, 50, 100])
    ax_top.set_yticklabels(["0", "50", "100"])
    ax_top.set_ylabel("SFR\nrange", fontsize=7.5, labelpad=2)
    ax_top.set_title("Cross-model SFR distribution per tier (* mean | red bar median | grey range)",
                     fontsize=8.5, pad=4, loc="left")
    for s in ("top", "right"):
        ax_top.spines[s].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.tick_params(axis="x", length=0)

    # ------- Left sidebar: original-probe accuracy -------
    y = np.arange(n)
    blue_cmap = LinearSegmentedColormap.from_list(
        "cap_blue", ["#cfd8e5", "#7da6c9", "#1f4f7a"], N=256
    )
    bar_colors_left = [blue_cmap(v / 100.0) for v in orig_acc]
    ax_left.barh(y, orig_acc, color=bar_colors_left, edgecolor="white",
                 linewidth=0.5, height=0.78)
    # Value labels OUTSIDE bars (to the left, away from the heatmap),
    # in dark text on white background — easier to read.
    for i, v in enumerate(orig_acc):
        ax_left.text(v + 2, i, f"{v:.0f}", va="center", ha="left",
                     fontsize=7.2, color="#1f2c3a", fontweight="bold")
    # Hide y-tick labels on the orig-acc bar — model names live in ax_models.
    ax_left.set_yticks(y)
    ax_left.set_yticklabels([""] * n)
    ax_left.tick_params(axis="y", which="both", length=0)

    # Dedicated model+icon column
    ax_models.set_ylim(ax_left.get_ylim())  # placeholder, finalised below
    ax_models.set_xlim(0, 1)
    ax_models.set_yticks(y)
    ax_models.set_yticklabels([""] * n)
    ax_models.tick_params(axis="y", which="both", length=0)
    ax_models.tick_params(axis="x", which="both", length=0, labelsize=0)
    for s in ax_models.spines.values():
        s.set_visible(False)
    for i, (prov, lbl) in enumerate(zip(provs, labels)):
        icon_path = ICON_PATHS.get(prov)
        if icon_path and Path(icon_path).exists():
            try:
                img = Image.open(icon_path).convert("RGBA")
                img.thumbnail((48, 48), Image.LANCZOS)
                oi = OffsetImage(img, zoom=0.30)
                ab = AnnotationBbox(
                    oi, (0.10, i),
                    xycoords="data",
                    box_alignment=(0.5, 0.5),
                    frameon=False, pad=0.0,
                )
                ax_models.add_artist(ab)
            except Exception:
                pass
        ax_models.text(0.20, i, lbl,
                       ha="left", va="center", fontsize=8, color="#222")
    ax_left.set_xlim(95, 0)  # inverted: bar grows from right (zero) toward heatmap
    ax_left.set_xlabel("Original\nacc (%)", fontsize=8, labelpad=2)
    # Heatmap ylim is the ground truth (imshow origin='upper' sets it correctly).
    # Force the three sidebar axes to use the same exact extent so each row
    # aligns pixel-perfectly with its model name, original-accuracy bar, and
    # SFR_w bar.
    ax_left.invert_yaxis()
    ax_models.invert_yaxis()
    target_ylim = ax.get_ylim()
    ax_left.set_ylim(*target_ylim)
    ax_models.set_ylim(*target_ylim)
    ax_left.tick_params(axis="x", labelsize=7)
    for s in ("top", "left", "right"):
        ax_left.spines[s].set_visible(False)
    ax_left.spines["bottom"].set_color("#aaaaaa")

    # ------- Right sidebar: SFR_w (risk-weighted) -------
    bar_colors = [cmap(v / 100) for v in sfrw]
    ax_right.barh(y, sfrw, color=bar_colors, edgecolor="#333",
                  linewidth=0.4, height=0.78)
    for i, v in enumerate(sfrw):
        ax_right.text(v + 1.5, i, f"{v:.1f}", va="center", fontsize=7.0)
    ax_right.set_yticks([])
    ax_right.set_xlim(0, 92)
    ax_right.set_ylim(*target_ylim)
    ax_right.set_xlabel(r"SFR$_w$" + "\n(risk-wt.)", fontsize=8, labelpad=2)
    ax_right.tick_params(axis="x", labelsize=7)
    for s in ("right", "top"):
        ax_right.spines[s].set_visible(False)

    # Suptitle with key insight
    fig.suptitle(
        "Trap silent-failure rate climbs with clinical risk for most models  "
        r"(green = safe refusal · red = silent failure)",
        fontsize=10, y=0.995,
    )

    # Colorbar
    cbar_ax = fig.add_axes([0.30, -0.04, 0.40, 0.012])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=6.5)
    cbar.set_label("Trap SFR (%)", fontsize=7, labelpad=2)

    fig.subplots_adjust(left=0.04, right=0.96, top=0.91, bottom=0.13)

    base = OUT_DIR / "fig_sfr_heatmap"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"[ok] wrote {base}.{{pdf,svg,png}}")


if __name__ == "__main__":
    main()
