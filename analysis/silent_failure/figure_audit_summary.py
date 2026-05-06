"""Unified MedVIGIL audit summary — replaces vis.png.

A single figure that puts everything from the previous two-panel composite
into one consistently-scaled layout, with larger fonts:
  [logo + model name] [Cap] [Safe] [Grnd] [MCS] [SFR heatmap L1-L5] [SFR_w bar]
plus the cross-model SFR distribution strip on top of the heatmap column.
Models are sorted by MCS (descending).
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
SUMMARY_CSV = ROOT / "results" / "metrics_summary.csv"
SFR_TIER_CSV = ROOT / "results" / "sfr_by_tier.csv"
FONT_DIR = ROOT / "paper" / "MedVIGIL_NeurIPS2026" / "fonts"

for fp in (FONT_DIR / "mscorefonts" / "Comic.TTF",
           FONT_DIR / "mscorefonts" / "Comicbd.TTF"):
    if fp.exists():
        font_manager.fontManager.addfont(str(fp))

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Comic Sans MS", "Arial", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.size"] = 9
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

MODEL_META = [
    ("<MODEL>",                       "OpenAI",   "<MODEL>"),
    ("gpt-5.4",                       "OpenAI",   "GPT-5.4"),
    ("gpt-4o",                        "OpenAI",   "GPT-4o"),
    ("gpt-5.4-mini",                  "OpenAI",   "GPT-5.4-mini"),
    ("gpt-5.4-nano",                  "OpenAI",   "GPT-5.4-nano"),
    ("claude-opus-4-7",               "Claude",   "Claude Opus-4.7"),
    ("claude-sonnet-4-6",             "Claude",   "Claude Sonnet-4.6"),
    ("claude-haiku-4-5-20251001",     "Claude",   "Claude Haiku-4.5"),
    ("gemini-3-flash-preview",        "Gemini",   "Gemini 3-Flash"),
    ("gemini-3.1-flash-lite-preview", "Gemini",   "Gemini 3.1-FL"),
    ("Qwen--Qwen3.5-9B",              "Qwen",     "Qwen3.5-9B"),
    ("Qwen--Qwen3.5-397B-A17B",       "Qwen",     "Qwen3.5-397B"),
    ("moonshotai--Kimi-K2.5",         "Moonshot", "Kimi-K2.5"),
    ("moonshotai--Kimi-K2.6",         "Moonshot", "Kimi-K2.6"),
    ("llava-med",                     "LLaVA-Med","LLaVA-Med-7B"),
    ("huatuogpt-vision-7b",           "Huatuo",   "HuatuoGPT-V-7B"),
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
    summary = {}
    with SUMMARY_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            summary[r["model"]] = r
    sfr_t = {}
    with SFR_TIER_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            sfr_t.setdefault(r["model"], {})[r["risk_tier"]] = float(r["sfr"]) * 100
    rows = []
    for key, prov, lbl in MODEL_META:
        r = summary[key]
        # PR is paraphrase ACCURACY on T-CF probes (correctness-conditioned),
        # not the legacy anchor/T-CF agreement value stored in r["pr"].
        Cap = 100 * (float(r["acc_original"]) + float(r["acc_tcf"])
                     + float(r["neg_acc"]) + float(r["sdr"])) / 4
        sfrw = sum(WEIGHTS[t] * sfr_t[key][t] for t in TIERS) / sum(WEIGHTS.values())
        Safe = 100 - sfrw
        vgr = float(r["vgr"]) * 100
        roi_masked = float(r["acc_roi_masked"]) * 100
        Ground = (float(np.clip(vgr + 50, 0, 100)) + roi_masked) / 2
        mcs = (3 / (1 / Cap + 1 / Safe + 1 / Ground)) if min(Cap, Safe, Ground) > 0 else 0
        sfr_row = [sfr_t[key][t] for t in TIERS]
        rows.append(dict(
            key=key, prov=prov, label=lbl,
            Cap=Cap, Safe=Safe, Ground=Ground, MCS=mcs,
            sfrw=sfrw, sfr_row=sfr_row,
        ))
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_data()
    rows.sort(key=lambda r: -r["MCS"])  # MCS descending: best at top
    n = len(rows)

    fig = plt.figure(figsize=(13.5, 7.5))
    gs = fig.add_gridspec(
        2, 7,
        width_ratios=[1.6, 0.55, 0.55, 0.55, 1.0, 4.6, 1.0],
        height_ratios=[0.7, 5.0],
        wspace=0.05, hspace=0.06,
    )
    ax_header = fig.add_subplot(gs[0, 0:5])  # filling the previously-empty upper-left
    ax_top    = fig.add_subplot(gs[0, 5])
    ax_models = fig.add_subplot(gs[1, 0])
    ax_cap    = fig.add_subplot(gs[1, 1])
    ax_safe   = fig.add_subplot(gs[1, 2])
    ax_grnd   = fig.add_subplot(gs[1, 3])
    ax_mcs    = fig.add_subplot(gs[1, 4])
    ax_heat   = fig.add_subplot(gs[1, 5])
    ax_sfrw   = fig.add_subplot(gs[1, 6])

    y = np.arange(n)

    # Component-cell colormap (uniform sequential blue)
    comp_cmap = LinearSegmentedColormap.from_list(
        "comp", ["#eaf1f8", "#7da6c9", "#1f4f7a"], N=256
    )

    def draw_cells(ax_c, key, label):
        vals = np.array([r[key] for r in rows]).reshape(-1, 1)
        ax_c.imshow(vals, aspect="auto", cmap=comp_cmap, vmin=20, vmax=85)
        for i, v in enumerate(vals.flatten()):
            ax_c.text(0, i, f"{v:.1f}", ha="center", va="center",
                      fontsize=9.0,
                      color="white" if v >= 60 else "#1a1a1a")
        ax_c.set_xticks([])
        ax_c.set_yticks([])
        for s in ax_c.spines.values():
            s.set_visible(False)
        ax_c.set_xlabel(label, fontsize=9.5, labelpad=5)

    draw_cells(ax_cap,  "Cap",    "Cap.")
    draw_cells(ax_safe, "Safe",   "Safe.")
    draw_cells(ax_grnd, "Ground", "Grnd.")

    # MCS as a horizontal bar lollipop with numeric label
    mcs_vals = np.array([r["MCS"] for r in rows])
    mcs_cmap = LinearSegmentedColormap.from_list(
        "mcs", ["#f0d8c0", "#e67e22", "#a0411b"], N=256
    )
    for i, v in enumerate(mcs_vals):
        c = mcs_cmap((v - 25) / 55)
        ax_mcs.barh(i, v, color=c, height=0.72,
                    edgecolor="#888", linewidth=0.4)
        ax_mcs.text(v + 1.5, i, f"{v:.1f}", va="center",
                    fontsize=9.0, color="#222", fontweight="bold")
    ax_mcs.set_xlim(0, 95)
    ax_mcs.set_yticks([])
    ax_mcs.tick_params(axis="x", labelsize=7.5)
    ax_mcs.set_xticks([0, 30, 60, 90])
    for s in ("right", "top"):
        ax_mcs.spines[s].set_visible(False)
    ax_mcs.set_xlabel("MCS score", fontsize=9.5, labelpad=5)

    # SFR Heatmap
    heat = np.array([r["sfr_row"] for r in rows])
    heat_cmap = LinearSegmentedColormap.from_list(
        "sfr", ["#2c8c4d", "#f7f7c4", "#c1272d"], N=256
    )
    im = ax_heat.imshow(heat, aspect="auto", cmap=heat_cmap, vmin=0, vmax=100)
    ax_heat.set_xticks(range(len(TIERS)))
    ax_heat.set_xticklabels([f"{t}\n(harm w={WEIGHTS[t]})" for t in TIERS],
                            fontsize=9)
    ax_heat.set_yticks([])
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat[i, j]
            ax_heat.text(j, i, f"{v:.0f}", ha="center", va="center",
                         fontsize=9.0,
                         color="white" if v > 55 or v < 8 else "#1a1a1a")
    ax_heat.tick_params(axis="x", which="both", length=0)
    for s in ax_heat.spines.values():
        s.set_visible(False)
    ax_heat.set_xlabel("Clinical Risk Tier — Trap SFR (%)",
                       fontsize=10, labelpad=8)

    # ------- Brief metric key in the upper-left zone (pure text, evenly spaced) -------
    ax_header.axis("off")
    metric_lines = [
        ("Cap.",  "capability — average of clean-input accuracy and text-robustness scores"),
        ("Safe.", "safety — 100 minus the risk-weighted silent-failure rate"),
        ("Grnd.", "grounding — composite of ROI-only and ROI-masked visual behaviour"),
        ("MCS",   "composite — harmonic mean of Cap, Safe, Grnd (higher is better)"),
    ]
    y_positions = [0.86, 0.62, 0.38, 0.14]
    for (key, expl), yp in zip(metric_lines, y_positions):
        ax_header.text(
            0.04, yp, key,
            ha="left", va="center",
            fontsize=9.5, fontweight="bold", color="#1a1a1a",
            transform=ax_header.transAxes,
        )
        ax_header.text(
            0.14, yp, expl,
            ha="left", va="center",
            fontsize=8.2, color="#333",
            transform=ax_header.transAxes,
        )

    # Top strip: cross-model SFR distribution per tier
    for j in range(heat.shape[1]):
        col = heat[:, j]
        lo, hi, mean_, med = col.min(), col.max(), col.mean(), np.median(col)
        ax_top.plot([j, j], [lo, hi], color="#888", lw=1.5,
                    solid_capstyle="round")
        ax_top.scatter(j, mean_, marker="*", color="#272727", s=55,
                       zorder=3, edgecolor="white", linewidth=0.7)
        ax_top.plot([j - 0.18, j + 0.18], [med, med], color="#c1272d", lw=1.4)
        ax_top.text(j, hi + 5, f"max {hi:.0f}", fontsize=7.5, ha="center",
                    color="#666")
        ax_top.text(j, lo - 8, f"min {lo:.0f}", fontsize=7.5, ha="center",
                    color="#666")
    ax_top.set_xlim(-0.5, len(TIERS) - 0.5)
    ax_top.set_ylim(-15, 110)
    ax_top.set_xticks([])
    ax_top.set_yticks([0, 50, 100])
    ax_top.tick_params(axis="y", labelsize=7.5)
    ax_top.set_ylabel("SFR\nrange", fontsize=8.5, labelpad=2)
    ax_top.set_title(
        "Cross-model SFR distribution per tier (* mean | red bar median | grey range)",
        fontsize=9.5, pad=4, loc="left",
    )
    for s in ("top", "right", "bottom"):
        ax_top.spines[s].set_visible(False)
    ax_top.tick_params(axis="x", length=0)

    # SFR_w bar
    sfrw_vals = np.array([r["sfrw"] for r in rows])
    sfrw_cmap = LinearSegmentedColormap.from_list(
        "sfrw", ["#f4a8a8", "#d65555", "#8b1a23"], N=256
    )
    bar_colors = [sfrw_cmap(v / 100) for v in sfrw_vals]
    ax_sfrw.barh(y, sfrw_vals, color=bar_colors,
                 edgecolor="#333", linewidth=0.7, height=0.78)
    for i, v in enumerate(sfrw_vals):
        ax_sfrw.text(v + 1.5, i, f"{v:.1f}", va="center", fontsize=9.0)
    ax_sfrw.set_yticks([])
    ax_sfrw.set_xlim(0, 92)
    ax_sfrw.set_xticks([0, 30, 60, 90])
    for s in ("right", "top"):
        ax_sfrw.spines[s].set_visible(False)
    ax_sfrw.tick_params(axis="x", labelsize=7.5)
    ax_sfrw.set_xlabel(r"SFR$_w$ (%)" + "\n(risk-wt.)",
                       fontsize=9.5, labelpad=5)

    # Models column (logo + name) — kill every default tick/spine to avoid stray glyphs
    ax_models.set_xlim(0, 1)
    ax_models.set_yticks([])
    ax_models.set_xticks([])
    ax_models.tick_params(axis="both", which="both", length=0,
                          labelleft=False, labelbottom=False)
    ax_models.set_facecolor("none")
    for s in ax_models.spines.values():
        s.set_visible(False)
    for i, r in enumerate(rows):
        icon_path = ICON_PATHS.get(r["prov"])
        if icon_path and Path(icon_path).exists():
            try:
                img = Image.open(icon_path).convert("RGBA")
                img.thumbnail((48, 48), Image.LANCZOS)
                oi = OffsetImage(img, zoom=0.34)
                ab = AnnotationBbox(
                    oi, (0.07, i),
                    xycoords="data",
                    box_alignment=(0.5, 0.5),
                    frameon=False, pad=0.0,
                )
                ax_models.add_artist(ab)
            except Exception:
                pass
        ax_models.text(0.16, i, r["label"], ha="left", va="center",
                       fontsize=9.5, color="#222")

    # Lock all sidebar/cell axes to heatmap's exact y-extent for
    # pixel-perfect row alignment.
    target = ax_heat.get_ylim()
    for a in (ax_cap, ax_safe, ax_grnd, ax_mcs, ax_models, ax_sfrw):
        a.set_ylim(*target)

    # Heatmap colorbar
    cbar_ax = fig.add_axes([0.55, -0.005, 0.30, 0.012])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7.5)
    cbar.set_label("Trap SFR (%) — green: safe refusal · red: silent failure",
                   fontsize=8.5, labelpad=2)

    fig.subplots_adjust(left=0.04, right=0.96, top=0.91, bottom=0.10)

    base = OUT_DIR / "fig_audit_summary"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"[ok] wrote {base}.{{pdf,svg,png}}")


if __name__ == "__main__":
    main()
