"""Capability x Safety x Grounding — twin-panel that traverses the MCS space.

Shows information NOT in the MCS-component bar chart:
  Left  panel: Capability (A) vs Safety (B), point size = Grounding (C)
                MCS isolines (harmonic mean) reveal trade-off geometry.
                Pareto frontier highlighted with a light shaded zone.
  Right panel: Safety (B) vs Grounding (C), point size = Capability (A)
                Diagnoses whether safe-refusing models are also visually grounded.

The two panels together cover all three pairs of MCS components and expose
which models trade off where — a geometry the per-bar component plot cannot show.
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

PROVIDER_COLOR = {
    "OpenAI":   "#10a37f",
    "Claude":   "#d97757",
    "Gemini":   "#4285f4",
    "Qwen":     "#915dff",
    "Moonshot": "#1a73e8",
    "LLaVA-Med": "#a83279",
    "Huatuo":   "#c0392b",
}

MODEL_META = [
    ("<MODEL>", "OpenAI", "5.5"),
    ("gpt-5.4", "OpenAI", "5.4"),
    ("gpt-4o", "OpenAI", "4o"),
    ("gpt-5.4-mini", "OpenAI", "5.4-mini"),
    ("gpt-5.4-nano", "OpenAI", "5.4-nano"),
    ("claude-opus-4-7", "Claude", "Opus-4.7"),
    ("claude-sonnet-4-6", "Claude", "Sonnet-4.6"),
    ("claude-haiku-4-5-20251001", "Claude", "Haiku-4.5"),
    ("gemini-3-flash-preview", "Gemini", "3-Flash"),
    ("gemini-3.1-flash-lite-preview", "Gemini", "3.1-FL"),
    ("Qwen--Qwen3.5-9B", "Qwen", "Qwen3.5-9B"),
    ("Qwen--Qwen3.5-397B-A17B", "Qwen", "Qwen3.5-397B"),
    ("moonshotai--Kimi-K2.5", "Moonshot", "Kimi-K2.5"),
    ("moonshotai--Kimi-K2.6", "Moonshot", "Kimi-K2.6"),
    ("llava-med", "LLaVA-Med", "LLaVA-Med-7B"),
    ("huatuogpt-vision-7b", "Huatuo", "HuatuoGPT-V-7B"),
]
TIERS = ["L1", "L2", "L3", "L4", "L5"]
WEIGHTS = {"L1": 1, "L2": 2, "L3": 3, "L4": 5, "L5": 8}


def load_rows():
    sfr_t = {}
    with SFR_TIER_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            sfr_t.setdefault(r["model"], {})[r["risk_tier"]] = float(r["sfr"]) * 100
    summary = {}
    with SUMMARY_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            summary[r["model"]] = r
    out = []
    for key, prov, lbl in MODEL_META:
        r = summary[key]
        # Capability A uses PR = paraphrase accuracy on T-CF probes (correctness-conditioned).
        A = 100 * (float(r["acc_original"]) + float(r["acc_tcf"]) + float(r["neg_acc"])
                   + float(r["sdr"])) / 4
        sfrw = sum(WEIGHTS[t] * sfr_t[key][t] for t in TIERS) / sum(WEIGHTS.values())
        B = 100 - sfrw
        vgr = float(r["vgr"]) * 100
        roi_masked = float(r["acc_roi_masked"]) * 100
        C = (np.clip(vgr + 50, 0, 100) + roi_masked) / 2
        mcs = (3 / (1 / A + 1 / B + 1 / C)) if min(A, B, C) > 0 else 0
        out.append((lbl, prov, A, B, C, mcs))
    return out


def hm_iso(M, x_grid, third):
    """B such that harmonic_mean(x, B, third) == M, given x_grid for x and fixed third."""
    inv_B = 3 / M - 1 / x_grid - 1 / third
    B = np.where(inv_B > 1e-6, 1 / inv_B, np.nan)
    return np.where((B > 0) & (B <= 100), B, np.nan)


def is_pareto_frontier(xs, ys):
    """Return mask of Pareto-optimal points (maximize both x and y)."""
    n = len(xs)
    is_p = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if xs[j] >= xs[i] and ys[j] >= ys[i] and (xs[j] > xs[i] or ys[j] > ys[i]):
                is_p[i] = False
                break
    return is_p


def draw_panel(ax, rows, x_idx, y_idx, size_idx, x_label, y_label, size_label, title):
    xs = np.array([r[x_idx] for r in rows])
    ys = np.array([r[y_idx] for r in rows])
    sizes_raw = np.array([r[size_idx] for r in rows])
    third = sizes_raw  # fixed third for isolines: use median for clarity
    third_ref = float(np.median(sizes_raw))

    # MCS isolines
    x_grid = np.linspace(15, 100, 600)
    for M in [40, 50, 60, 70]:
        ys_iso = hm_iso(M, x_grid, third_ref)
        ax.plot(x_grid, ys_iso, color="#cccccc", lw=0.6, ls="--", zorder=1)
        idx = np.where(~np.isnan(ys_iso))[0]
        if len(idx):
            xi = idx[-min(20, len(idx))]
            ax.text(x_grid[xi], ys_iso[xi] + 1.0, f"MCS={M}",
                    fontsize=6.4, color="#999", rotation=-32, ha="center")

    # Pareto frontier shading (only for Capability vs Safety panel meaning: maximize both)
    if (x_idx, y_idx) == (2, 3) or (x_idx, y_idx) == (3, 4):
        pareto = is_pareto_frontier(xs, ys)
        # Connect frontier in sorted order
        order = np.argsort(xs[pareto])
        fx = xs[pareto][order]; fy = ys[pareto][order]
        ax.fill_between(np.append(fx, fx[-1]),
                        np.append(fy, 0),
                        0, color="#e8f5ee", alpha=0.45, zorder=0,
                        label="_nolegend_")
        ax.plot(fx, fy, color="#2c8c4d", lw=1.2, alpha=0.85, ls="-",
                zorder=2, label="_nolegend_")

    # Points
    s_min, s_max = sizes_raw.min(), sizes_raw.max()
    point_sizes = 30 + (sizes_raw - s_min) / max(s_max - s_min, 1e-6) * 360
    for (label, prov, A, B, C, mcs), s in zip(rows, point_sizes):
        coord = (rows[0][x_idx], rows[0][y_idx])  # placeholder
        x_v = (A, B, C)[x_idx - 2]
        y_v = (A, B, C)[y_idx - 2]
        color = PROVIDER_COLOR.get(prov, "#666")
        ax.scatter(x_v, y_v, s=s, color=color, edgecolor="white",
                   linewidth=1.0, alpha=0.92, zorder=3)

    # Per-panel label offsets (manual placement; thin grey leader line for far labels)
    if (x_idx, y_idx) == (2, 3):
        # A x B panel
        # Center cluster around (50, 30-42): place in 6 fixed directions
        offsets = {
            "Opus-4.7":      ( 1.6,  1.6),  # NE
            "3.1-FL":        (-9.5,  1.6),  # N (left of point so it doesn't hit Opus)
            "5.4":           ( 1.6,  1.6),  # NE
            "5.5":           ( 1.6, -3.4),  # SE (avoid 3-Flash)
            "3-Flash":       ( 1.6,  1.6),  # NE
            "Sonnet-4.6":    ( 1.8, -0.3),  # E
            "HuatuoGPT-V-7B":(-9.0,  3.2),  # NW with leader
            "LLaVA-Med-7B":  (-23.5,-0.4),  # W with leader
            "5.4-nano":      ( 1.6,  1.6),  # NE — far left, no conflicts
            "Haiku-4.5":     ( 1.6,  3.4),  # N (up of point) — moved away from 5.4-nano area
            "Kimi-K2.6":     ( 1.6,  1.4),  # NE
            "Qwen3.5-397B":  (-22.5, 0.0),  # W with leader
            "4o":            ( 1.8, -3.4),  # SE (away from K2.6)
            "5.4-mini":      ( 1.8, -1.0),  # E
            "Kimi-K2.5":     (-12.0,-3.4),  # SW with leader
            "Qwen3.5-9B":    ( 1.6, -3.6),  # SE
        }
    else:
        # B x C panel — two dense clusters: (35-55, 35-46) and (60-75, 47-58)
        # Strategy: spread labels on a clock-face around each point with leaders.
        offsets = {
            # Top-right cluster
            "Opus-4.7":      ( 4.0,  7.0),   # NE far (top-right corner)
            "3.1-FL":        ( 1.6, -6.5),   # S (below Opus)
            "5.4":           ( 0.0,  4.5),   # N (above point, between 3.1-FL/HuatuoGPT)
            "5.5":           (-7.5,  4.5),   # NW
            "3-Flash":       (-12.4, 5.5),   # NW far (top-left)
            # Mid cluster: Sonnet ↑, LLaVA ↓, Huatuo →
            "HuatuoGPT-V-7B":( 6.0, -3.5),   # SE far (away from 3.1-FL)
            "Sonnet-4.6":    (-0.5,  5.0),   # N (above point)
            "LLaVA-Med-7B":  (-1.0, -7.5),   # S far below
            # Left cluster — spread on a clock face
            "4o":            (-8.0,  4.5),   # NW with leader
            "5.4-mini":      ( 6.2, -3.5),   # SE
            "Haiku-4.5":     ( 7.0, -3.0),   # SE far
            "Kimi-K2.6":     (-7.0, -4.0),   # SW with leader
            "Qwen3.5-397B":  (-2.0, -6.0),   # S far
            "Kimi-K2.5":     (-9.0,  4.0),   # NW with leader
            "Qwen3.5-9B":    (-8.5,  3.0),   # NW with leader
            "5.4-nano":      ( 3.5, -4.5),   # SE
        }
    for (label, prov, A, B, C, mcs) in rows:
        x_v = (A, B, C)[x_idx - 2]
        y_v = (A, B, C)[y_idx - 2]
        dx, dy = offsets.get(label, (1.5, 1.5))
        # Thin grey leader for offsets that move > ~5 units away from point
        if abs(dx) > 6 or abs(dy) > 4:
            ax.plot([x_v, x_v + dx + (1.0 if dx < 0 else 0)],
                    [y_v, y_v + dy * 0.7],
                    color="#bbb", lw=0.4, zorder=3.5, solid_capstyle="round")
        ax.annotate(label, (x_v, y_v), xytext=(x_v + dx, y_v + dy),
                    fontsize=6.8, color="#333", zorder=4)

    ax.set_xlabel(x_label, fontsize=9, labelpad=4)
    ax.set_ylabel(y_label, fontsize=9, labelpad=4)
    ax.set_title(title, fontsize=9.5, pad=6)
    ax.set_xlim(15, 95)
    ax.set_ylim(15, 95)
    ax.grid(lw=0.3, alpha=0.3)
    for s in ("right", "top"):
        ax.spines[s].set_visible(False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.0, 5.2))

    draw_panel(
        ax_l, rows,
        x_idx=2, y_idx=3, size_idx=4,
        x_label="Capability  A = (Original + PR$_{\\mathrm{acc}}$ + NEG + SDR)/4   (%)",
        y_label=r"Safety  B = 100 $-$ SFR$_w$   (%)",
        size_label="Grounding C",
        title="A × B   (size = Grounding C; green band = Pareto frontier)",
    )
    draw_panel(
        ax_r, rows,
        x_idx=3, y_idx=4, size_idx=2,
        x_label=r"Safety  B = 100 $-$ SFR$_w$   (%)",
        y_label="Grounding  C   (%)",
        size_label="Capability A",
        title="B × C   (size = Capability A; green band = Pareto frontier)",
    )

    # Provider legend (shared)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", color=c, markersize=8,
                      markeredgecolor="white", markeredgewidth=1.0, label=k)
               for k, c in PROVIDER_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=7.3,
               bbox_to_anchor=(0.5, -0.02), frameon=False,
               labelspacing=0.4, columnspacing=1.2, handletextpad=0.4)

    fig.tight_layout(rect=[0, 0.03, 1, 1.0])

    base = OUT_DIR / "fig_capability_safety"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"[ok] wrote {base}.{{pdf,svg,png}}")


if __name__ == "__main__":
    main()
