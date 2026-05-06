"""Generate a Nature-style MCS component decomposition figure.

The figure is regenerated from the same aggregate CSV files used by the
NeurIPS draft tables so Figure 1 stays synchronized with the latest results.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper" / "MedVIGIL_NeurIPS2026" / "figures"
SUMMARY_CSV = ROOT / "results" / "metrics_summary.csv"
SFR_TIER_CSV = ROOT / "results" / "sfr_by_tier.csv"
FONT_DIR = ROOT / "paper" / "MedVIGIL_NeurIPS2026" / "fonts"
MS_COMIC_FONT = FONT_DIR / "mscorefonts" / "Comic.TTF"
MS_COMIC_BOLD_FONT = FONT_DIR / "mscorefonts" / "Comicbd.TTF"


# Nature-figure mandatory SVG/font settings.
for font_path in (MS_COMIC_FONT, MS_COMIC_BOLD_FONT):
    if not font_path.exists():
        raise FileNotFoundError(f"Comic Sans MS font file is missing: {font_path}")
    font_manager.fontManager.addfont(str(font_path))

# Register the project-local Comic Sans files and use Comic Sans MS as the
# figure-wide face so exported PNG/PDF/SVG stay visually consistent in LaTeX.
font_manager.findfont(
    font_manager.FontProperties(family="Comic Sans MS"),
    fallback_to_default=False,
)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Comic Sans MS",
    "Arial",
    "DejaVu Sans",
]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.size"] = 8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "neutral_light": "#D8D8D8",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
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
    ("gemini-3.1-flash-lite-preview", "Gemini", "3.1-Flash-Lite"),
    ("Qwen--Qwen3.5-9B", "Qwen", "9B"),
    ("Qwen--Qwen3.5-397B-A17B", "Qwen", "397B-A17B"),
    ("moonshotai--Kimi-K2.5", "Moonshot", "Kimi-K2.5"),
    ("moonshotai--Kimi-K2.6", "Moonshot", "Kimi-K2.6"),
    ("llava-med", "LLaVA-Med", "LLaVA-Med-7B"),
    ("huatuogpt-vision-7b", "Huatuo", "Huatuo-V-7B"),
]


ICON_PATHS = {
    "OpenAI": ROOT / "paper" / "MedVIGIL_NeurIPS2026" / "icons" / "openai.png",
    "Claude": ROOT / "paper" / "MedVIGIL_NeurIPS2026" / "icons" / "claude.png",
    "Gemini": ROOT / "paper" / "MedVIGIL_NeurIPS2026" / "icons" / "gemini.png",
    "Qwen": ROOT / "paper" / "MedVIGIL_NeurIPS2026" / "icons" / "qwen.png",
    "Moonshot": ROOT / "paper" / "MedVIGIL_NeurIPS2026" / "icons" / "moonshot.png",
    "LLaVA-Med": ROOT / "paper" / "MedVIGIL_NeurIPS2026" / "icons" / "llava-med.png",
    "Huatuo": ROOT / "paper" / "MedVIGIL_NeurIPS2026" / "icons" / "huggingface.png",
}


def _read_summary() -> dict[str, dict[str, str]]:
    with SUMMARY_CSV.open(newline="") as f:
        return {row["model"]: row for row in csv.DictReader(f)}


def _read_sfr_by_tier() -> dict[str, dict[str, float]]:
    by_model: dict[str, dict[str, float]] = {}
    with SFR_TIER_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            by_model.setdefault(row["model"], {})[row["risk_tier"]] = float(row["sfr"])
    return by_model


def load_component_rows() -> list[tuple[str, str, float, float, float, float, float, float, float, float, float]]:
    summary = _read_summary()
    sfr_by_tier = _read_sfr_by_tier()
    weights = {"L1": 1, "L2": 2, "L3": 3, "L4": 5, "L5": 8}
    rows: list[tuple[str, str, float, float, float, float, float, float, float, float, float]] = []

    for model_key, provider, display_label in MODEL_META:
        if model_key not in summary:
            raise KeyError(f"Missing model in {SUMMARY_CSV}: {model_key}")
        if model_key not in sfr_by_tier:
            raise KeyError(f"Missing model in {SFR_TIER_CSV}: {model_key}")

        row = summary[model_key]
        cap = 100 * (
            float(row["acc_original"])
            + float(row["pr"])
            + float(row["neg_acc"])
            + float(row["sdr"])
        ) / 4
        sfr_w = (
            sum(weights[tier] * sfr_by_tier[model_key][tier] for tier in weights)
            / sum(weights.values())
        )
        safe = 100 * (1 - sfr_w)
        shifted_vgr = np.clip(100 * float(row["vgr"]) + 50, 0, 100)
        original_acc = 100 * float(row["acc_original"])
        roi_only_acc = 100 * float(row["acc_roi_only"])
        roi_masked_safe = 100 * float(row["acc_roi_masked"])
        ground = (shifted_vgr + roi_masked_safe) / 2
        mcs = 3 * cap * safe * ground / (cap * safe + cap * ground + safe * ground)
        lpl = 100 * float(row["lpl"])
        takeover = lpl * (100 - roi_masked_safe) / 100
        rows.append(
            (
                display_label,
                provider,
                cap,
                safe,
                ground,
                mcs,
                takeover,
                lpl,
                original_acc,
                roi_only_acc,
                roi_masked_safe,
            )
        )

    return rows


def make_figure() -> plt.Figure:
    rows = sorted(load_component_rows(), key=lambda item: item[5], reverse=True)
    labels = [row[0] for row in rows]
    providers = [row[1] for row in rows]
    components = np.array([[row[2], row[3], row[4]] for row in rows], dtype=float)
    mcs = np.array([row[5] for row in rows], dtype=float)
    column_names = ["Cap.", "Safe.", "Ground."]
    cmap = LinearSegmentedColormap.from_list(
        "component_heat",
        ["#F4F5F2", "#DDEBE8", "#9FBED0", "#4D7FA4"],
    )
    n_rows = len(rows)

    icon_cache = {
        provider: np.asarray(Image.open(path).convert("RGBA"))
        for provider, path in ICON_PATHS.items()
    }

    fig = plt.figure(figsize=(4.85, 5.95))
    ax_left = fig.add_axes([0.04, 0.12, 0.94, 0.77])

    ax_left.set_ylim(-1.05, n_rows - 0.35)
    ax_left.invert_yaxis()
    ax_left.set_xlim(0, 6.05)
    ax_left.axis("off")

    x_icon = 0.18
    x_label = 0.43
    x0 = 2.15
    cell_w = 0.72
    cell_h = 0.64
    score_x0 = 4.70
    score_w = 1.08
    score_min = 30
    score_max = 72

    ax_left.text(
        0.02,
        -0.95,
        "MCS components",
        ha="left",
        va="center",
        fontsize=8.7,
        color=PALETTE["neutral_black"],
    )
    for j, name in enumerate(column_names):
        ax_left.text(
            x0 + j * cell_w + cell_w / 2,
            -0.65,
            name,
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=PALETTE["neutral_dark"],
        )
    ax_left.text(
        score_x0 + score_w / 2,
        -0.65,
        "MCS score",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color=PALETTE["neutral_dark"],
    )

    def score_to_x(value: float) -> float:
        frac = np.clip((value - score_min) / (score_max - score_min), 0, 1)
        return score_x0 + frac * score_w

    for tick in [30, 50, 70]:
        x_tick = score_to_x(tick)
        ax_left.plot(
            [x_tick, x_tick],
            [-0.28, n_rows - 0.35],
            color="#E4E7E7",
            linewidth=0.7,
            zorder=0,
        )
        ax_left.text(
            x_tick,
            n_rows - 0.04,
            f"{tick}",
            ha="center",
            va="top",
            fontsize=7.0,
            color=PALETTE["neutral_mid"],
        )
    ax_left.plot(
        [score_x0, score_x0 + score_w],
        [n_rows - 0.20, n_rows - 0.20],
        color=PALETTE["neutral_dark"],
        linewidth=0.8,
    )

    for i, (label, provider) in enumerate(zip(labels, providers), start=1):
        y = i - 1

        # Normalize heterogeneous logo PNGs to the same visual height.
        imagebox = OffsetImage(icon_cache[provider], zoom=11.0 / icon_cache[provider].shape[0])
        ab = AnnotationBbox(
            imagebox,
            (x_icon, y),
            frameon=False,
            box_alignment=(0.5, 0.5),
            pad=0,
        )
        ax_left.add_artist(ab)
        ax_left.text(
            x_label,
            y,
            label,
            ha="left",
            va="center",
            fontsize=7.8,
            color=PALETTE["neutral_black"],
        )

        for j in range(components.shape[1]):
            value = components[i - 1, j]
            x = x0 + j * cell_w
            normalized = np.clip((value - 25) / 60, 0, 1)
            ax_left.add_patch(
                Rectangle(
                    (x, y - cell_h / 2),
                    cell_w,
                    cell_h,
                    facecolor=cmap(normalized),
                    edgecolor="white",
                    linewidth=0.55,
                )
            )
            ax_left.text(
                x + cell_w / 2,
                y,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7.6,
                color=PALETTE["neutral_black"],
            )

        mcs_value = mcs[i - 1]
        x_score = score_to_x(mcs_value)
        ax_left.plot(
            [score_x0, score_x0 + score_w],
            [y, y],
            color="#EDF0F0",
            linewidth=6.6,
            solid_capstyle="round",
            zorder=0.8,
        )
        ax_left.plot(
            [score_x0, x_score],
            [y + 0.035, y + 0.035],
            color="#111111",
            alpha=0.16,
            linewidth=7.0,
            solid_capstyle="round",
            zorder=1.0,
        )
        ax_left.plot(
            [score_x0, x_score],
            [y, y],
            color="#5F6B73",
            alpha=0.55,
            linewidth=2.7,
            solid_capstyle="round",
            zorder=1.2,
        )
        ax_left.scatter(
            [x_score],
            [y],
            s=24,
            color="#272727",
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
        ax_left.text(
            score_x0 + score_w + 0.13,
            y,
            f"{mcs_value:.1f}",
            ha="left",
            va="center",
            fontsize=7.7,
            color=PALETTE["neutral_black"],
            zorder=2,
        )

    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    for suffix in ("svg", "pdf", "png"):
        path = OUT_DIR / f"fig_mcs_components.{suffix}"
        save_kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            save_kwargs["dpi"] = 450
        fig.savefig(path, **save_kwargs)
        print(f"[ok] wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
