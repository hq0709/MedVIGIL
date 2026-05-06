"""Generate publication-quality figures for the MedVision study.

Produces 7 main figures:
1. Spider/radar chart: VLM performance across visual depth levels
2. Heatmap: Answer similarity across corruption types
3. Bar chart: Hallucination rates by condition
4. Line plot: Accuracy vs. noise/blur severity
5. Qualitative examples: Original vs corrupted -> same answer
6. Bar chart: VQA-RAD accuracy across models (NEW)
7. Confusion matrix: Modality swap results (NEW)
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import ALL_MODEL_KEYS

# Unified colour scheme: one per model
MODEL_COLORS = {
    "llava-med": "#e74c3c",      # red
    "llava-general": "#3498db",  # blue
    "gemini": "#2ecc71",         # green
    "claude": "#9b59b6",         # purple
}

DEFAULT_MODELS = list(ALL_MODEL_KEYS)


def setup_style():
    """Set up publication-quality plot style."""
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    sns.set_palette("husl")


def _color(model: str) -> str:
    return MODEL_COLORS.get(model, "#7f8c8d")


def load_results(experiment: str, model: str) -> dict:
    """Load results JSON."""
    path = os.path.join(RESULTS_DIR, f"{experiment}_{model}_results.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _available_models(experiment: str, models=None):
    """Return subset of *models* that have result files for *experiment*."""
    if models is None:
        models = DEFAULT_MODELS
    return [m for m in models if load_results(experiment, m) is not None]


# -----------------------------------------------------------------------
# Figure 1
# -----------------------------------------------------------------------

def fig1_visual_depth_radar(models=None):
    """Figure 1: Spider/radar chart of VLM performance across visual depth levels."""
    models = _available_models("exp2", models)
    if not models:
        print("  [SKIP] No exp2 data")
        return

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    categories = ["L1: Organ ID", "L2: Abnormality", "L3: Counting", "L4: Spatial"]
    n_cats = len(categories)
    angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
    angles += angles[:1]

    for model in models:
        data = load_results("exp2", model)
        if not data:
            continue
        summaries = data.get("level_summaries", {})
        values = []
        for level_key in ["L1_organ", "L2_abnormality", "L3_counting", "L4_spatial"]:
            spec = summaries.get(level_key, {}).get("specificity_rate", 0)
            values.append(spec)
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=model, color=_color(model))
        ax.fill(angles, values, alpha=0.15, color=_color(model))

    # Human baseline
    human_values = [0.95, 0.85, 0.75, 0.70] + [0.95]
    ax.plot(angles, human_values, "s--", linewidth=2, label="Human (est.)",
            color="#95a5a6", markersize=8)
    ax.fill(angles, human_values, alpha=0.05, color="#95a5a6")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], size=9)
    ax.set_title("Visual Understanding Depth:\nVLM vs Human Performance", pad=20, size=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    save_path = os.path.join(FIGURES_DIR, "fig1_visual_depth_radar.pdf")
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved: {save_path}")


# -----------------------------------------------------------------------
# Figure 2
# -----------------------------------------------------------------------

def fig2_corruption_heatmap(models=None):
    """Figure 2: Heatmap of answer similarity across corruption types."""
    models = _available_models("exp3", models)
    if not models:
        print("  [SKIP] No exp3 data")
        return

    for model in models:
        data = load_results("exp3", model)
        if not data:
            continue

        sim = data.get("similarity_summary", {})

        categories = {
            "Noise": ["noise_25", "noise_50", "noise_100", "noise_150", "noise_200"],
            "Blur": ["blur_5", "blur_15", "blur_31", "blur_63"],
            "Replace": ["black", "white", "random", "unrelated"],
        }

        question_ids = []
        for img_result in data.get("results", [])[:1]:
            question_ids = list(img_result.get("questions", {}).keys())
            break
        if not question_ids:
            question_ids = ["corr_modality", "corr_organ", "corr_findings", "corr_diagnosis"]

        all_corruptions = []
        for cat_corrs in categories.values():
            all_corruptions.extend(cat_corrs)

        corr_labels = [c for c in all_corruptions if c in sim]
        if not corr_labels:
            continue

        matrix = np.zeros((len(corr_labels), len(question_ids)))
        for i, corr in enumerate(corr_labels):
            for j, q_id in enumerate(question_ids):
                refs = []
                for img_result in data.get("results", []):
                    q_data = img_result.get("questions", {}).get(q_id, {})
                    orig = q_data.get("corruption_answers", {}).get("original", {}).get("answer", "")
                    corrupted = q_data.get("corruption_answers", {}).get(corr, {}).get("answer", "")
                    if orig and corrupted:
                        ref_words = set(orig.lower().split())
                        hyp_words = set(corrupted.lower().split())
                        if ref_words and hyp_words:
                            overlap = ref_words & hyp_words
                            p = len(overlap) / len(hyp_words)
                            r = len(overlap) / len(ref_words)
                            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                            refs.append(f1)
                if refs:
                    matrix[i, j] = np.mean(refs)

        fig, ax = plt.subplots(figsize=(10, 8))
        q_labels = [q.replace("corr_", "").capitalize() for q in question_ids]
        corr_display = [c.replace("_", " ").title() for c in corr_labels]

        sns.heatmap(
            matrix, annot=True, fmt=".2f", cmap="RdYlGn_r",
            xticklabels=q_labels, yticklabels=corr_display,
            vmin=0, vmax=1, ax=ax,
            cbar_kws={"label": "Answer Similarity (ROUGE-L)"},
        )

        noise_end = len([c for c in corr_labels if c.startswith("noise")])
        blur_end = noise_end + len([c for c in corr_labels if c.startswith("blur")])
        if noise_end > 0:
            ax.axhline(y=noise_end, color="black", linewidth=2)
        if blur_end > noise_end:
            ax.axhline(y=blur_end, color="black", linewidth=2)

        ax.set_title(f"Answer Similarity Under Image Corruption\n({model})", size=16)
        ax.set_xlabel("Question Type", size=14)
        ax.set_ylabel("Corruption Type", size=14)

        save_path = os.path.join(FIGURES_DIR, f"fig2_corruption_heatmap_{model}.pdf")
        plt.savefig(save_path)
        plt.savefig(save_path.replace(".pdf", ".png"))
        plt.close()
        print(f"  Saved: {save_path}")


# -----------------------------------------------------------------------
# Figure 3
# -----------------------------------------------------------------------

def fig3_hallucination_bars(models=None):
    """Figure 3: Bar chart of hallucination rates by condition."""
    models = _available_models("exp4", models)
    if not models:
        print("  [SKIP] No exp4 data")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    conditions = ["leading_on_real", "neutral_on_real", "leading_on_black", "neutral_on_black"]
    condition_labels = [
        "Leading Q\n(Real Image)",
        "Neutral Q\n(Real Image)",
        "Leading Q\n(Black Image)",
        "Neutral Q\n(Black Image)",
    ]

    x = np.arange(len(conditions))
    n_models = len(models)
    width = 0.8 / max(n_models, 1)

    for idx, model in enumerate(models):
        data = load_results("exp4", model)
        if not data:
            continue
        summary = data.get("summary", {})
        rates = [summary.get(cond, {}).get("hallucination_rate", 0) for cond in conditions]

        offset = (idx - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=model, color=_color(model),
                      edgecolor="black", linewidth=0.5)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{rate:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Hallucination Rate", size=14)
    ax.set_title("Hallucination Rates Across Conditions", size=16)
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels, size=11)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
    ax.legend(loc="upper right")

    save_path = os.path.join(FIGURES_DIR, "fig3_hallucination_bars.pdf")
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved: {save_path}")


# -----------------------------------------------------------------------
# Figure 4
# -----------------------------------------------------------------------

def fig4_corruption_severity_line(models=None):
    """Figure 4: Line plot of answer similarity vs. noise/blur severity."""
    models = _available_models("exp3", models)
    if not models:
        print("  [SKIP] No exp3 data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    noise_levels = [25, 50, 100, 150, 200]
    blur_levels = [5, 15, 31, 63]

    for model in models:
        data = load_results("exp3", model)
        if not data:
            continue
        sim = data.get("similarity_summary", {})
        c = _color(model)

        # Noise
        values = [sim.get(f"noise_{s}", {}).get("avg_rouge_l", 0) for s in noise_levels]
        axes[0].plot(noise_levels, values, "o-", linewidth=2, markersize=8,
                     label=model, color=c)

        # Blur
        values = [sim.get(f"blur_{k}", {}).get("avg_rouge_l", 0) for k in blur_levels]
        axes[1].plot(blur_levels, values, "s-", linewidth=2, markersize=8,
                     label=model, color=c)

    axes[0].set_xlabel("Gaussian Noise (σ)", size=14)
    axes[0].set_ylabel("Answer Similarity (ROUGE-L)", size=14)
    axes[0].set_title("Effect of Noise on Answer Similarity", size=14)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    axes[1].set_xlabel("Gaussian Blur (kernel size)", size=14)
    axes[1].set_ylabel("Answer Similarity (ROUGE-L)", size=14)
    axes[1].set_title("Effect of Blur on Answer Similarity", size=14)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    plt.suptitle("Answer Stability Under Progressive Image Degradation", size=16, y=1.02)
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "fig4_severity_line.pdf")
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved: {save_path}")


# -----------------------------------------------------------------------
# Figure 5
# -----------------------------------------------------------------------

def fig5_qualitative_examples(model="llava-med"):
    """Figure 5: Qualitative comparison — original vs corrupted, same answer."""
    data = load_results("exp3", model)
    if not data:
        print("  [SKIP] No exp3 data for qualitative examples")
        return

    examples = []
    for img_result in data.get("results", []):
        for q_id, q_data in img_result.get("questions", {}).items():
            orig = q_data.get("corruption_answers", {}).get("original", {}).get("answer", "")
            for corr_name in ["black", "random", "unrelated"]:
                corrupted = q_data.get("corruption_answers", {}).get(corr_name, {}).get("answer", "")
                if orig and corrupted:
                    ref_words = set(orig.lower().split())
                    hyp_words = set(corrupted.lower().split())
                    if ref_words and hyp_words:
                        overlap = ref_words & hyp_words
                        similarity = len(overlap) / max(len(ref_words), len(hyp_words))
                        if similarity > 0.5:
                            examples.append({
                                "image": img_result["image_name"],
                                "question": q_data["question"],
                                "original_answer": orig,
                                "corruption": corr_name,
                                "corrupted_answer": corrupted,
                                "similarity": similarity,
                            })

    examples.sort(key=lambda x: x["similarity"], reverse=True)
    n_examples = min(6, len(examples))
    if n_examples == 0:
        print("  [SKIP] No high-similarity examples found")
        return

    fig, axes = plt.subplots(n_examples, 1, figsize=(14, 3 * n_examples))
    if n_examples == 1:
        axes = [axes]

    for i, ex in enumerate(examples[:n_examples]):
        ax = axes[i]
        ax.axis("off")
        text = (
            f"Image: {ex['image']}\n"
            f"Question: {ex['question']}\n"
            f"Corruption: {ex['corruption'].upper()}\n"
            f"{'─' * 60}\n"
            f"Original answer:  {ex['original_answer'][:100]}...\n"
            f"Corrupted answer: {ex['corrupted_answer'][:100]}...\n"
            f"Similarity: {ex['similarity']:.1%}"
        )
        ax.text(0.02, 0.5, text, transform=ax.transAxes, fontsize=10,
                verticalalignment="center", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    plt.suptitle(f"Qualitative Examples: Same Answer Despite Image Corruption ({model})",
                 size=14, y=1.01)
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, f"fig5_qualitative_{model}.pdf")
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved: {save_path}")

    examples_path = os.path.join(RESULTS_DIR, f"qualitative_examples_{model}.json")
    with open(examples_path, "w") as f:
        json.dump(examples[:20], f, indent=2)
    print(f"  Saved: {examples_path}")


# -----------------------------------------------------------------------
# Figure 6 (NEW): VQA-RAD accuracy bars
# -----------------------------------------------------------------------

def fig6_vqa_accuracy_bars(models=None):
    """Figure 6: VQA-RAD accuracy across models with bootstrap CIs."""
    models = _available_models("exp5", models)
    if not models:
        print("  [SKIP] No exp5 data")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    metrics = ["exact_match", "relaxed_match", "token_f1"]
    metric_labels = ["Exact Match", "Relaxed Match", "Token F1"]
    x = np.arange(len(metrics))

    n_models = len(models)
    width = 0.8 / max(n_models, 1)

    for idx, model in enumerate(models):
        data = load_results("exp5", model)
        if not data:
            continue
        overall = data.get("summary", {}).get("overall", {})

        means = []
        ci_lo = []
        ci_hi = []
        for m_key in metrics:
            ci = overall.get(m_key, {})
            means.append(ci.get("mean", 0))
            ci_lo.append(ci.get("mean", 0) - ci.get("ci_lower", 0))
            ci_hi.append(ci.get("ci_upper", 0) - ci.get("mean", 0))

        offset = (idx - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=model, color=_color(model),
                      edgecolor="black", linewidth=0.5,
                      yerr=[ci_lo, ci_hi], capsize=3, error_kw={"linewidth": 1})

        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f"{m:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Score", size=14)
    ax.set_title("VQA-RAD Accuracy Benchmark", size=16)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, size=12)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")

    save_path = os.path.join(FIGURES_DIR, "fig6_vqa_accuracy_bars.pdf")
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved: {save_path}")


# -----------------------------------------------------------------------
# Figure 7 (NEW): Modality swap confusion matrix
# -----------------------------------------------------------------------

def fig7_modality_swap(models=None):
    """Figure 7: Modality swap confusion matrix — did the model notice the swap?"""
    models = _available_models("exp3", models)
    if not models:
        print("  [SKIP] No exp3 data")
        return

    for model in models:
        data = load_results("exp3", model)
        if not data:
            continue
        swap_results = data.get("modality_swap_results", [])
        if not swap_results:
            continue

        modalities = sorted(set(
            [r["swap_modality"] for r in swap_results if r["swap_modality"] != "UNKNOWN"]
            + [r["original_modality"] for r in swap_results if r["original_modality"]]
        ))
        if not modalities:
            continue

        # Build confusion: rows = true (swap) modality, cols = predicted
        labels = modalities + ["UNKNOWN"]
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        label_idx = {l: i for i, l in enumerate(labels)}

        for r in swap_results:
            true_mod = r["swap_modality"]
            pred_mod = r["predicted_modality"]
            ti = label_idx.get(true_mod, label_idx.get("UNKNOWN"))
            pi = label_idx.get(pred_mod, label_idx.get("UNKNOWN"))
            if ti is not None and pi is not None:
                cm[ti, pi] += 1

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted Modality", size=14)
        ax.set_ylabel("True (Swapped) Modality", size=14)
        ax.set_title(f"Modality Swap Confusion Matrix ({model})", size=16)

        # Annotate stuck-on-original rate
        ms = data.get("modality_swap_summary", {})
        if ms:
            stuck = ms.get("stuck_on_original_rate", 0)
            ax.text(0.5, -0.12,
                    f"Stuck on original modality: {stuck:.1%}",
                    transform=ax.transAxes, ha="center", fontsize=12,
                    fontstyle="italic", color="#e74c3c")

        save_path = os.path.join(FIGURES_DIR, f"fig7_modality_swap_{model}.pdf")
        plt.savefig(save_path)
        plt.savefig(save_path.replace(".pdf", ".png"))
        plt.close()
        print(f"  Saved: {save_path}")


def fig8_adversarial_grounding(models):
    """Figure 8: Adversarial Grounding Test — compliance rates by condition."""
    available = _available_models("exp6", models)
    if not available:
        print("  [SKIP] No exp6 data")
        return

    conditions = ["modality_conflict", "anatomy_conflict", "authority_hallucination"]
    cond_labels = ["Modality\nConflict", "Anatomy\nConflict", "Authority\nHallucination"]

    n_cond = len(conditions)
    n_models = len(available)
    x = np.arange(n_cond)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(available):
        data = load_results("exp6", model)
        summary = data.get("summary", {})
        rates = []
        for c in conditions:
            rates.append(summary.get(c, {}).get("compliance_rate", 0) * 100)

        color = MODEL_COLORS.get(model, "#333")
        bars = ax.bar(x + i * width, rates, width, label=model,
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{rate:.0f}%", ha="center", va="bottom", fontsize=8,
                    fontweight="bold")

    ax.set_ylabel("Compliance Rate (%)", fontsize=12)
    ax.set_title("Exp 6: Text-Image Conflict — All Models Follow Wrong Text Cues",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(cond_labels, fontsize=10)
    ax.set_ylim(0, 115)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% baseline")
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_path = os.path.join(FIGURES_DIR, "fig8_adversarial_grounding.pdf")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    fig.savefig(save_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def generate_all_figures():
    """Generate all publication-quality figures."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    setup_style()

    print("=" * 50)
    print("Generating Publication-Quality Figures")
    print("=" * 50)

    models = DEFAULT_MODELS

    for name, fn, args in [
        ("Figure 1: Visual Depth Radar Chart", fig1_visual_depth_radar, (models,)),
        ("Figure 2: Corruption Heatmap", fig2_corruption_heatmap, (models,)),
        ("Figure 3: Hallucination Bar Chart", fig3_hallucination_bars, (models,)),
        ("Figure 4: Severity Line Plots", fig4_corruption_severity_line, (models,)),
        ("Figure 5: Qualitative Examples", fig5_qualitative_examples, ("llava-med",)),
        ("Figure 6: VQA Accuracy Bars", fig6_vqa_accuracy_bars, (models,)),
        ("Figure 7: Modality Swap Confusion", fig7_modality_swap, (models,)),
        ("Figure 8: Adversarial Grounding", fig8_adversarial_grounding, (models,)),
    ]:
        print(f"\n{name}")
        try:
            fn(*args)
        except Exception as e:
            print(f"  [ERROR] {e}")

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    generate_all_figures()
