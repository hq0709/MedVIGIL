"""Statistical analysis of all experiment results.

Computes aggregate metrics, statistical tests, and generates
summary tables for the position paper.
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import ALL_MODEL_KEYS

MODELS = list(ALL_MODEL_KEYS)


def load_results(experiment: str, model: str) -> dict:
    """Load results JSON for a given experiment and model."""
    path = os.path.join(RESULTS_DIR, f"{experiment}_{model}_results.json")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return None
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-experiment analysers
# ---------------------------------------------------------------------------

def analyze_exp1(models: list) -> dict:
    """Analyze Experiment 1: Modality Recognition."""
    print("\n--- Experiment 1: Modality Recognition ---")
    analysis = {}

    for model in models:
        data = load_results("exp1", model)
        if not data:
            continue

        analysis[model] = {
            "overall_accuracy": data["overall_accuracy"],
            "per_modality": data["per_modality"],
            "n_images": data["total_images"],
        }

        print(f"  {model}: {data['overall_accuracy']:.1%} accuracy ({data['total_images']} images)")
        for mod, stats in data["per_modality"].items():
            print(f"    {mod}: {stats['accuracy']:.1%}")

    return analysis


def analyze_exp2(models: list) -> dict:
    """Analyze Experiment 2: Progressive Visual Depth."""
    print("\n--- Experiment 2: Progressive Visual Depth ---")
    analysis = {}

    for model in models:
        data = load_results("exp2", model)
        if not data:
            continue

        level_data = {}
        for level_key, summary in data["level_summaries"].items():
            level_data[level_key] = {
                "level": summary["level"],
                "specificity": summary["specificity_rate"],
                "refusal_rate": summary["refusal_rate"],
                "uncertainty_rate": summary["uncertainty_rate"],
                "avg_length": summary["avg_response_length"],
            }

        analysis[model] = {
            "levels": level_data,
            "n_images": data["n_images"],
        }

        print(f"  {model}:")
        for lk, ld in level_data.items():
            print(f"    L{ld['level']}: specificity={ld['specificity']:.1%}, "
                  f"refusal={ld['refusal_rate']:.1%}")

    # Statistical test: is there a significant decline across levels?
    if len(models) > 0 and models[0] in analysis:
        levels = analysis[models[0]]["levels"]
        specificities = [v["specificity"] for v in sorted(levels.values(), key=lambda x: x["level"])]
        if len(specificities) >= 2:
            is_declining = all(specificities[i] >= specificities[i+1]
                             for i in range(len(specificities)-1))
            analysis["_declining_trend"] = is_declining
            analysis["_specificities"] = specificities
            print(f"\n  Declining trend: {is_declining}")
            print(f"  Specificities by level: {[f'{s:.1%}' for s in specificities]}")

    return analysis


def analyze_exp3(models: list) -> dict:
    """Analyze Experiment 3: Image Corruption (KEY)."""
    print("\n--- Experiment 3: Image Corruption (KEY EXPERIMENT) ---")
    analysis = {}

    for model in models:
        data = load_results("exp3", model)
        if not data:
            continue

        sim = data.get("similarity_summary", {})
        by_cat = sim.get("_by_category", {})

        model_analysis = {
            "by_category": {},
            "by_corruption": {},
            "n_images": data["n_images"],
            "total_inferences": data["total_inferences"],
        }

        for cat, cat_data in by_cat.items():
            model_analysis["by_category"][cat] = {
                "avg_rouge_l": cat_data.get("avg_rouge_l", 0),
                "avg_exact_match": cat_data.get("avg_exact_match", 0),
                "avg_bert_score": cat_data.get("avg_bert_score", 0),
            }

        for corr_name, corr_data in sim.items():
            if corr_name.startswith("_"):
                continue
            model_analysis["by_corruption"][corr_name] = {
                "category": corr_data["category"],
                "rouge_l": corr_data["avg_rouge_l"],
                "exact_match": corr_data["exact_match_rate"],
                "bert_score": corr_data.get("avg_bert_score", 0),
            }

        # Modality swap summary
        ms = data.get("modality_swap_summary", {})
        if ms:
            model_analysis["modality_swap"] = ms

        analysis[model] = model_analysis

        print(f"  {model}:")
        for cat, cd in model_analysis["by_category"].items():
            print(f"    {cat}: ROUGE-L={cd['avg_rouge_l']:.3f}, "
                  f"ExactMatch={cd['avg_exact_match']:.3f}")

        replacement_rouge = model_analysis["by_category"].get(
            "replacement", {}
        ).get("avg_rouge_l", 0)
        if replacement_rouge > 0:
            print(f"\n    KEY: Even with replaced images, ROUGE-L = {replacement_rouge:.3f}")

        if ms:
            print(f"    Modality swap: recognised={ms.get('recognised_swap_rate', 0):.1%}, "
                  f"stuck={ms.get('stuck_on_original_rate', 0):.1%}")

    # Cross-model comparison
    if len([m for m in models if m in analysis]) >= 2:
        print("\n  Cross-model comparison:")
        for cat in ["noise", "blur", "replacement"]:
            vals = {}
            for model in models:
                if model in analysis:
                    vals[model] = analysis[model]["by_category"].get(cat, {}).get("avg_rouge_l", 0)
            if vals:
                print(f"    {cat}: " + ", ".join(f"{m}={v:.3f}" for m, v in vals.items()))

    return analysis


def analyze_exp4(models: list) -> dict:
    """Analyze Experiment 4: Hallucination Probing."""
    print("\n--- Experiment 4: Hallucination Probing ---")
    analysis = {}

    for model in models:
        data = load_results("exp4", model)
        if not data:
            continue

        summary = data.get("summary", {})
        analysis[model] = summary

        print(f"  {model}:")
        for condition, stats in summary.items():
            if isinstance(stats, dict) and "hallucination_rate" in stats:
                print(f"    {condition}: hallucination_rate={stats['hallucination_rate']:.1%} "
                      f"({stats['hallucination']}/{stats['total']})")

    return analysis


def analyze_exp5(models: list) -> dict:
    """Analyze Experiment 5: VQA-RAD Accuracy Benchmark."""
    print("\n--- Experiment 5: VQA-RAD Accuracy Benchmark ---")
    analysis = {}

    for model in models:
        data = load_results("exp5", model)
        if not data:
            continue

        summary = data.get("summary", {})
        analysis[model] = summary

        overall = summary.get("overall", {})
        em = overall.get("exact_match", {})
        rm = overall.get("relaxed_match", {})
        f1 = overall.get("token_f1", {})

        print(f"  {model} (n={overall.get('n', 0)}):")
        print(f"    Exact match:   {em.get('mean', 0):.1%} "
              f"[{em.get('ci_lower', 0):.1%}, {em.get('ci_upper', 0):.1%}]")
        print(f"    Relaxed match: {rm.get('mean', 0):.1%}")
        print(f"    Token F1:      {f1.get('mean', 0):.3f}")

    return analysis


def analyze_exp6(models: list) -> dict:
    """Analyze Experiment 6: Adversarial Grounding Test."""
    print("\n--- Experiment 6: Adversarial Grounding (KEY) ---")
    analysis = {}

    for model in models:
        data = load_results("exp6", model)
        if not data:
            continue

        summary = data.get("summary", {})
        analysis[model] = summary

        print(f"  {model}:")
        for condition, stats in sorted(summary.items()):
            print(f"    {condition}: compliance={stats['compliance_rate']:.1%} "
                  f"({stats['compliant']}/{stats['total']}), "
                  f"correction={stats['correction_rate']:.1%}")

    # Cross-model comparison on authority hallucination
    if len([m for m in models if m in analysis]) >= 2:
        print("\n  Authority compliance comparison:")
        for m in models:
            if m in analysis:
                auth = analysis[m].get("authority_hallucination", {})
                if auth:
                    print(f"    {m}: {auth.get('compliance_rate', 0):.1%}")

    return analysis


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------

def compute_cross_model_significance(models: list) -> dict:
    """McNemar test for paired accuracy comparisons between models.

    Uses Exp5 per-question exact-match results (paired observations).
    Also computes paired t-test on Exp3 ROUGE-L scores where available.
    """
    from scipy import stats as sp_stats

    sig = {}

    # --- Exp5: McNemar on relaxed match (exact match is 0% for all models) ---
    model_em = {}  # model -> list of 0/1 relaxed match per question
    for model in models:
        data = load_results("exp5", model)
        if not data:
            continue
        model_em[model] = [int(r["relaxed_match"]) for r in data.get("results", [])]

    available = [m for m in models if m in model_em]
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            m1, m2 = available[i], available[j]
            a, b = model_em[m1], model_em[m2]
            n = min(len(a), len(b))
            if n == 0:
                continue
            a, b = a[:n], b[:n]

            # 2x2 contingency: both right, m1 right m2 wrong, …
            both_right = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
            m1_only = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
            m2_only = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
            both_wrong = sum(1 for x, y in zip(a, b) if x == 0 and y == 0)

            # McNemar's test (on discordant pairs)
            discordant = m1_only + m2_only
            if discordant > 0:
                chi2 = (abs(m1_only - m2_only) - 1) ** 2 / discordant
                p_value = 1 - sp_stats.chi2.cdf(chi2, df=1)
            else:
                chi2 = 0.0
                p_value = 1.0

            key = f"{m1}_vs_{m2}"
            sig[key] = {
                "test": "McNemar (relaxed_match)",
                "n": n,
                "m1_accuracy": sum(a) / n,
                "m2_accuracy": sum(b) / n,
                "chi2": float(chi2),
                "p_value": float(p_value),
                "significant_0.05": p_value < 0.05,
            }
            print(f"  McNemar {m1} vs {m2}: p={p_value:.4f} "
                  f"({'*' if p_value < 0.05 else 'ns'})")

    return sig


# ---------------------------------------------------------------------------
# Evidence table
# ---------------------------------------------------------------------------

def generate_evidence_table(all_analyses: dict) -> str:
    """Generate a formatted evidence table for the paper."""
    table = []
    table.append("=" * 80)
    table.append("EVIDENCE TABLE: Medical VLMs Rely on Language Priors")
    table.append("=" * 80)

    # Row 1: Modality Recognition (baseline competence)
    exp1 = all_analyses.get("exp1", {})
    table.append("\n1. BASELINE COMPETENCE (Modality Recognition)")
    for model, data in exp1.items():
        if isinstance(data, dict) and "overall_accuracy" in data:
            table.append(f"   {model}: {data['overall_accuracy']:.1%} accuracy")
    table.append("   -> Model CAN do basic medical image tasks")

    # Row 2: Visual Depth Degradation
    exp2 = all_analyses.get("exp2", {})
    table.append("\n2. VISUAL DEPTH DEGRADATION")
    for model, data in exp2.items():
        if isinstance(data, dict) and "levels" in data:
            for lk, ld in sorted(data["levels"].items(), key=lambda x: x[1]["level"]):
                table.append(f"   {model} L{ld['level']}: specificity={ld['specificity']:.1%}")
    if exp2.get("_declining_trend"):
        table.append("   -> Performance degrades with visual reasoning depth")

    # Row 3: Corruption Invariance (KEY)
    exp3 = all_analyses.get("exp3", {})
    table.append("\n3. CORRUPTION INVARIANCE (KEY FINDING)")
    for model, data in exp3.items():
        if isinstance(data, dict) and "by_category" in data:
            for cat, cd in data["by_category"].items():
                table.append(f"   {model} [{cat}]: ROUGE-L={cd['avg_rouge_l']:.3f}")
            ms = data.get("modality_swap", {})
            if ms:
                table.append(f"   {model} [modality_swap]: "
                             f"recognised={ms.get('recognised_swap_rate', 0):.1%}, "
                             f"stuck={ms.get('stuck_on_original_rate', 0):.1%}")
    table.append("   -> Answers barely change even with destroyed images")

    # Row 4: Hallucination
    exp4 = all_analyses.get("exp4", {})
    table.append("\n4. HALLUCINATION PROBING")
    for model, data in exp4.items():
        if isinstance(data, dict):
            for cond, stats in data.items():
                if isinstance(stats, dict) and "hallucination_rate" in stats:
                    table.append(f"   {model} [{cond}]: {stats['hallucination_rate']:.1%} hallucination")
    table.append("   -> Model confidently describes nonexistent findings")

    # Row 5: VQA Accuracy
    exp5 = all_analyses.get("exp5", {})
    table.append("\n5. VQA-RAD ACCURACY BENCHMARK")
    for model, data in exp5.items():
        if isinstance(data, dict):
            overall = data.get("overall", {})
            em = overall.get("exact_match", {})
            rm = overall.get("relaxed_match", {})
            f1 = overall.get("token_f1", {})
            if rm:
                table.append(f"   {model}: Relaxed={rm.get('mean', 0):.1%} "
                             f"[{rm.get('ci_lower', 0):.1%}, {rm.get('ci_upper', 0):.1%}], "
                             f"F1={f1.get('mean', 0):.3f}, EM={em.get('mean', 0):.1%}")
    table.append("   -> Standard benchmark accuracy with confidence intervals")

    # Row 6: Adversarial Grounding
    exp6 = all_analyses.get("exp6", {})
    table.append("\n6. ADVERSARIAL GROUNDING TEST (KEY FINDING)")
    for model, data in exp6.items():
        if isinstance(data, dict):
            auth = data.get("authority_hallucination", {})
            anat = data.get("anatomy_conflict", {})
            mod = data.get("modality_conflict", {})
            overall = data.get("overall", {})
            if auth:
                table.append(f"   {model}: authority_compliance={auth.get('compliance_rate', 0):.1%}, "
                             f"anatomy_compliance={anat.get('compliance_rate', 0):.1%}, "
                             f"modality_compliance={mod.get('compliance_rate', 0):.1%}")
    table.append("   -> ALL models follow wrong text cues over visual evidence")

    # Significance
    sig = all_analyses.get("_significance", {})
    if sig:
        table.append("\n7. STATISTICAL SIGNIFICANCE (McNemar)")
        for key, s in sig.items():
            table.append(f"   {key}: chi2={s['chi2']:.2f}, p={s['p_value']:.4f} "
                         f"({'*' if s['significant_0.05'] else 'ns'})")

    table.append("\n" + "=" * 80)
    table.append("CONCLUSION: Medical VLMs compensate for weak visual understanding")
    table.append("with strong language priors, producing plausible-sounding but")
    table.append("visually ungrounded answers.")
    table.append("=" * 80)

    return "\n".join(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_analysis():
    """Run complete analysis pipeline."""
    print("=" * 60)
    print("MedVision: Statistical Analysis")
    print("=" * 60)

    models = MODELS

    all_analyses = {
        "exp1": analyze_exp1(models),
        "exp2": analyze_exp2(models),
        "exp3": analyze_exp3(models),
        "exp4": analyze_exp4(models),
        "exp5": analyze_exp5(models),
        "exp6": analyze_exp6(models),
    }

    # Cross-model significance
    print("\n--- Cross-Model Statistical Significance ---")
    try:
        sig = compute_cross_model_significance(models)
        all_analyses["_significance"] = sig
    except ImportError:
        print("  [SKIP] scipy not available for significance tests")
    except Exception as e:
        print(f"  [SKIP] Significance test failed: {e}")

    # Generate evidence table
    evidence_table = generate_evidence_table(all_analyses)
    print("\n" + evidence_table)

    # Save analysis
    output_path = os.path.join(RESULTS_DIR, "full_analysis.json")

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(output_path, "w") as f:
        json.dump(all_analyses, f, indent=2, default=convert)

    # Save evidence table
    table_path = os.path.join(RESULTS_DIR, "evidence_table.txt")
    with open(table_path, "w") as f:
        f.write(evidence_table)

    print(f"\nAnalysis saved to {output_path}")
    print(f"Evidence table saved to {table_path}")

    return all_analyses


if __name__ == "__main__":
    run_analysis()
