"""Generate Table 1 (SFR), Table 2 (CFHR), Table 3 (SDM) for the paper from
results/silent_failure_v0.json. Emits a LaTeX file ready to \\input{} from
the experiments section.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


MODEL_DISPLAY = {
    "gpt4o": "GPT-4o",
    "claude": "Claude",
    "gemini": "Gemini",
    "llava-med": "LLaVA-Med",
    "llava-general": "LLaVA-1.6",
}

DEG_DISPLAY = {
    "clean": r"$\delta_\text{clean}$",
    "delta_black": r"$\delta_\text{black}$",
    "delta_white": r"$\delta_\text{white}$",
    "delta_noise": r"$\delta_\text{noise}$",
    "delta_swap": r"$\delta_\text{swap}$",
}

DEG_ORDER = ["delta_black", "delta_white", "delta_noise", "delta_swap"]
MODEL_ORDER = ["gpt4o", "claude", "gemini", "llava-med", "llava-general"]


def fmt_pct(x):
    if x is None or (isinstance(x, float) and (x != x)):
        return "--"
    return f"{x*100:.1f}\\%"


def fmt_signed(x):
    if x is None or (isinstance(x, float) and (x != x)):
        return "--"
    sign = "+" if x >= 0 else ""
    return f"${sign}{x:.2f}$"


def render(metric: str, summary: dict, fmt, prompt_label: str = "leading") -> str:
    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{Per-model {metric} on the {prompt_label}-prompt control set "
        f"(reanalysis of existing exp4 generations using the released severity lexicon and refusal judge). "
        f"Numbers are v0 estimates pending the full IIST run.}}\n"
        f"\\label{{tab:{metric.lower()}-v0}}\n"
        "\\begin{tabular}{l" + "c" * len(DEG_ORDER) + "}\n"
        "\\toprule\n"
        "Model & " + " & ".join(DEG_DISPLAY[d] for d in DEG_ORDER) + " \\\\\n"
        "\\midrule\n"
    )
    rows = []
    for m in MODEL_ORDER:
        if m not in summary:
            continue
        cells = [MODEL_DISPLAY[m]]
        for d in DEG_ORDER:
            row = summary[m].get(d, {})
            cells.append(fmt(row.get(metric.lower())))
        rows.append(" & ".join(cells) + " \\\\")
    footer = "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return header + "\n".join(rows) + footer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--leading", default=str(ROOT / "results" / "silent_failure_v0.json"),
                   help="leading-prompt v0 summary (for SFR/CFHR)")
    p.add_argument("--neutral", default=str(ROOT / "results" / "silent_failure_v0_neutral.json"),
                   help="neutral-prompt v0 summary (for SDM)")
    p.add_argument("--out", default=str(ROOT / "paper" / "figures" / "tables_v0.tex"))
    args = p.parse_args()

    with open(args.leading, "r") as f:
        leading = json.load(f)["summary"]
    with open(args.neutral, "r") as f:
        neutral = json.load(f)["summary"]

    out_lines = [
        "% =====================================================================\n"
        "% Auto-generated from results/silent_failure_v0*.json\n"
        "% Regenerate via: python -m analysis.silent_failure.figure_main_table\n"
        "% SFR / CFHR are from the 'leading' prompt variant (which aggressively\n"
        "% probes hallucination on degraded inputs); SDM is from the 'neutral'\n"
        "% variant (which avoids prompting the model with a finding term).\n"
        "% =====================================================================\n\n"
    ]
    out_lines.append(render("SFR", leading, fmt_pct, prompt_label="leading"))
    out_lines.append("\n")
    out_lines.append(render("CFHR", leading, fmt_pct, prompt_label="leading"))
    out_lines.append("\n")
    out_lines.append(render("SDM", neutral, fmt_signed, prompt_label="neutral"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.writelines(out_lines)
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
