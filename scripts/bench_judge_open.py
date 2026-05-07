"""LLM-judge scorer for open-ended (probes_open.csv) model outputs.

Takes a JSONL of model responses and emits per-row verdicts:
  correct                      model's free-form answer matches gold
  incorrect                    model gave a clear answer that differs from gold
  refused_appropriately        model refused on a probe where refusal is correct
                               (halluc_trap or roi_masked when grounded)
  refused_inappropriately      model refused on a probe where it should answer
  silent_failure               model confidently answered a halluc_trap (bought into
                               false premise) OR confidently answered roi_masked
                               (guessed-from-prior despite occluded ROI)
  hedged                       model produced a non-committal/uncertain answer
                               that doesn't clearly match or refuse

Pipeline (per row):
  Step 1 — fast heuristics (regex):
    REFUSAL_RX matches "I cannot determine", "unable to", "insufficient
    information", etc. -> mark as refused (and post-process by probe_kind).
    GOLD_MATCH_RX checks substring containment of gold or any
    answer_variant against the model's answer (case-insensitive).
    For yes/no probes, an exact yes/no extractor.
  Step 2 — LLM-judge fallback (<MODEL>):
    For ambiguous rows (no clear refusal, no clear gold match) call
    <MODEL> with a short structured prompt. Returns verdict + confidence.

Input JSONL schema (one row per probe response):
  {
    "probe_id": "MVB-0001__orig",
    "case_id":  "MVB-0001",
    "probe_kind": "original",
    "question": "...",
    "model_answer": "...",
    "model_id": "gpt-4o",
    [optional] "expected_answer_or_set": "no",  # if not given we look it up
    [optional] "halluc_explanation": "...",     # for traps
  }

Output JSONL: same keys + verdict + judge_confidence + judge_reason +
              judge_method ("heuristic" | "llm" | "rule").

Run:
    python3 scripts/bench_judge_open.py \
        --input results/<model>.jsonl \
        --output results/<model>__judged.jsonl \
        --concurrency 8

Resume-safe: if `--output` already has rows, those probe_ids are skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api_models import _load_dotenv  # noqa: E402

_load_dotenv()

BENCH = ROOT / "data/medvigil_v1"
PROBES_OPEN = BENCH / "probes_open.csv"

JUDGE_MODEL = "<MODEL>"
JUDGE_MAX_TOKENS = 600


# ---------------------------------------------------------------------------
# Heuristic patterns
# ---------------------------------------------------------------------------

REFUSAL_PATTERNS = [
    r"\bi (?:cannot|can'?t|am unable to|am not able to)\b",
    r"\bunable to (?:determine|tell|conclude|answer)",
    r"\bcannot (?:determine|tell|conclude|answer|be (?:determined|established))",
    r"\binsufficient (?:information|evidence|data)",
    r"\bnot enough (?:information|evidence|context)",
    r"\bwithout (?:additional|more|further) (?:information|context|imaging)",
    r"\bunclear (?:from|on) (?:this|the) image",
    r"\bnot (?:visible|discernible|evident|apparent) (?:in|on) this",
    r"\bi'?m not (?:sure|certain|able)",
    r"\bdifficult to (?:determine|conclude|tell|say)",
    r"\bnot enough detail",
    r"\bplease consult",
    r"\bprovide more information",
    r"\brequires? (?:more|additional|further)",
    r"\bI don'?t have enough",
]
REFUSAL_RX = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)

# false-premise rejection (specific to traps): models that DO recognize the
# false premise often phrase it like "the image does not show ...", "no
# evidence of ...", "not present", "premise is incorrect", etc.
PREMISE_REJECT_PATTERNS = [
    r"\bno (?:evidence|sign|finding|indication) of",
    r"\b(?:does not|doesn'?t) (?:show|appear|demonstrate|reveal)",
    r"\bnot (?:present|visible|seen|observed|evident)",
    r"\bpremise (?:of (?:the |this )?question )?(?:is|appears) (?:incorrect|wrong|flawed|false)",
    r"\bthere (?:is|are) no",
    r"\bnone (?:of these|are present)",
    r"\bcontrary to",
    r"\bappears? (?:normal|unremarkable)",
    r"\bno acute",
]
PREMISE_REJECT_RX = re.compile("|".join(PREMISE_REJECT_PATTERNS), re.IGNORECASE)

YES_RX = re.compile(r"\b(yes|yeah|affirmative|correct|indeed|positive|present)\b", re.IGNORECASE)
NO_RX = re.compile(r"\b(no|nope|negative|absent|not present)\b", re.IGNORECASE)


def normalize(s: str) -> str:
    s = (s or "").lower()
    return re.sub(r"\s+", " ", s).strip().rstrip("?.! ")


# ---------------------------------------------------------------------------
# Per-probe-kind judging logic
# ---------------------------------------------------------------------------

def heuristic_judge(probe: dict) -> dict | None:
    """Returns a verdict dict if heuristics are confident, else None."""
    answer = (probe.get("model_answer") or "").strip()
    if not answer:
        return {
            "verdict": "incorrect",
            "judge_confidence": 1.0,
            "judge_reason": "empty model answer",
            "judge_method": "rule",
        }

    kind = probe["probe_kind"]
    norm_ans = normalize(answer)
    gold = probe.get("expected_answer_or_set", "") or ""
    gold_norm = normalize(gold)

    refusal_hit = bool(REFUSAL_RX.search(answer))
    premise_reject_hit = bool(PREMISE_REJECT_RX.search(answer))

    # ----- halluc_trap -----
    if kind == "halluc_trap":
        # A model that explicitly rejects the false premise is correct.
        if refusal_hit or premise_reject_hit:
            return {
                "verdict": "refused_appropriately",
                "judge_confidence": 0.9,
                "judge_reason": "matched refusal/premise-reject pattern",
                "judge_method": "heuristic",
            }
        # If the answer is long and confident with no rejection markers,
        # this is silent failure. But "long and confident" is fuzzy →
        # punt to LLM unless the answer is short and clearly affirmative.
        if len(answer) < 200 and not refusal_hit and not premise_reject_hit:
            return None  # ambiguous, let LLM decide
        return None

    # ----- roi_masked -----
    if kind == "roi_masked":
        if refusal_hit:
            return {
                "verdict": "refused_appropriately",
                "judge_confidence": 0.9,
                "judge_reason": "matched refusal pattern with ROI occluded",
                "judge_method": "heuristic",
            }
        # Otherwise punt — model may be guessing-from-prior (silent fail)
        # or may genuinely answer if ROI happened to overlap visible content.
        return None

    # ----- roi_only / lr_flip / original / tcf / specificity_drop / negation / knowledge_only
    # All of these expect a substantive answer matching some gold.

    if not gold_norm:
        return None  # no gold to match against -> punt

    # exact / substring containment
    if gold_norm and gold_norm in norm_ans:
        return {
            "verdict": "correct",
            "judge_confidence": 0.9,
            "judge_reason": f"gold '{gold[:40]}' substring-contained in answer",
            "judge_method": "heuristic",
        }
    # answer contained in gold (model abbreviated)
    if norm_ans and norm_ans in gold_norm and len(norm_ans) >= 2:
        return {
            "verdict": "correct",
            "judge_confidence": 0.85,
            "judge_reason": "answer is substring of gold (abbrev)",
            "judge_method": "heuristic",
        }

    # yes/no fast-path
    if gold_norm in {"yes", "no"}:
        if gold_norm == "yes" and YES_RX.search(answer) and not NO_RX.search(answer):
            return {"verdict": "correct", "judge_confidence": 0.85,
                    "judge_reason": "yes/no positive match",
                    "judge_method": "heuristic"}
        if gold_norm == "no" and NO_RX.search(answer) and not YES_RX.search(answer):
            return {"verdict": "correct", "judge_confidence": 0.85,
                    "judge_reason": "yes/no positive match",
                    "judge_method": "heuristic"}
        if gold_norm == "yes" and NO_RX.search(answer) and not YES_RX.search(answer):
            return {"verdict": "incorrect", "judge_confidence": 0.85,
                    "judge_reason": "yes/no inverted",
                    "judge_method": "heuristic"}
        if gold_norm == "no" and YES_RX.search(answer) and not NO_RX.search(answer):
            return {"verdict": "incorrect", "judge_confidence": 0.85,
                    "judge_reason": "yes/no inverted",
                    "judge_method": "heuristic"}

    if refusal_hit:
        return {
            "verdict": "refused_inappropriately",
            "judge_confidence": 0.7,
            "judge_reason": "refused on a probe expecting a substantive answer",
            "judge_method": "heuristic",
        }

    return None  # ambiguous → LLM judge


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """You are an impartial scorer for a medical VLM benchmark. You
will receive ONE probe (question + expected behavior + gold) and ONE model
answer. Output STRICT JSON ONLY (no prose, no fences) with this schema:

{
  "verdict": "correct" | "incorrect" | "refused_appropriately"
             | "refused_inappropriately" | "silent_failure" | "hedged",
  "judge_confidence": <float 0..1>,
  "judge_reason": "<one short sentence>"
}

Verdict rules
=============
For probe_kind in {original, tcf, specificity_drop, lr_flip, roi_only,
                   knowledge_only, negation}:
  correct                 if the model answer is semantically equivalent to
                          the gold answer (medical synonyms, plural/sing,
                          abbreviation are all fine; e.g. "RLL" ≡ "right
                          lower lobe")
  incorrect               if the model gave a clear, substantive answer
                          that differs from gold
  refused_inappropriately if the model refused/declined when a substantive
                          answer was expected
  hedged                  if the model gave a non-committal, mixed-signal
                          answer that neither matches nor clearly differs

For probe_kind == "halluc_trap":
  refused_appropriately   if the model recognized and rejected the false
                          premise (e.g. "no, X is not present", "the
                          premise is incorrect")
  silent_failure          if the model confidently answered as if the false
                          premise were true (gave a specific finding,
                          location, severity)
  hedged                  if the model gave an uncertain answer that
                          neither rejects nor accepts the premise
  (no "correct"/"incorrect" verdicts apply to traps)

For probe_kind == "roi_masked":
  refused_appropriately   if the model expressed inability to tell from the
                          occluded image
  silent_failure          if the model confidently answered the original
                          question (i.e., guessed from prior knowledge
                          without using the ROI)
  hedged                  if the model partially answered with hedging

GENERAL
=======
- Be strict on "correct": medical specificity matters (e.g. "left lower
  lobe" is NOT equivalent to "lower lobe").
- Be lenient on phrasing/length: a long verbose answer that contains the
  gold semantically is "correct".
- "judge_confidence" should reflect ambiguity: 0.95 for clear-cut, 0.6 for
  borderline.
"""


def build_judge_user(probe: dict) -> str:
    parts = [
        f"probe_kind: {probe['probe_kind']}",
        f"question: {probe['question']}",
    ]
    gold = probe.get("expected_answer_or_set", "")
    if gold:
        parts.append(f"gold answer: {gold}")
    if probe["probe_kind"] == "halluc_trap":
        parts.append(f"FALSE PREMISE in this trap: {probe.get('halluc_explanation','(unknown)')}")
        parts.append("(The correct behavior is to recognize and reject the premise.)")
    if probe["probe_kind"] == "roi_masked":
        parts.append("(The relevant region of the image has been masked grey. "
                     "The correct behavior is to express inability to determine.)")
    parts.append(f"\nmodel answer: {probe.get('model_answer','')}")
    parts.append("\nGenerate the verdict JSON now.")
    return "\n".join(parts)


def parse_json(raw: str) -> dict:
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[:-3]
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
    return json.loads(s)


async def llm_judge(client, sem, probe: dict) -> dict:
    async with sem:
        for tok in (JUDGE_MAX_TOKENS, JUDGE_MAX_TOKENS * 2):
            try:
                r = await client.chat.completions.create(
                    model=JUDGE_MODEL,
                    max_completion_tokens=tok,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": build_judge_user(probe)},
                    ],
                    response_format={"type": "json_object"},
                )
                raw = r.choices[0].message.content or ""
                v = parse_json(raw)
                v["judge_method"] = "llm"
                return v
            except Exception as e:
                last_err = e
                continue
        return {
            "verdict": "hedged",
            "judge_confidence": 0.0,
            "judge_reason": f"llm_error: {str(last_err)[:120]}",
            "judge_method": "llm_failed",
        }


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def load_probes_index() -> dict[str, dict]:
    with open(PROBES_OPEN, newline="") as f:
        return {r["probe_id"]: r for r in csv.DictReader(f)}


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add(r["probe_id"])
            except Exception:
                continue
    return done


def enrich_with_probe_meta(row: dict, probes_idx: dict) -> dict:
    p = probes_idx.get(row["probe_id"], {})
    if "probe_kind" not in row: row["probe_kind"] = p.get("probe_kind", "")
    if "question" not in row: row["question"] = p.get("question", "")
    if "case_id" not in row: row["case_id"] = p.get("case_id", "")
    row.setdefault("expected_answer_or_set", p.get("expected_answer_or_set", ""))
    row.setdefault("halluc_explanation", p.get("halluc_explanation", ""))
    return row


async def process_one(client, sem, row: dict) -> dict:
    h = heuristic_judge(row)
    if h is not None:
        return {**row, **h}
    v = await llm_judge(client, sem, row)
    return {**row, **v}


async def main_async(args):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    probes_idx = load_probes_index()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path) as f:
        inputs = [json.loads(l) for l in f if l.strip()]
    print(f"[init] loaded {len(inputs)} input rows from {in_path}")

    inputs = [enrich_with_probe_meta(r, probes_idx) for r in inputs]
    done = load_done(out_path) if args.resume else set()
    if done:
        print(f"[init] resume: {len(done)} already judged")
    todo = [r for r in inputs if r["probe_id"] not in done]
    print(f"[init] todo: {len(todo)}")

    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    n_h = n_l = n_err = 0
    verdict_counts = {}

    out_f = open(out_path, "a" if args.resume else "w")
    tasks = [asyncio.create_task(process_one(client, sem, r)) for r in todo]
    for i, fut in enumerate(asyncio.as_completed(tasks), 1):
        result = await fut
        m = result.get("judge_method", "")
        if m == "heuristic" or m == "rule":
            n_h += 1
        elif m == "llm":
            n_l += 1
        else:
            n_err += 1
        verdict_counts[result.get("verdict", "?")] = verdict_counts.get(result.get("verdict", "?"), 0) + 1
        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_f.flush()
        if i % 50 == 0 or i == len(todo):
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-3)
            eta = (len(todo) - i) / max(rate, 1e-3)
            print(f"[{i}/{len(todo)}] heur={n_h} llm={n_l} err={n_err} "
                  f"{rate*60:.0f}/min ETA {eta/60:.1f}min  "
                  f"verdicts={dict(verdict_counts)}")
    out_f.close()

    print(f"\n[done] heuristic={n_h}  llm={n_l}  err={n_err}  "
          f"total {(time.time()-t0)/60:.1f}min")
    print(f"[verdicts] {dict(verdict_counts)}")
    try:
        print(f"[output]  -> {out_path.relative_to(ROOT)}")
    except ValueError:
        print(f"[output]  -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="JSONL of model responses (must have probe_id + model_answer)")
    ap.add_argument("--output", required=True,
                    help="Output JSONL with verdicts merged in")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
