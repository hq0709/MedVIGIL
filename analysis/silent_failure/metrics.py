"""Silent-failure metric implementations: SFR, CFHR, SDM, VFR, CSC.

Inputs are lists of generation records, each a dict with at least:
    {
        'case_id': str,
        'question': str,
        'answer': str,
        'gold_severity': int (0-4) or None,
        'text_only_answerable': bool,
        'degradation': str,
        'seed': int,
        'confidence': float or None,
    }
This module is decoupled from any specific data source so the same code
can score IIST runs, exp4 reanalysis, or future benchmarks.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from .refusal import RefusalJudge
from .severity import SeverityScorer


@dataclass
class ScoredRecord:
    case_id: str
    question: str
    answer: str
    degradation: str
    seed: int
    text_only_answerable: bool
    gold_severity: Optional[int]
    pred_severity: int
    is_refusal: bool
    confidence: Optional[float]


def score_records(
    records: Iterable[dict],
    scorer: Optional[SeverityScorer] = None,
    judge: Optional[RefusalJudge] = None,
) -> List[ScoredRecord]:
    scorer = scorer or SeverityScorer()
    judge = judge or RefusalJudge(scorer=scorer)
    out: List[ScoredRecord] = []
    for r in records:
        ans = r.get("answer", "")
        sev = scorer.score(ans)
        decision = judge.judge(r.get("question", ""), ans)
        out.append(
            ScoredRecord(
                case_id=r.get("case_id", ""),
                question=r.get("question", ""),
                answer=ans,
                degradation=r.get("degradation", "clean"),
                seed=int(r.get("seed", 0)),
                text_only_answerable=bool(r.get("text_only_answerable", False)),
                gold_severity=r.get("gold_severity"),
                pred_severity=sev.sigma,
                is_refusal=(decision.label == "REFUSAL"),
                confidence=r.get("confidence"),
            )
        )
    return out


def compute_sfr(scored: List[ScoredRecord], degradation: str) -> float:
    """Silent Failure Rate on the vision-required subset (text_only=False)."""
    subset = [s for s in scored if s.degradation == degradation and not s.text_only_answerable]
    if not subset:
        return float("nan")
    non_refusal = sum(1 for s in subset if not s.is_refusal)
    return non_refusal / len(subset)


def compute_cfhr(
    scored: List[ScoredRecord],
    degradation: str,
    critical_threshold: int = 4,
) -> float:
    """Critical-Findings Hallucination Rate."""
    subset = [s for s in scored if s.degradation == degradation and not s.text_only_answerable]
    if not subset:
        return float("nan")
    n_crit = 0
    for s in subset:
        gold = s.gold_severity if s.gold_severity is not None else 0
        if s.pred_severity >= critical_threshold and gold < critical_threshold:
            n_crit += 1
    return n_crit / len(subset)


def compute_sdm(scored: List[ScoredRecord], degradation: str) -> float:
    """Severity Drift Magnitude vs the clean condition.

    Requires both 'clean' and the target degradation to be present in scored.
    """
    deg_subset = [s for s in scored if s.degradation == degradation]
    clean_subset = [s for s in scored if s.degradation == "clean"]
    if not deg_subset or not clean_subset:
        return float("nan")
    return float(np.mean([s.pred_severity for s in deg_subset])) - float(
        np.mean([s.pred_severity for s in clean_subset])
    )


def compute_vfr(scored: List[ScoredRecord], k_pct: int) -> float:
    """Visual Faithfulness Rate at top-k% mask.

    Expects scored to include records with degradation='topk_<pct>' and
    matching 'clean' baselines per case_id.
    """
    deg_label = f"topk_{k_pct}"
    clean_by_case = {s.case_id: s for s in scored if s.degradation == "clean"}
    deg_records = [s for s in scored if s.degradation == deg_label]
    if not deg_records:
        return float("nan")
    n_changed = 0
    for s in deg_records:
        base = clean_by_case.get(s.case_id)
        if base is None:
            continue
        if s.pred_severity != base.pred_severity:
            n_changed += 1
    return n_changed / len(deg_records)


def compute_csc(scored: List[ScoredRecord], degradation: str) -> float:
    """Confidence-Severity Coupling correlation under a degradation."""
    subset = [
        s
        for s in scored
        if s.degradation == degradation and s.confidence is not None
    ]
    if len(subset) < 3:
        return float("nan")
    confs = np.array([s.confidence for s in subset])
    sevs = np.array([s.pred_severity for s in subset])
    if confs.std() == 0 or sevs.std() == 0:
        return float("nan")
    return float(np.corrcoef(confs, sevs)[0, 1])


def severity_histogram(scored: List[ScoredRecord], degradation: str) -> Dict[int, int]:
    h: Dict[int, int] = defaultdict(int)
    for s in scored:
        if s.degradation == degradation:
            h[s.pred_severity] += 1
    return dict(h)


def summarize(scored: List[ScoredRecord]) -> Dict:
    """Convenience: produce a per-degradation summary dict."""
    degs = sorted({s.degradation for s in scored})
    summary: Dict = {}
    for d in degs:
        summary[d] = {
            "n": sum(1 for s in scored if s.degradation == d),
            "sfr": compute_sfr(scored, d),
            "cfhr": compute_cfhr(scored, d),
            "sdm": compute_sdm(scored, d) if d != "clean" else 0.0,
            "csc": compute_csc(scored, d),
            "severity_hist": severity_histogram(scored, d),
        }
    return summary
