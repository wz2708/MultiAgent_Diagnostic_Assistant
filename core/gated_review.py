from __future__ import annotations
from typing import List, Dict, Any, Tuple
import math

def _safe(x: float) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

def _softmax(xs: List[float], temperature: float = 1.0) -> List[float]:
    if temperature <= 0:
        temperature = 1.0
    m = max(xs) if xs else 0.0
    exps = [math.exp((x - m) / temperature) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]

def _qa_bonus(candidate_json: Dict[str, Any], bonus_if_fixed_dominate: float = 0.02) -> float:
    """
    Read candidate.rounds[*].critic.review list and count tags:
      "A# fixed", "A# not_fixed", "A# cannot_fix"
    If fixed > (not_fixed + cannot_fix), return bonus; else 0.
    """
    rounds = candidate_json.get("rounds", [])
    fixed = not_fixed = cannot_fix = 0
    for r in rounds:
        rev = (r.get("critic") or {}).get("review", []) or []
        for tag in rev:
            t = str(tag).strip().lower()
            if "fixed" in t and "not_fixed" not in t and "cannot" not in t:
                fixed += 1
            elif "not_fixed" in t:
                not_fixed += 1
            elif "cannot_fix" in t or "cannot fix" in t:
                cannot_fix += 1
    return bonus_if_fixed_dominate if fixed > (not_fixed + cannot_fix) else 0.0

def doctor_first_gate_rerank(
    candidates: List[Dict[str, Any]],
    review_aggregate: List[Dict[str, Any]] | None,
    alpha: float = 0.9,
    beta: float = 0.1,
    gamma: float = 0.0,
    p_lock: float = 0.6,
    margin: float = 0.15,
    swap_tau: float = 0.2,
    temperature: float = 1.0,
    enable_bonus: bool = True,
    bonus_if_fixed_dominate: float = 0.02
) -> List[Dict[str, Any]]:
    """
    Inputs:
      candidates: [
        {
          "diagnosis": str,
          "department": List[str],
          "doctor_conf": float,
          "expert_conf": float,
          "review_score": float,  # optional; if missing, will be looked up from review_aggregate
          "rounds": [...]         # critic/expert rounds for bonus inspection (optional)
        },
        ...
      ]
      review_aggregate: [
        {"diagnosis": "...", "final_score": float, ...},
        ...
      ]
    Returns candidates with fields:
      "score" (final fused score), "final_rank" (1-based)
    """
    # Map review scores if not attached
    name2rev = {}
    if review_aggregate:
        for it in review_aggregate:
            dx = str(it.get("diagnosis", "")).strip().lower()
            name2rev[dx] = _safe(it.get("final_score", 0.0))

    # Ensure review_score populated
    for c in candidates:
        if c.get("review_score") is None:
            dx_l = str(c.get("diagnosis", "")).strip().lower()
            c["review_score"] = _safe(name2rev.get(dx_l, 0.0))

    # Softmax normalize each channel for stability
    doc_vec = [_safe(c.get("doctor_conf", 0.0)) for c in candidates]
    exp_vec = [_safe(c.get("expert_conf", 0.0)) for c in candidates]
    rev_vec = [_safe(c.get("review_score", 0.0)) for c in candidates]

    doc_n = _softmax(doc_vec, temperature=temperature)
    exp_n = _softmax(exp_vec, temperature=temperature)
    rev_n = _softmax(rev_vec, temperature=temperature) if any(rev_vec) else [0.0]*len(candidates)

    # Gating conditions decided on the doctor's top1 confidence and margin
    # Find top1 and top2 doctor confidences
    sorted_by_doc = sorted(
        [(i, d) for i, d in enumerate(doc_vec)],
        key=lambda x: x[1], reverse=True
    )
    if not sorted_by_doc:
        # Degenerate: fallback to review-heavy
        for i, c in enumerate(candidates):
            c["score"] = 0.2*doc_n[i] + 0.2*exp_n[i] + 0.6*rev_n[i]
        ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
        for r, c in enumerate(ranked, 1):
            c["final_rank"] = r
        return ranked

    i1, top1 = sorted_by_doc[0]
    top2 = sorted_by_doc[1][1] if len(sorted_by_doc) > 1 else 0.0
    doc_gap = top1 - top2

    # Gate open?
    gate_review = (top1 < p_lock) or (doc_gap < margin)

    # Compute candidate raw fused scores
    fused = []
    for i, c in enumerate(candidates):
        doc_s = doc_n[i]
        exp_s = exp_n[i]
        rev_s = rev_n[i]
        bonus = _qa_bonus(c, bonus_if_fixed_dominate) if (gate_review and enable_bonus) else 0.0

        if gate_review:
            s = alpha*doc_s + beta*exp_s + gamma*rev_s + bonus
        else:
            # Doctor-first strict ranking
            s = 1.0*doc_s + 0.2*exp_s  # tiny preference to expert as tie-breaker
        fused.append((s, i))

    # If gate closed: rank by fused (which is essentially doc+tiny exp)
    if not gate_review:
        fused.sort(key=lambda x: x[0], reverse=True)
        ranked = [candidates[j] for (_, j) in fused]
        for r, c in enumerate(ranked, 1):
            c["score"] = fused[r-1][0]
            c["final_rank"] = r
        return ranked

    # Gate open: allow reviewer to influence, but apply swap threshold
    # Compare new top1 vs original doctor top1
    fused.sort(key=lambda x: x[0], reverse=True)
    # Index currently favored by fused + gate
    new_top_idx = fused[0][1]
    old_top_idx = i1
    # Margin advantage in fused space
    if new_top_idx != old_top_idx:
        new_top_score = fused[0][0]
        old_top_score = [s for (s, j) in fused if j == old_top_idx][0]
        if (new_top_score - old_top_score) < swap_tau:
            # Force keep doctor's original #1 on top
            fused = [(s, j) for (s, j) in fused if j != old_top_idx]
            fused.insert(0, (old_top_score, old_top_idx))

    ranked = [candidates[j] for (_, j) in fused]
    for r, c in enumerate(ranked, 1):
        c["score"] = fused[r-1][0]
        c["final_rank"] = r
    return ranked
