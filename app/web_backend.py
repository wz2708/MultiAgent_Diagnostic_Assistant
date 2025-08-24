"""
Backend helpers for Streamlit demo.
- Reuse existing agents
- Add timing logs and optional disk cache
- Explain Reviewer gate decision
- Return compact objects ready for UI rendering
"""
from __future__ import annotations

import os
import io
import json
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.util import ResourcePool
from core.gated_review import doctor_first_gate_rerank  # gate decision
from agents.GeneralDoct_agent import GeneralDoctorAgent, DeptDiagnosisResult
from agents.DeptCritic_agent import DeptCriticAgent, CriticResult
from agents.DeptExpert_agent import DeptExpertAgent, DeptExpertResult
from agents.Review_agent import ReviewAgent
from agents.Decision_agent import DiseaseMatchAgent


DEFAULT_REVIEWER_WEIGHTS: Dict[str, float] = {
    "accuracy": 0.3, "coverage": 0.3, "interpretability": 0.2, "specificity": 0.2
}
DEFAULT_FUSION_PARAMS: Dict[str, Any] = {
    "alpha": 0.8, "beta": 0.15, "gamma": 0.05,
    "p_lock": 0.4, "margin": 0.15,
    "swap_tau": 0.2, "temperature": 1.0,
    "enable_bonus": True, "bonus_if_fixed_dominate": 0.02,
}

# Small disk cache to avoid repeated LLM charges for identical inputs.
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

def _hash_key(payload: Dict[str, Any]) -> str:
    """Stable sha1 for dict payload."""
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()

def _cache_load(key: str) -> Optional[Dict[str, Any]]:
    p = CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text("utf-8"))
        except Exception:
            return None
    return None

def _cache_save(key: str, obj: Dict[str, Any]) -> None:
    p = CACHE_DIR / f"{key}.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)

# Utilities
def _as_one_name(x) -> str:
    if isinstance(x, list):
        return x[0] if x else ""
    return str(x or "")

def _as_one_conf(x) -> float:
    if isinstance(x, list):
        return float(x[0]) if x else 0.0
    try:
        return float(x)
    except Exception:
        return 0.0

def normalize_gd_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure expected keys exist."""
    d = d or {}
    d.setdefault("diagnoses", [])
    for it in d["diagnoses"]:
        it.setdefault("department", [])
        it.setdefault("diagnose", [])
        it.setdefault("confidence", [])
        it.setdefault("reasoning", [])
        it.setdefault("reference", [])
    return d

def _build_fusion_inputs(candidates_json: List[Dict[str, Any]], aggregate: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reformat inputs for doctor_first_gate_rerank (unchanged)."""
    score_map = {x["diagnosis"]: x["final_score"] for x in aggregate} if aggregate else {}
    out = []
    for c in candidates_json:
        dx = c["diagnosis"]
        out.append({
            "diagnosis": dx,
            "department": c.get("department") or [],
            "doctor_conf": float(c.get("doctor_confidence", 0.0)),
            "expert_conf": float(c.get("expert_confidence_last_round", 0.0)),
            "review_score": float(score_map.get(dx, 0.0)),
        })
    return out

def _format_ranking_output(ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(ranked, start=1):
        r["final_rank"] = i
    return ranked

async def run_three_rounds_for_candidate(
    *,
        department: str,
        candidate_dx: str,
        symptoms_text: str,
        patient_history: str,
        doctor_reasoning: List[str],
        doctor_reference: List[str],
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Run up to 3 Critic–Expert rounds with early stop if critique is empty.
    Return (rounds_json, expert_conf_last_round).
    """
    critic = DeptCriticAgent(
        department=department or "General Medicine",
        candidate=candidate_dx,
        symptoms=symptoms_text,
        history=patient_history
    )
    expert = DeptExpertAgent(
        department=department or "General Medicine",
        candidate=candidate_dx,
        symptoms=symptoms_text,
        history=patient_history
    )

    rounds: List[Dict[str, Any]] = []
    last_expert_conf: float = 0.0

    # DeptHead-style payload for the critic (compatible with your prompt)
    doctor_payload = {
        "diagnose": [candidate_dx],
        "confidence": [0.0],
        "reasoning": doctor_reasoning or [],
        "reference": doctor_reference or []
    }

    for _r in range(1, 4):
        # (1) Critic turn
        critic_res: CriticResult = await critic.run(doctor_result=doctor_payload)
        critic_json = critic_res.model_dump()

        if not critic_json.get("critique"):
            rounds.append({"critic": critic_json})
            break

        # (2) Expert turn (only reached when there are non-empty questions)
        expert_res: DeptExpertResult = await expert.run(critic_feedback=critic_json)
        expert_json = expert_res.model_dump()
        last_expert_conf = expert_res.confidence[0]

        rounds.append({"critic": critic_json, "expert": expert_json})

    return rounds, last_expert_conf

async def prepare_agents_for_app() -> Dict[str, Any]:
    """Reuse heavy clients across UI interactions."""
    gd_agent = GeneralDoctorAgent()
    decision_agent = DiseaseMatchAgent()
    return {"gd": gd_agent, "decision": decision_agent}

def should_run_reviewers(
    candidates_json: List[Dict[str, Any]],
    p_lock: float = 0.4,
    margin: float = 0.15
) -> bool:
    """
    Same gate as your fusion logic:
      - If top1 doctor_conf >= p_lock -> no reviewer
      - If gap(top1 - top2) >= margin -> no reviewer
      - Else -> reviewer
    """
    if not candidates_json:
        return False
    confs = sorted([float(c["doctor_confidence"]) for c in candidates_json], reverse=True)
    top1 = confs[0]
    top2 = confs[1] if len(confs) > 1 else 0.0
    if top1 >= p_lock:
        return False
    if (top1 - top2) >= margin:
        return False
    return True

def _gate_explanation(candidates: List[Dict[str, Any]], p_lock: float, margin: float) -> Dict[str, Any]:
    """
    Create a human-readable gate reason.
    - delta = top1_conf - top2_conf (if any)
    - need_review = decision from _should_run_reviewers
    """
    if not candidates:
        return {"need_review": False, "reason": "No candidates", "delta": 0.0, "p_top1": 0.0}
    # gather doctor confidences
    confs = sorted((float(c.get("doctor_confidence", 0.0)) for c in candidates), reverse=True)
    p_top1 = confs[0] if confs else 0.0
    delta = (confs[0] - confs[1]) if len(confs) >= 2 else confs[0]
    need_review = should_run_reviewers(candidates_json=candidates, p_lock=p_lock, margin=margin)
    if need_review:
        reason = f"Triggered: Δdoctor={delta:.2f} < margin({margin:.2f}) OR p_top1({p_top1:.2f}) < p_lock({p_lock:.2f})"
    else:
        reason = f"Skipped: Δdoctor={delta:.2f} ≥ margin({margin:.2f}) AND p_top1({p_top1:.2f}) ≥ p_lock({p_lock:.2f})"
    return {"need_review": bool(need_review), "reason": reason, "delta": delta, "p_top1": p_top1}

async def run_case_for_app(
    *,
    agents: Dict[str, Any],
    symptoms_text: str,
    patient_history: str,
    topk_from_doctor: int,
    reviewer_weights: Dict[str, float],
    fusion_params: Dict[str, Any],
    truth: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run full flow for a single case with:
    - Disk cache to avoid repeated costs
    - Timing logs
    - Gate explanation
    - Return compact JSON for UI
    """
    # -------- Input validation --------
    if not symptoms_text or len(symptoms_text.strip()) < 8:
        raise ValueError("Symptoms text is too short. Please provide more details.")
    if topk_from_doctor <= 0:
        raise ValueError("Top-K must be >= 1.")

    # -------- Cache key --------
    cache_key = _hash_key({
        "symptoms": symptoms_text.strip(),
        "history": patient_history.strip(),
        "topk": topk_from_doctor,
        "weights": reviewer_weights,
        "fusion": fusion_params,
    })
    cached = _cache_load(cache_key)
    if cached:
        cached["summary"]["from_cache"] = True
        return cached

    # -------- Timers --------
    t0 = time.perf_counter()
    timings: Dict[str, float] = {}
    llm_calls = 0

    # -------- General Doctor --------
    td0 = time.perf_counter()
    gd: GeneralDoctorAgent = agents["gd"]
    gd_res: DeptDiagnosisResult = await gd.run(symptoms_text=symptoms_text, patient_history=patient_history)
    timings["general_doctor_s"] = time.perf_counter() - td0
    llm_calls += 1

    gd_dict = normalize_gd_dict(gd_res.model_dump())
    items = (gd_dict.get("diagnoses") or [])[: topk_from_doctor]

    candidates_json: List[Dict[str, Any]] = []
    candidate_names: List[str] = []
    reports_for_reviewer: List[List[str]] = []

    # -------- Critic–Expert rounds --------
    for idx, item in enumerate(items, start=1):
        dept = item.get("department") or []
        dx = _as_one_name(item.get("diagnose"))
        dconf = _as_one_conf(item.get("confidence"))
        doc_reason = item.get("reasoning") or []
        doc_refs = item.get("reference") or []

        tr0 = time.perf_counter()
        rounds, expert_conf = await run_three_rounds_for_candidate(
            department=(dept[0] if dept else "General"),
            candidate_dx=dx,
            symptoms_text=symptoms_text,
            patient_history=patient_history,
            doctor_reasoning=doc_reason,
            doctor_reference=doc_refs
        )
        timings[f"candidate_{idx}_dialog_s"] = time.perf_counter() - tr0
        # Rough LLM call estimate: 1 critic + (optional) 1 expert per round
        for r in rounds:
            llm_calls += 1  # critic
            if "expert" in r:
                llm_calls += 1

        candidates_json.append({
            "id": f"c{idx}",
            "department": dept,
            "diagnosis": dx,
            "doctor_confidence": dconf,
            "rounds": rounds,
            "expert_confidence_last_round": float(expert_conf)
        })
        candidate_names.append(dx)

        # Minimal doc-only report (one paragraph) for reviewers
        snippet = (", ".join(doc_reason) or "(no reasoning)")[:1200]
        reports_for_reviewer.append([snippet])

    # -------- Reviewer gate decision --------
    gate = _gate_explanation(candidates_json, p_lock=float(fusion_params["p_lock"]),
                             margin=float(fusion_params["margin"]))

    # -------- Reviewers (conditional) --------
    reviewers_json = {"accuracy": {}, "coverage": {}, "interpretability": {}, "specificity": {}}
    aggregate: List[Dict[str, Any]] = []
    if candidate_names and gate["need_review"]:
        r0 = time.perf_counter()
        Acc = ReviewAgent(perspective="Accuracy", symptoms=symptoms_text, candidates=candidate_names, reports=reports_for_reviewer)
        Cov = ReviewAgent(perspective="Coverage", symptoms=symptoms_text, candidates=candidate_names, reports=reports_for_reviewer)
        Intp = ReviewAgent(perspective="Interpretability", symptoms=symptoms_text, candidates=candidate_names, reports=reports_for_reviewer)
        Spec = ReviewAgent(perspective="Specificity", symptoms=symptoms_text, candidates=candidate_names, reports=reports_for_reviewer)

        acc_res = await Acc.run(); llm_calls += 1
        cov_res = await Cov.run(); llm_calls += 1
        intp_res = await Intp.run(); llm_calls += 1
        spec_res = await Spec.run(); llm_calls += 1

        reviewers_json = {
            "accuracy": acc_res.model_dump(),
            "coverage": cov_res.model_dump(),
            "interpretability": intp_res.model_dump(),
            "specificity": spec_res.model_dump()
        }
        # Aggregate
        def aggregate_reviews(accuracy, coverage, interpretability, specificity, weights):
            out = []
            diags = accuracy.get("diagnosis", [])
            for i, d in enumerate(diags):
                final = (accuracy["scores"][i] * weights["accuracy"] +
                         coverage["scores"][i] * weights["coverage"] +
                         interpretability["scores"][i] * weights["interpretability"] +
                         specificity["scores"][i] * weights["specificity"])
                out.append({
                    "diagnosis": d,
                    "final_score": float(final),
                    "breakdown": {
                        "Accuracy": accuracy["scores"][i],
                        "Coverage": coverage["scores"][i],
                        "Interpretability": interpretability["scores"][i],
                        "Specificity": specificity["scores"][i],
                    }
                })
            out.sort(key=lambda x: x["final_score"], reverse=True)
            return out

        aggregate = aggregate_reviews(
            reviewers_json["accuracy"], reviewers_json["coverage"],
            reviewers_json["interpretability"], reviewers_json["specificity"],
            reviewer_weights
        )
        timings["reviewers_s"] = time.perf_counter() - r0

    # -------- Fusion ranking --------
    fusion_inputs = _build_fusion_inputs(candidates_json, aggregate)
    ranked = doctor_first_gate_rerank(
        candidates=fusion_inputs,
        review_aggregate=aggregate,
        **fusion_params
    )
    ranking = _format_ranking_output(ranked)

    # -------- Decision (optional) --------
    decision_json: Optional[Dict[str, Any]] = None
    if truth is not None:
        dec_agent: DiseaseMatchAgent = agents["decision"]
        # short-circuit in your main pipeline; here keep simple
        # (reuse your own decide function if you have one)
        decision_json = {"truth": truth, "best_rank": None, "mrr": 0.0}
        for j, r in enumerate(ranking, start=1):
            if r["diagnosis"].lower().strip() == truth.lower().strip():
                decision_json["best_rank"] = j
                decision_json["mrr"] = 1.0 / j
                break

    total_s = time.perf_counter() - t0

    out = {
        "input": {
            "symptoms_text": symptoms_text,
            "patient_history": patient_history
        },
        "gate": gate,                               # human-readable gate explanation
        "timing": {**timings, "total_s": total_s},  # timing stats
        "llm_calls": llm_calls,                     # rough count
        "general_doctor": gd_dict,
        "candidates": candidates_json,
        "reviewers": reviewers_json,
        "review_aggregate": aggregate,
        "ranking": ranking,
        "decision": decision_json,
        "summary": {
            "latency_ms": int(total_s * 1000),
            "tokens_in": None,             # (optional) integrate if your model returns usage
            "tokens_out": None,
            "cost_usd": None,
            "from_cache": False,
            "top1": (ranking[0]["diagnosis"] if ranking else None)
        }
    }

    # Save cache
    _cache_save(cache_key, out)
    return out
