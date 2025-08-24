from __future__ import annotations
import os
import re
import json
import argparse
import asyncio
from datetime import datetime, timezone

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import math
import pandas as pd
from tqdm import tqdm

from core.util import ResourcePool
from core.gated_review import doctor_first_gate_rerank
from agents.GeneralDoct_agent import GeneralDoctorAgent, DeptDiagnosisResult
from agents.DeptCritic_agent import DeptCriticAgent, CriticResult
from agents.DeptExpert_agent import DeptExpertAgent, DeptExpertResult
from agents.Review_agent import ReviewAgent
from agents.Decision_agent import DiseaseMatchAgent

# ========== Storage helpers ==========
def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)

def atomic_write_json(path: Path, obj: Any, indent: int = 2) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=indent)
    atomic_write_text(path, text)

def slugify(s: str, max_len: int = 80) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s\-]+", "", s)
    s = re.sub(r"\s+", "-", s)
    return s[:max_len] or "na"

def _as_one_name(diag_field: Any) -> str:
    if isinstance(diag_field, list) and len(diag_field) > 0:
        return str(diag_field[0])
    if isinstance(diag_field, str):
        return diag_field
    return "Unknown"

def _as_one_conf(conf_field: Any) -> float:
    if isinstance(conf_field, list) and len(conf_field) > 0 and isinstance(conf_field[0], (int, float)):
        return float(conf_field[0])
    if isinstance(conf_field, (int, float)):
        return float(conf_field)
    return 0.0

def normalize_gd_dict(gd_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure 'diagnoses' items have diagnose: List[str] and confidence: List[float]."""
    for item in gd_dict.get("diagnoses", []):
        diag = item.get("diagnose")
        if isinstance(diag, str):
            item["diagnose"] = [diag]
        elif isinstance(diag, list) and diag:
            item["diagnose"] = [str(diag[0])]
        else:
            item["diagnose"] = ["Unknown"]

        conf = item.get("confidence")
        if isinstance(conf, (int, float)):
            item["confidence"] = [float(conf)]
        elif isinstance(conf, list) and conf and isinstance(conf[0], (int, float)):
            item["confidence"] = [float(conf[0])]
        else:
            item["confidence"] = [0.33]
    return gd_dict

def build_report_md(dept: List[str], dx: str, doc_reason: List[str], doc_refs: List[str],
                    rounds: List[Dict[str, Any]]) -> str:
    """Compact, human-readable report for Reviewer (one long paragraph list)."""
    lines = []
    lines.append(f"# Diagnostic Report: {dx}")
    lines.append(f"**Department:** {', '.join(dept) if dept else 'N/A'}")
    lines.append("\n## 1) GeneralDoctor Reasoning")
    for r in doc_reason or []:
        lines.append(f"- {r}")

    for ridx, rd in enumerate(rounds, start=1):
        critic = rd.get("critic", {})
        expert = rd.get("expert", {})
        lines.append(f"\n## {ridx}. Critic Round {ridx} Questions")
        for q in critic.get("critique", []) or []:
            lines.append(f"- {q}")
        if critic.get("review"):
            lines.append("\n**Critic Review:**")
            for a in critic["review"]:
                lines.append(f"- {a}")

        lines.append(f"\n## {ridx}. Expert Round {ridx} Responses (confidence={expert.get('confidence')})")
        for a in expert.get("response", []) or []:
            lines.append(f"- {a}")
    return "\n".join(lines)

def _build_fusion_inputs(
    candidates_json: List[Dict[str, Any]],
    review_aggregate: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Map your per-candidate JSON to the fusion function input format.
    Keeps department/diagnosis/doctor_conf/expert_conf/review_score/rounds.
    """
    # (Optional) pre-build a map from aggregated review scores
    name2rev = { (it.get("diagnosis","") or "").strip().lower(): float(it.get("final_score", 0.0))
                 for it in (review_aggregate or []) }

    fused_inputs: List[Dict[str, Any]] = []
    for c in candidates_json:
        dx = str(c.get("diagnosis",""))
        dx_l = dx.strip().lower()
        fused_inputs.append({
            "diagnosis": dx,
            "department": c.get("department", []),
            "doctor_conf": float(c.get("doctor_confidence", 0.0)),
            "expert_conf": float(c.get("expert_confidence_last_round", 0.0)),
            # Let the fusion function auto-fill from review_aggregate if None
            "review_score": name2rev.get(dx_l, None),
            "rounds": c.get("rounds", [])
        })
    return fused_inputs


# ========== Debate between Critic & Expert for one candidate ==========
async def run_three_rounds_for_candidate(
    department: str,
    candidate_dx: str,
    symptoms_text: str,
    patient_history: str,
    doctor_reasoning: List[str],
    doctor_reference: List[str],
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Hand-wired 3 rounds Critic<->Expert for ONE candidate.
    Return (rounds_json_list, last_expert_confidence).
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


# ========== Reviewers & aggregation ==========
def aggregate_reviews(
    accuracy: Dict,
    coverage: Dict,
    interpretability: Dict,
    specificity: Dict,
    weights: Dict[str, float],
) -> List[Dict]:
    """Merge 4 ReviewAgent outputs into a final ranked list; follows your previous logic and shapes."""
    def _norm_list(lst, n):
        if len(lst) < n:
            return lst + [""] * (n - len(lst))
        return lst[:n]

    diagnoses = accuracy["diagnosis"]
    n = len(diagnoses)
    for block in (accuracy, coverage, interpretability, specificity):
        block["references"] = _norm_list(block.get("references", []), n)
        block["scores"] = _norm_list(block.get("scores", []), n)

    reports = []
    for i, diag in enumerate(diagnoses):
        a = accuracy["scores"][i]
        c = coverage["scores"][i]
        it = interpretability["scores"][i]
        s = specificity["scores"][i]
        final = a * weights["accuracy"] + c * weights["coverage"] + it * weights["interpretability"] + s * weights["specificity"]
        refs = {
            "Accuracy": accuracy["references"][i],
            "Coverage": coverage["references"][i],
            "Interpretability": interpretability["references"][i],
            "Specificity": specificity["references"][i],
        }
        reports.append({
            "diagnosis": diag,
            "final_score": round(float(final), 4),
            "breakdown": {"Accuracy": a, "Coverage": c, "Interpretability": it, "Specificity": s},
            "references": refs,
        })
    reports.sort(key=lambda x: x["final_score"], reverse=True)
    return reports


def _should_run_reviewers(
    candidates_json: List[Dict[str, Any]],
    p_lock: float = 0.4,
    margin: float = 0.15,
) -> bool:
    """
    Doctor-first gate BEFORE building reviewers:
    - If top1 doctor_conf >= p_lock AND (top1 - top2) >= margin => skip reviewers
    - Otherwise => run reviewers
    """
    if not candidates_json:
        return False  # nothing to review
    vals = sorted(
        [float(c.get("doctor_confidence", 0.0)) for c in (candidates_json or [])],
        reverse=True
    )
    # pad with zeros if fewer than 2
    while len(vals) < 2:
        vals.append(0.0)
    top1, top2 = vals[:2]

    if (top1 >= p_lock) and ((top1 - top2) >= margin):
        return False  # reviewers not needed (doctor locked)
    return True       # reviewers needed

# ========== Decision with Decision Agent ==========
async def decide_with_short_circuit_reuse(
    ranking: List[Dict[str, Any]],
    truth: str,
    decision_agent: Optional[Any] = None
) -> Dict[str, Any]:
    hits = {f"top{k}_hit": False for k in range(1, 6)}
    best_rank: Optional[int] = None

    norm = lambda s: (s or "").strip().lower()

    # direct normalized string match
    for idx, item in enumerate(ranking[:5], start=1):
        if norm(item["diagnosis"]) == norm(truth):
            best_rank = idx
            hits[f"top{idx}_hit"] = True
            return {"truth": truth, "best_rank": best_rank, "mrr": 1.0 / best_rank, **hits}

    # DecisionAgent (reuse if provided)
    agent = decision_agent
    if agent is not None:
        for idx, item in enumerate(ranking[:5], start=1):
            pred = item["diagnosis"]
            try:
                res = await agent.run(pred=pred, truth=truth)
                if bool(getattr(res, "is_same", False)):
                    best_rank = idx
                    hits[f"top{idx}_hit"] = True
                    break
            except Exception:
                pass

    mrr = 1.0 / best_rank if best_rank else 0.0
    return {"truth": truth, "best_rank": best_rank, "mrr": mrr, **hits}


# ========== Single case orchestration ==========
async def run_single_case(
    case_id: str,
    symptoms_text: str,
    patient_history: str,
    topk_from_doctor: int = 5,
    reviewer_weights: Optional[Dict[str, float]] = None,
    truth: Optional[str] = None,
    tmp_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
        Purpose: Run the full diagnostic pipeline for one patient case.
        Inputs:
            case_id: unique ID for the case.
            symptoms_text: free-text symptom description.
            patient_history: optional patient history string.
            topk_from_doctor: number of top diagnoses to consider from the GeneralDoctor.
            reviewer_weights: weighting for Accuracy / Coverage / Interpretability / Specificity reviewers.
            truth: optional ground truth label (used for dataset eval).
            tmp_dir: optional dir to write resume-safe temporary snapshot.
        Steps:
            General Doctor Agent → produce initial candidate diagnoses (gd_res).
            Decision Agent (disease matching, evaluation baseline).
            Candidate Loop: For each top-k diagnosis:
                Run Expert–Critic three rounds (run_three_rounds_for_candidate).
                Collect reasoning + confidence.
                Optionally dump to .tmp file for safety.
            Reviewer Gate: Decide whether to invoke Review Agents.
                If needed: spawn 4 reviewers (Accuracy, Coverage, Interpretability, Specificity).
                Collect judgments and aggregate via weighted fusion.
            Ranking Fusion: Combine doctor, expert, and reviewer signals using Doctor-First gated re-ranking.
            Decision: If ground truth is available → run short-circuit evaluation (decide_with_short_circuit_reuse).
            Assemble Final JSON: Single structured object with meta, input, candidates, reviewers, fusion ranking, decision.
        Output: Full JSON dictionary describing every step of reasoning + final ranking
    """
    reviewer_weights = reviewer_weights or {"accuracy": 0.3, "coverage": 0.3, "interpretability": 0.2, "specificity": 0.2}
    ts = datetime.now(timezone.utc).isoformat()

    # GeneralDoctor
    gd = GeneralDoctorAgent()
    gd_res: DeptDiagnosisResult = await gd.run(symptoms_text=symptoms_text, patient_history=patient_history)
    gd_dict = normalize_gd_dict(gd_res.model_dump())

    # Decision Agent
    decision_agent = DiseaseMatchAgent()

    # choose topK
    items = (gd_dict.get("diagnoses") or [])[:topk_from_doctor]

    # candidates flow
    candidates_json: List[Dict[str, Any]] = []
    candidate_names: List[str] = []
    reports_for_reviewer: List[List[str]] = []

    for idx, item in enumerate(items, start=1):
        dept = item.get("department") or []
        dx = _as_one_name(item.get("diagnose"))
        dconf = _as_one_conf(item.get("confidence"))
        doc_reason = item.get("reasoning") or []
        doc_refs = item.get("reference") or []

        rounds, expert_conf = await run_three_rounds_for_candidate(
            department=(dept[0] if dept else "General Medicine"),
            candidate_dx=dx,
            symptoms_text=symptoms_text,
            patient_history=patient_history,
            doctor_reasoning=doc_reason,
            doctor_reference=doc_refs
        )
        # per-candidate object
        cand_obj = {
            "id": f"c{idx}",
            "department": dept,
            "diagnosis": dx,
            "doctor_confidence": dconf,
            "rounds": rounds,  # raw Critic/Expert JSON for all 3 rounds
            "expert_confidence_last_round": float(expert_conf)
        }
        candidates_json.append(cand_obj)

        # reviewer input
        candidate_names.append(dx)
        report_md = build_report_md(dept=dept, dx=dx, doc_reason=doc_reason, doc_refs=doc_refs, rounds=rounds)
        reports_for_reviewer.append([report_md])

        # optional tmp snapshot
        if tmp_dir:
            tmp_payload = {
                "meta": {"case_id": case_id, "timestamp": ts},
                "input": {"symptoms_text": symptoms_text, "patient_history": patient_history},
                "general_doctor": gd_dict,
                "candidates": candidates_json
            }
            atomic_write_json(tmp_dir / f".case_{slugify(case_id)}.tmp", tmp_payload)

    # Reviewers (build only if needed by gate)
    reviewers_json = {"accuracy": {}, "coverage": {}, "interpretability": {}, "specificity": {}}
    aggregate = []
    reviewer_used = False  # mark for logging/metrics

    # gate BEFORE building reviewer agents
    need_review = _should_run_reviewers(
        candidates_json=candidates_json,
        p_lock=0.4,  # same as fusion_params (keep consistent)
        margin=0.15
    )

    if candidate_names and need_review:
        reviewer_used = True
        Acc = ReviewAgent(perspective="Accuracy", symptoms=symptoms_text, candidates=candidate_names,
                          reports=reports_for_reviewer)
        Cov = ReviewAgent(perspective="Coverage", symptoms=symptoms_text, candidates=candidate_names,
                          reports=reports_for_reviewer)
        Intp = ReviewAgent(perspective="Interpretability", symptoms=symptoms_text, candidates=candidate_names,
                           reports=reports_for_reviewer)
        Spec = ReviewAgent(perspective="Specificity", symptoms=symptoms_text, candidates=candidate_names,
                           reports=reports_for_reviewer)

        acc_res = await Acc.run()
        cov_res = await Cov.run()
        intp_res = await Intp.run()
        spec_res = await Spec.run()

        reviewers_json = {
            "accuracy": acc_res.model_dump(),
            "coverage": cov_res.model_dump(),
            "interpretability": intp_res.model_dump(),
            "specificity": spec_res.model_dump()
        }
        weights = {"accuracy": 0.3, "coverage": 0.3, "interpretability": 0.2, "specificity": 0.2}
        aggregate = aggregate_reviews(
            reviewers_json["accuracy"], reviewers_json["coverage"],
            reviewers_json["interpretability"], reviewers_json["specificity"],
            weights
        )

    # Ranking fusion
    fusion_params = {
        "alpha": 0.8, "beta": 0.15, "gamma": 0.05,
        "p_lock": 0.4, "margin": 0.15,
        "swap_tau": 0.2, "temperature": 1.0,
        "enable_bonus": True, "bonus_if_fixed_dominate": 0.02,
    }
    fusion_inputs = _build_fusion_inputs(candidates_json, aggregate)
    ranked = doctor_first_gate_rerank(
        candidates=fusion_inputs,
        review_aggregate=aggregate,
        **fusion_params
    )

    ranking = []
    for r in ranked:
        ranking.append({
            "diagnosis": r.get("diagnosis",""),
            "department": r.get("department", []),
            "doctor_conf": float(r.get("doctor_conf", 0.0)),
            "expert_conf": float(r.get("expert_conf", 0.0)),
            "review_score": float(r.get("review_score", 0.0)) if r.get("review_score") is not None else 0.0,
            "score": float(r.get("score", 0.0)),
            "final_rank": int(r.get("final_rank", 0))
        })
    # Decision (dataset mode when truth provided)
    decision_json: Optional[Dict[str, Any]] = None
    if truth is not None:
        decision_json = await decide_with_short_circuit_reuse(ranking, truth, decision_agent)

    # Assemble ONE JSON for this case
    case_json = {
        "meta": {
            "case_id": case_id,
            "timestamp_utc": ts,
            "config": {
                "topk_from_doctor": topk_from_doctor,
                "reviewer_weights": reviewer_weights,
                "reviewer_used": reviewer_used
            }
        },
        "input": {
            "symptoms_text": symptoms_text,
            "patient_history": patient_history
        },
        "general_doctor": gd_dict,          # raw GD output (normalized)
        "candidates": candidates_json,      # raw 3-round dialog per candidate
        "reviewers": reviewers_json,        # raw four reviewers
        "review_aggregate": aggregate,      # list with final_score per diagnosis
        "ranking": ranking,                 # final fused ranking
        "decision": decision_json,          # may be None in single mode
        "status": "ok"
    }
    return case_json

# ========== Database orchestration ==========
async def run_dataset(
    run_dir: Path,
    csv_path: str,
    case_col: str = "symptom",
    truth_col: str = "disease",
    topk_from_doctor: int = 5
) -> None:
    """
        Purpose: Batch-mode runner for a CSV dataset (symptom → disease).
        Inputs:
            run_dir: output directory (stores results + checkpoints).
            csv_path: path to dataset (must have symptom and disease columns).
            case_col: column name for symptoms.
            truth_col: column name for ground truth diseases.
            topk_from_doctor: how many GD candidates to propagate downstream.
        Steps:
            Load CSV into pandas.
            Resume from checkpoint if available (checkpoint.txt).
            Reuse GeneralDoctorAgent + DecisionAgent across all rows for efficiency.
            For each row:
                Run GeneralDoctor for candidates.
                Run Expert–Critic multi-round debate for each candidate.
                Prepare reviewer reports, optionally gated by _should_run_reviewers.
                If triggered: spawn 4 Reviewers → aggregate judgments.
                Apply fusion ranking (doctor-first + reviewer signals).
                Evaluate decision against truth label.
                Save one-line JSON per case (cases.jsonl).
                Maintain resume-safe .tmp snapshots.
            After completion: aggregate metrics into metrics.json.
        Output: JSONL file with per-case results; metrics summary.
    """
    df = pd.read_csv(csv_path)
    n = len(df)
    cases_jsonl = run_dir / "cases.jsonl"
    checkpoint = run_dir / "checkpoint.txt"
    tmp_dir = run_dir

    # Reuse heavy agents (GeneralDoctor, Decision)
    gd_agent = GeneralDoctorAgent()
    decision_agent = DiseaseMatchAgent()

    # checkpoint
    if not checkpoint.exists():
        start_idx = -1
    try:
        start_idx = int(checkpoint.read_text(encoding="utf-8").strip())
    except Exception:
        start_idx = -1

    start = start_idx + 1
    total_left = max(0, n - start)

    for i in tqdm(range(start, n), total=total_left, desc="Dataset", unit="case"):
        row = df.iloc[i]
        case_id = f"row{i}"
        symptoms_text = str(row[case_col])
        patient_history = ""
        truth = str(row[truth_col])

        try:
            ts = datetime.now(timezone.utc).isoformat()

            # 1) General Doctor (reused agent)
            gd_res: DeptDiagnosisResult = await gd_agent.run(
                symptoms_text=symptoms_text,
                patient_history=patient_history
            )
            gd_dict = normalize_gd_dict(gd_res.model_dump())
            items = (gd_dict.get("diagnoses") or [])[:topk_from_doctor]

            # 2) Per-candidate 3 rounds (with early stop)
            candidates_json: List[Dict[str, Any]] = []
            candidate_names: List[str] = []
            reports_for_reviewer: List[List[str]] = []

            for idx_c, item in enumerate(items, start=1):
                dept = item.get("department") or []
                dx = _as_one_name(item.get("diagnose"))
                dconf = _as_one_conf(item.get("confidence"))
                reason = item.get("reasoning") or []
                refs = item.get("reference") or []

                rounds, expert_conf = await run_three_rounds_for_candidate(
                    department=(dept[0] if dept else "General Medicine"),
                    candidate_dx=dx,
                    symptoms_text=symptoms_text,
                    patient_history=patient_history,
                    doctor_reasoning=reason,
                    doctor_reference=refs,
                )

                cand_obj = {
                    "id": f"c{idx_c}",
                    "department": dept,
                    "diagnosis": dx,
                    "doctor_confidence": dconf,
                    "rounds": rounds,
                    "expert_confidence_last_round": float(expert_conf),
                }
                candidates_json.append(cand_obj)

                # Reviewer input (summarized)
                candidate_names.append(dx)
                report_md = build_report_md(dept=dept, dx=dx, doc_reason=reason, doc_refs=refs, rounds=rounds)
                reports_for_reviewer.append([report_md])

                # optional: write compact tmp snapshot for safety
                tmp_payload = {
                    "meta": {"case_id": case_id, "timestamp_utc": ts},
                    "input": {"symptoms_text": symptoms_text, "patient_history": patient_history},
                    "general_doctor": gd_dict,
                    "candidates": candidates_json
                }
                atomic_write_text(tmp_dir / f".case_{slugify(case_id)}.tmp",
                                  json.dumps(tmp_payload, ensure_ascii=False, separators=(",", ":")))

            # 3) (Optional) Reviewers per case (build only if needed by gate)
            reviewers_json = {"accuracy": {}, "coverage": {}, "interpretability": {}, "specificity": {}}
            aggregate = []
            reviewer_used = False  # mark for logging/metrics

            # gate BEFORE building reviewer agents
            need_review = _should_run_reviewers(
                candidates_json=candidates_json,
                p_lock=0.4,  # same as fusion_params (keep consistent)
                margin=0.15
            )

            if candidate_names and need_review:
                reviewer_used = True
                Acc = ReviewAgent(perspective="Accuracy", symptoms=symptoms_text, candidates=candidate_names,
                                  reports=reports_for_reviewer)
                Cov = ReviewAgent(perspective="Coverage", symptoms=symptoms_text, candidates=candidate_names,
                                  reports=reports_for_reviewer)
                Intp = ReviewAgent(perspective="Interpretability", symptoms=symptoms_text, candidates=candidate_names,
                                   reports=reports_for_reviewer)
                Spec = ReviewAgent(perspective="Specificity", symptoms=symptoms_text, candidates=candidate_names,
                                   reports=reports_for_reviewer)

                acc_res = await Acc.run()
                cov_res = await Cov.run()
                intp_res = await Intp.run()
                spec_res = await Spec.run()

                reviewers_json = {
                    "accuracy": acc_res.model_dump(),
                    "coverage": cov_res.model_dump(),
                    "interpretability": intp_res.model_dump(),
                    "specificity": spec_res.model_dump()
                }
                weights = {"accuracy": 0.3, "coverage": 0.3, "interpretability": 0.2, "specificity": 0.2}
                aggregate = aggregate_reviews(
                    reviewers_json["accuracy"], reviewers_json["coverage"],
                    reviewers_json["interpretability"], reviewers_json["specificity"],
                    weights
                )

            # 4) Ranking fusion (Doctor-First gated re-rank)
            fusion_params = {
                "alpha": 0.8,
                "beta": 0.15,
                "gamma": 0.05,
                "p_lock": 0.4,
                "margin": 0.15,
                "swap_tau": 0.2,
                "temperature": 1.0,
                "enable_bonus": True,
                "bonus_if_fixed_dominate": 0.02,
            }

            fusion_inputs = _build_fusion_inputs(candidates_json, aggregate)
            ranked = doctor_first_gate_rerank(
                candidates=fusion_inputs,
                review_aggregate=aggregate,
                **fusion_params
            )
            ranking = []
            for r in ranked:
                ranking.append({
                    "diagnosis": r.get("diagnosis", ""),
                    "department": r.get("department", []),
                    "doctor_conf": float(r.get("doctor_conf", 0.0)),
                    "expert_conf": float(r.get("expert_conf", 0.0)),
                    "review_score": float(r.get("review_score", 0.0)) if r.get("review_score") is not None else 0.0,
                    "score": float(r.get("score", 0.0)),
                    "final_rank": int(r.get("final_rank", 0))
                })

            # 5) Decision with reuse
            decision_json = await decide_with_short_circuit_reuse(
                ranking=ranking,
                truth=truth,
                decision_agent=decision_agent
            )

            # 6) Assemble one-line JSON and append
            case_json = {
                "meta": {
                    "case_id": case_id,
                    "timestamp_utc": ts,
                    "config": {
                        "topk_from_doctor": topk_from_doctor,
                        "reviewer_weights": {"accuracy": 0.3, "coverage": 0.3, "interpretability": 0.2, "specificity": 0.2},
                        "reviewer_used": reviewer_used
                    }
                },
                "input": {"symptoms_text": symptoms_text, "patient_history": patient_history, "truth": truth},
                "general_doctor": gd_dict,
                "candidates": candidates_json,
                "reviewers": reviewers_json,
                "review_aggregate": aggregate,
                "ranking": ranking,
                "decision": decision_json,
                "status": "ok"
            }
            with cases_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(case_json, ensure_ascii=False, separators=(",", ":")) + "\n")

            # cleanup tmp
            tmp_path = tmp_dir / f".case_{slugify(case_id)}.tmp"
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

        except Exception as e:
            err_json = {
                "meta": {"case_id": case_id, "timestamp_utc": datetime.now(timezone.utc).isoformat()},
                "input": {"symptoms_text": symptoms_text, "patient_history": patient_history, "truth": truth},
                "status": "error",
                "error": str(e)
            }
            with cases_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(err_json, ensure_ascii=False, separators=(",", ":")) + "\n")
        # checkpoint after each row (resume-safe)
        tmp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
        tmp.write_text(str(i), encoding="utf-8")
        os.replace(tmp, checkpoint)


# ========== Main ==========
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnostic pipeline (single case or dataset) with one-line JSON outputs.")
    p.add_argument("--run_dir", type=str, default='outputs/runs/demo_full', help="Output run directory.")
    p.add_argument("--case_out_dir", type=str, default='case_out/history_case_rag', help="CaseRAG index dir (history_fused.index etc.)")
    p.add_argument("--umls_sqlite", type=str, default='umls_out/mini_kit.sqlite', help="UMLS sqlite path.")
    p.add_argument("--st_model_name", type=str, default="intfloat/e5-large-v2")

    sub = p.add_subparsers(dest="mode", required=True)

    s1 = sub.add_parser("single", help="Run a single case.")
    s1.add_argument("--case_id", type=str, default="demo_case")
    s1.add_argument("--symptoms", type=str, default="I've been coughing up phlegm and my back hurts. I also feel weak and disoriented, and my neck hurts.")
    s1.add_argument("--history", type=str, default="")
    s1.add_argument("--topk", type=int, default=3)

    s2 = sub.add_parser("dataset", help="Run a CSV dataset with columns: symptom, disease.")
    s2.add_argument("--csv", type=str, default='/home/wz2708/MultiAgents/data/symptom2disease/test.csv')
    s2.add_argument("--case_col", type=str, default="symptom")  # input column name in database
    s2.add_argument("--truth_col", type=str, default="disease") # output column name in database
    s2.add_argument("--topk", type=int, default=3)

    return p.parse_args()

def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # RAG resource init
    ResourcePool.initialize(
        case_out_dir=args.case_out_dir,
        st_model_name=args.st_model_name,
        umls_sqlite_path=args.umls_sqlite
    )

    if args.mode == "single":
        async def _go():
            case_json = await run_single_case(
                case_id=args.case_id,
                symptoms_text=args.symptoms,
                patient_history=args.history,
                topk_from_doctor=args.topk,
                truth=None,         # single mode usually no ground truth
                tmp_dir=None
            )
            # For single mode, just write a readable JSON file besides JSONL if you want
            cases_path = run_dir / "cases.jsonl"
            with cases_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(case_json, ensure_ascii=False, separators=(",", ":")) + "\n")

            print("=== Final ranking ===")
            for r in case_json.get("ranking", []):
                print(
                    f"{r['final_rank']}. {r['diagnosis']}  (score={r['score']}, doctor={r['doctor_conf']}, expert={r['expert_conf']}, review={r['review_score']})")

            print(f"\nAppended one-line JSON to: {cases_path}")

        asyncio.run(_go())
    else:
        async def _go():
            await run_dataset(
                run_dir=run_dir,
                csv_path=args.csv,
                case_col=args.case_col,
                truth_col=args.truth_col,
                topk_from_doctor=args.topk
            )
            print(f"Dataset finished. Artifacts saved to: {run_dir}")
            # Print metrics
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
        asyncio.run(_go())


if __name__ == "__main__":
    main()
