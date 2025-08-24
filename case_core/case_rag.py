import os, re, json, argparse, hashlib, pickle, faiss, asyncio, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from dataclasses import dataclass
from typing import List, Dict, Any

from agents.Extraction_agent import TeachCOTAgent
from sentence_transformers import SentenceTransformer
from case_core.build_case_rag import _tolist, build_fused_str, load_index


@dataclass
class RAGContext:
    '''
        Container for all resources needed during retrieval.
            out_dir: path to saved index/data files
            st_model: SentenceTransformer model used for encoding
            idx_fused: FAISS index for fused strings
            idx_raw: FAISS index for raw text
            ids: list of case IDs (parallel to FAISS index)
            df: dataframe with metadata and case contents
    '''
    out_dir: str
    st_model: SentenceTransformer
    idx_fused: any
    idx_raw: any
    ids: list
    df: pd.DataFrame

# -------------------------------- Helfer Function --------------------------------
def exact_sym_hits(q_pos: List[str], cand_pos: List[str]) -> int:
    # Counts exact overlaps between query positive symptoms and candidate symptoms (case-insensitive).
    # Purpose: enforce stricter symptom matching.
    qs = {str(x).lower().strip() for x in _tolist(q_pos) if x}
    cs = {str(x).lower().strip() for x in _tolist(cand_pos) if x}
    return len(qs & cs)

def _search_fused_raw(ctx: RAGContext, q_emb_fused, q_emb_raw,
                      topk_fused=200, topk_raw=200, w_fused=0.8, w_raw=0.2):
    # Searches both fused index and raw index, combines scores with linear weights.
    # Output: dict {case_idx: combined_score}.
    Df, If = ctx.idx_fused.search(q_emb_fused, topk_fused)
    Dr, Ir = ctx.idx_raw.search(q_emb_raw,   topk_raw)
    cand = {}
    for i, s in zip(If[0], Df[0]): cand[i] = cand.get(i, 0.0) + w_fused*float(s)
    for i, s in zip(Ir[0], Dr[0]): cand[i] = cand.get(i, 0.0) + w_raw  *float(s)
    return cand

def _apply_hit_constraint(ctx: RAGContext, cand_scores: dict, q_pos):
    # Re-ranks candidate rows by applying hit constraint: if <2 symptom overlaps → penalize (×0.8).
    # Output: sorted list of (adjusted_score, row_dict).
    items = []
    for idx_i, sc in cand_scores.items():
        rid = ctx.ids[idx_i]
        row = ctx.df[ctx.df["id"] == rid].iloc[0].to_dict()
        hits = exact_sym_hits(q_pos, row.get("sym_pos", []))
        adj  = float(sc) * (0.8 if hits < 2 else 1.0)
        items.append((adj, row))
    items.sort(key=lambda x: x[0], reverse=True)
    return items

def _jsonify_row(row, q_pos, score):
    # Converts a candidate row into serializable dict with limited fields:
    # {source, disease, symptoms[:8], matched_symptoms, cot[:4], raw_text, score}.
    # Purpose: frontend/demo-friendly output.
    sym_pos = _tolist(row.get("sym_pos", []))
    cot     = _tolist(row.get("cot", []))
    qset    = {str(x).lower() for x in _tolist(q_pos) if x}
    matched = sorted({s for s in sym_pos if str(s).lower() in qset})
    return {
        "source": str(row.get("source","")),
        "disease": str(row.get("disease","")),
        "symptoms": sym_pos[:8],
        "matched_symptoms": matched,
        "cot": cot[:4],
        "raw_text": str(row.get("raw_text","")),
        "score": float(round(float(score), 4)),
    }

# Core API: search function for general doctor
async def api_depthead(ctx: RAGContext, q_pos, q_emb_fused, q_emb_raw, topk:int=5):
    '''
    Simulates the Department Head: retrieves top-k diverse diseases from historical cases.
        Searches both fused/raw indices.
        Applies hit constraint.
        Deduplicates by disease.
    Output: dict with query symptoms and candidate list.
    '''
    cand = _search_fused_raw(ctx, q_emb_fused, q_emb_raw, topk_fused=50, topk_raw=50)
    items = _apply_hit_constraint(ctx, cand, q_pos)

    seen, picked = set(), []
    for sc, row in items:
        dx = str(row.get("disease","")).lower()
        if dx in seen: continue
        seen.add(dx); picked.append((sc, row))
        if len(picked) >= topk: break

    return {
        "query_symptoms": _tolist(q_pos),
        "candidates": [_jsonify_row(r, q_pos, sc) for sc, r in picked]
    }

# Core API: search function for expert round1
async def api_expert_single_round1(ctx: RAGContext, disease: str,
                                   q_pos, q_emb_fused, q_emb_raw,
                                   topk: int = 5):
    '''
        Simulates Expert Agent, Round 1:
            Retrieves only rows for the given disease.
            Re-scores by adding a bonus proportional to symptom overlap (+0.04*hits).
            Picks top-k.
        Output: JSON with candidates supporting the given disease
    '''
    disease_l = disease.strip().lower()
    cand = _search_fused_raw(ctx, q_emb_fused, q_emb_raw, topk_fused=200, topk_raw=200)
    items = _apply_hit_constraint(ctx, cand, q_pos)

    picked = []
    for sc, row in items:
        if str(row.get("disease","")).strip().lower() != disease_l:
            continue
        hits = exact_sym_hits(q_pos, row.get("sym_pos", []))
        sc2  = sc * (1.0 + 0.04*hits)
        picked.append((sc2, row))
    picked.sort(key=lambda x: x[0], reverse=True)
    picked = picked[:topk]

    return {
        "query_symptoms": _tolist(q_pos),
        "candidates": [_jsonify_row(r, q_pos, sc) for sc, r in picked]
    }

# Core API: search function for critic round1
async def api_critic_single_round1(ctx: RAGContext, disease: str,
                                   q_pos, q_emb_fused, q_emb_raw,
                                   topk: int = 5):
    '''
    Simulates Critic Agent, Round 1:
        Retrieves only rows for the given disease.
        Ranks candidates by fewer symptom hits first (prefers challenging cases), then by score.
        Picks top-k.
    Output: JSON with candidates challenging the disease
    '''
    disease_l = disease.strip().lower()
    cand = _search_fused_raw(ctx, q_emb_fused, q_emb_raw, topk_fused=200, topk_raw=200)
    items = _apply_hit_constraint(ctx, cand, q_pos)

    bucket = []
    for sc, row in items:
        if str(row.get("disease","")).strip().lower() != disease_l:
            continue
        hits = exact_sym_hits(q_pos, row.get("sym_pos", []))

        bucket.append((hits, -sc, sc, row))
    bucket.sort(key=lambda x: (x[0], x[1]))
    picked = [(sc, row) for (_, _, sc, row) in bucket[:topk]]

    return {
        "query_symptoms": _tolist(q_pos),
        "candidates": [_jsonify_row(r, q_pos, sc) for sc, r in picked]
    }

# Round 2 Utilities
def _disease_rows(ctx: RAGContext, disease: str):
    # Extracts all rows from dataframe for a given disease (case-insensitive).
    # Output: list of dicts.
    disease_l = disease.strip().lower()
    sub = ctx.df[ctx.df["disease"].str.lower() == disease_l]
    return [r._asdict() if hasattr(r, "_asdict") else r.to_dict() for _, r in sub.iterrows()]

def _greedy_diverse_pick(rows: list, topk: int):
    # Greedy selection ensuring diversity of symptoms/CoT coverage.
    # Iteratively picks cases that maximize new symptom coverage.
    # Output: list of picked rows.
    picked, covered = [], set()
    scored = []
    for r in rows:
        sym = {str(s).strip().lower() for s in _tolist(r.get("sym_pos", [])) if s}
        richness = len(sym) + 0.5*len(_tolist(r.get("cot", [])))
        scored.append((richness, r, sym))
    scored.sort(key=lambda x: x[0], reverse=True)

    for _, r, sym in scored:
        gain = len(sym - covered)
        if gain <= 0 and len(picked) >= 1:
            continue
        picked.append(r); covered |= sym
        if len(picked) >= topk:
            break
    return picked

async def api_expert_single_round2(ctx: RAGContext, disease: str, q_pos, topk: int = 5):
    '''
        Expert Round 2:
            Retrieves all cases of disease.
            Sorts by information richness (symptom count + 0.5 × CoT length).
            Applies greedy-diverse pick.
            Uses information richness as score.
        Output: JSON candidates.
    '''
    rows = _disease_rows(ctx, disease)
    # Pre-sort by information amount, then do greedy diversity
    rows.sort(key=lambda r: (len(_tolist(r.get("sym_pos", []))) + 0.5*len(_tolist(r.get("cot", [])))), reverse=True)
    picked = _greedy_diverse_pick(rows, topk)

    # Use "information content" as an approximate score display
    result = []
    for r in picked:
        sc = len(_tolist(r.get("sym_pos", []))) + 0.5*len(_tolist(r.get("cot", [])))
        result.append(_jsonify_row(r, q_pos, sc))
    return {"query_symptoms": _tolist(q_pos), "candidates": result}

async def api_critic_single_round2(ctx: RAGContext, disease: str, q_pos, topk: int = 5):
    '''
        Critic Round 2:
            Retrieves all cases of disease.
            Builds bucket with (symptom_hits, -info, info).
            Fewer hits prioritized (penalizes overlap), then more information.
            Picks top-k.
        Output: JSON candidates.
    '''
    rows = _disease_rows(ctx, disease)
    bucket = []
    for r in rows:
        hits = exact_sym_hits(q_pos, r.get("sym_pos", []))
        info = len(_tolist(r.get("sym_pos", []))) + 0.5*len(_tolist(r.get("cot", [])))
        # Fewer hits are preferred, followed by more information.
        bucket.append((hits, -info, info, r))
    bucket.sort(key=lambda x: (x[0], x[1]))
    picked = [(info, r) for (_, _, info, r) in bucket[:topk]]

    return {
        "query_symptoms": _tolist(q_pos),
        "candidates": [_jsonify_row(r, q_pos, sc) for sc, r in picked]
    }


# async def main_demo():
#     out_dir = "case_out/history_case_rag"
#
#     # Load index/model/agent at one time
#     idx_fused, idx_raw, ids, df = load_index(out_dir)
#     st_model = SentenceTransformer("intfloat/e5-large-v2")
#     student  = TeachCOTAgent(mode="student")
#
#     # Build RAGContext
#     ctx = RAGContext(out_dir=out_dir, st_model=st_model,
#                      idx_fused=idx_fused, idx_raw=idx_raw, ids=ids, df=df)
#
#     query   = "I have back pain, productive cough, and dizziness."
#     disease = "pneumonia"
#
#     # One-time extraction + encoding
#     fx = await student.run(text=query)
#     q_pos, q_neg = fx.positive_symptoms, fx.negative_symptoms
#     fused_q = build_fused_str(q_pos, q_neg, ctx=[])
#
#     q_emb_fused = st_model.encode([fused_q], normalize_embeddings=True).astype(np.float32)
#     q_emb_raw   = st_model.encode([raw_text], normalize_embeddings=True).astype(np.float32)
#
#     # deptHead
#     resp1 = await api_depthead(ctx, q_pos, q_emb_fused, q_emb_raw, topk=5)
#     print(json.dumps(resp1, indent=2, ensure_ascii=False))
#
#     # Expert Round1
#     e1 = await api_expert_single_round1(ctx, disease, q_pos, q_emb_fused, q_emb_raw, topk=5)
#     print("[Expert R1]"); print(json.dumps(e1, indent=2, ensure_ascii=False))
#
#     # Critic Round1
#     c1 = await api_critic_single_round1(ctx, disease, q_pos, q_emb_fused, q_emb_raw, topk=5)
#     print("[Critic R1]"); print(json.dumps(c1, indent=2, ensure_ascii=False))
#
#     # Expert Round2
#     e2 = await api_expert_single_round2(ctx, disease, q_pos, topk=5)
#     print("[Expert R2]"); print(json.dumps(e2, indent=2, ensure_ascii=False))
#
#     # Critic Round2
#     c2 = await api_critic_single_round2(ctx, disease, q_pos, topk=5)
#     print("[Critic R2]"); print(json.dumps(c2, indent=2, ensure_ascii=False))
#
# if __name__ == "__main__":
#     asyncio.run(main_demo())
