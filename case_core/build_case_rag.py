import os, re, json, argparse, hashlib, pickle, faiss, asyncio, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from dataclasses import dataclass
from typing import List, Dict, Any

from agents.Extraction_agent import TeachCOTAgent
from sentence_transformers import SentenceTransformer

CTX_MAP = {
    # meds
    r"\b(omeprazole|ppi)\b": "rx_omeprazole",
    r"\b(nsaid|ibuprofen|naproxen)\b": "rx_nsaid_recent",
    r"\b(antibiotic|azithromycin|amoxicillin)\b": "rx_antibiotic",
    # tests
    r"\b(chest\s*x[- ]?ray|cxr)\b": "test_chest_xray",
    r"\b(ct|mri)\b": "test_imaging",
    r"\b(stool (test|culture))\b": "test_stool_culture",
    r"\b(h[_-]?pylori)\b": "test_h_pylori",
    # exposures
    r"\b(travel).*(africa|asia|rural)\b": "hx_travel_risk",
    r"\b(sick contact|close contact)\b": "hx_sick_contact",
}

# ----------------------- Utility Functions -----------------------
def _tolist(x):
    '''
    Converts various input types (None, list, numpy array, pandas Series, str/bytes, iterable) into a Python list.
    Input: arbitrary type.
    Output: list of elements (or [x] fallback).
    '''
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, (np.ndarray, pd.Series)):
        return x.tolist()
    if isinstance(x, (str, bytes)):
        return [x.decode() if isinstance(x, bytes) else x]
    try:
        return list(x)
    except Exception:
        return [x]

def extract_ctx_tokens(text: str) -> List[str]:
    '''
    Extracts predefined context tokens (medications, tests, exposures) from free text using regex patterns in CTX_MAP.
    Input: patient raw text.
    Output: sorted list of matched context tags (e.g., ["rx_antibiotic", "test_chest_xray"]).
    '''
    t = (text or "").lower()
    found = set()
    for pat, tag in CTX_MAP.items():
        if re.search(pat, t):
            found.add(tag)
    return sorted(found)

def build_fused_str(sym_pos: List[str], sym_neg: List[str], ctx: List[str]) -> str:
    '''
    Constructs a canonical "fused string" from extracted positive/negative symptoms and contextual tags.
    Format: [sym+]fever [sym-]no_cough [ctx]rx_antibiotic.
    Output: single fused string (used for embedding & indexing).
    '''
    parts = []
    parts += [f"[sym+]{s}" for s in sym_pos]
    parts += [f"[sym-]{s}" for s in sym_neg]
    parts += [f"[ctx]{c}" for c in ctx]
    return " ".join(parts).strip()

def case_id(text: str, disease: str, source: str) -> str:
    '''
    Generates a unique, deterministic case identifier from (text, disease, source) using MD5 hash.
    Purpose: stable deduplication across datasets.
    '''
    h = hashlib.md5()
    h.update((source + "||" + (text or "") + "||" + (disease or "")).encode("utf-8"))
    return h.hexdigest()

# ----------------------- Core Class -----------------------
class STIndex:
    '''
        Wrapper around a sentence transformer + FAISS index.
        __init__(model_name): load sentence transformer.
        encode(texts, batch_size): return normalized embeddings (np.array).
        new_ip_index(dim): create FAISS inner-product index of dimension dim.
    '''
    def __init__(self, model_name="intfloat/e5-large-v2"):
        self.model = SentenceTransformer(model_name)
        self.index_fused = None
        self.index_raw   = None

    def encode(self, texts, batch_size=64):
        embs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embs, dtype=np.float32)

    @staticmethod
    def new_ip_index(dim: int):
        return faiss.IndexFlatIP(dim)

def _encode_and_build_indices(full_df: pd.DataFrame, out_dir: str, batch_size: int):
    # Helper: encode all cases into embeddings, build dual FAISS indices (fused vs raw), dump to disk (history_fused.index, history_raw.index, ids.pkl).
    full_df = full_df.sort_values("id").reset_index(drop=True)

    fused = full_df["fused_str"].fillna("").astype(str).tolist()
    raws  = full_df["raw_text"].fillna("").astype(str).tolist()

    emb_fused = STIndex(model_name="intfloat/e5-large-v2").encode(fused, batch_size=batch_size)
    emb_raw   = STIndex(model_name="intfloat/e5-large-v2").encode(raws,  batch_size=batch_size)
    dim = emb_fused.shape[1]

    idx_fused = STIndex.new_ip_index(dim); idx_fused.add(emb_fused)
    idx_raw   = STIndex.new_ip_index(dim); idx_raw.add(emb_raw)

    faiss.write_index(idx_fused, os.path.join(out_dir, "history_fused.index"))
    faiss.write_index(idx_raw,   os.path.join(out_dir, "history_raw.index"))
    with open(os.path.join(out_dir, "ids.pkl"), "wb") as f:
        pickle.dump(full_df["id"].tolist(), f)

    save_unified(full_df, out_dir)
    print(f"[OK] dual indices saved: fused={idx_fused.ntotal}, raw={idx_raw.ntotal}")

# ----------------------- Data Load/Save -----------------------
def load_train_csv(path_csv: str, source: str = "train") -> pd.DataFrame:
    # Loads a CSV of (symptom, disease) pairs. Drops NAs, attaches source column, normalizes into schema: [source, raw_text, disease].
    df = pd.read_csv(path_csv)
    df = df.dropna(subset=["symptom", "disease"]).copy()
    df["source"] = source
    df["raw_text"] = df["symptom"].astype(str)
    return df[["source", "raw_text", "disease"]]

def save_unified(df: pd.DataFrame, out_dir: str):
    # Saves a unified case dataframe into both parquet (history_cases.parquet) and JSONL (history_cases.jsonl) formats.
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(os.path.join(out_dir, "history_cases.parquet"), index=False)
    df.to_json(os.path.join(out_dir, "history_cases.jsonl"), lines=True, orient="records", force_ascii=False)

def load_index(out_dir: str):
    # Loads previously built FAISS indices (history_fused.index, history_raw.index), ID map, and metadata dataframe.
    # Output: (idx_fused, idx_raw, ids, df).
    idx_fused = faiss.read_index(os.path.join(out_dir, "history_fused.index"))
    idx_raw   = faiss.read_index(os.path.join(out_dir, "history_raw.index"))
    with open(os.path.join(out_dir, "ids.pkl"), "rb") as f:
        ids = pickle.load(f)
    df = pd.read_parquet(os.path.join(out_dir, "history_cases.parquet"))
    return idx_fused, idx_raw, ids, df


# ----------------------- Teacher Extraction -----------------------
async def run_teacher_batch(rows: List[Dict[str,Any]], concurrency: int = 8) -> List[Dict[str,Any]]:
    '''
    Runs a TeachCOTAgent in teacher mode over a batch of training rows.
    Extracts positive/negative symptoms, chain-of-thought (CoT) reasoning, and builds fused strings.
    Input: list of dict rows {raw_text, disease, source}.
    Output: enriched dicts with {id, sym_pos, sym_neg, cot, ctx, fused_str}.
    Concurrency: uses asyncio + semaphore to limit parallelism.
    '''
    agent = TeachCOTAgent(mode="teacher")

    sem = asyncio.Semaphore(concurrency)
    results = []

    async def _one(r):
        async with sem:
            try:
                out = await agent.run(text=r["raw_text"], gold=r["disease"])
                sym_pos = out.positive_symptoms or []
                sym_neg = out.negative_symptoms or []
                cot     = out.cot_summary or []
                ctx     = extract_ctx_tokens(r["raw_text"] + " " + " ".join(cot))
                fused   = build_fused_str(sym_pos, sym_neg, ctx)
                rid     = case_id(r["raw_text"], r["disease"], r["source"])
                return {
                    "id": rid,
                    "source": r["source"],
                    "disease": r["disease"],
                    "raw_text": r["raw_text"],
                    "sym_pos": sym_pos,
                    "sym_neg": sym_neg,
                    "cot": cot,
                    "ctx": ctx,
                    "facet_str": "; ".join(sym_pos),
                    "fused_str": fused,
                }
            except Exception as e:
                return {"error": str(e), **r}

    tasks = [asyncio.create_task(_one(r)) for r in rows]
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Teacher extracting"):
        results.append(await fut)
    return results

# ----------------------- Index Build/Update AND Query Demo -----------------------
async def build_or_update(args):
    '''
    Main entry point for building or incrementally updating the historical case index.
    Steps:
        Load symptom/conv training CSVs.
        Deduplicate by case_id.
        Run teacher extraction on new rows.
        Merge with existing data (if --append).
        Save unified dataframe.
        Encode fused/raw strings → build FAISS indices.
        Run lightweight audit.
    '''
    # 1) Setting up
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    hist_path = os.path.join(out_dir, "history_cases.parquet")

    frames = []
    if args.symptom_csv and os.path.exists(args.symptom_csv):
        df1 = load_train_csv(args.symptom_csv, "symptom2disease")  # -> [source, raw_text, disease]
        frames.append(df1)
    if args.conversation_csv and os.path.exists(args.conversation_csv):
        df2 = load_train_csv(args.conversation_csv, "conversation2disease")
        frames.append(df2)

    if not frames:
        print("No input rows.")
        return

    base = pd.concat(frames, ignore_index=True)
    base = base.dropna(subset=["raw_text","disease"]).reset_index(drop=True)
    base["id"] = base.apply(lambda r: case_id(r["raw_text"], r["disease"], r["source"]), axis=1)
    base = base.drop_duplicates(subset=["id"], keep="first")

    # 2) Increment: remove existing id
    exist_df = None
    existing_ids = set()
    if args.append and os.path.exists(hist_path):
        exist_df = pd.read_parquet(hist_path)
        if "id" not in exist_df.columns:
            raise RuntimeError("existing parquet missing 'id' column")
        exist_df = exist_df.drop_duplicates(subset=["id"], keep="last")
        existing_ids = set(exist_df["id"].tolist())

    to_process = base[~base["id"].isin(existing_ids)].copy()
    if to_process.empty:
        print("Nothing new to process.")
        if exist_df is not None and getattr(args, "reindex_if_no_new", False):
            full_df = exist_df.copy()
            _encode_and_build_indices(full_df, out_dir, args.batch_size)
        return

    # 3) Extract Teachers in Batch
    # Expect run_teacher_batch(inputs) to return: [{id, sym_pos, sym_neg, cot}, ...]
    inputs = to_process[["id","raw_text","disease","source"]].to_dict("records")
    extracted_list = await run_teacher_batch(inputs, concurrency=args.concurrency)
    extracted_list = [x for x in extracted_list if isinstance(x, dict) and "id" in x and "error" not in x]
    extr = pd.DataFrame(extracted_list)
    if extr.empty:
        print("Teacher batch produced no rows; abort.")
        return

    # 4) Primary key merge
    merged_new = to_process.merge(
        extr[["id","sym_pos","sym_neg","cot"]],
        on="id", how="left"
    )

    # Build fused_str (normalize: None → [], then build)
    merged_new["sym_pos"] = merged_new["sym_pos"].map(_tolist)
    merged_new["sym_neg"] = merged_new["sym_neg"].map(_tolist)
    merged_new["cot"]     = merged_new["cot"].map(_tolist)
    merged_new["fused_str"] = merged_new.apply(
        lambda r: build_fused_str(r["sym_pos"], r["sym_neg"], ctx=[]),
        axis=1
    )

    # 5) Unify the full table
    if exist_df is not None:
        full_df = pd.concat([exist_df, merged_new], ignore_index=True)
        full_df = full_df.drop_duplicates(subset=["id"], keep="last")
    else:
        full_df = merged_new.copy()

    full_df = full_df.sort_values("id").reset_index(drop=True)

    # 6) Save the unified file
    save_unified(full_df, out_dir)

    print(f"[OK] Unified saved: {len(full_df)} rows → {out_dir}")

    # 7) Encoding + Double Indexing
    _encode_and_build_indices(full_df, out_dir, args.batch_size)


async def query_demo(args):
    '''
    Runs a demo retrieval pipeline for a query case.
    Steps:
        Load FAISS indices + metadata.
        Use TeachCOTAgent in student mode to extract query symptoms.
        Build fused query string + encode.
        Search both fused index (80%) and raw index (20%).
        Adjust scores based on exact symptom overlap (hard constraint).
        Return ranked candidate cases with {source, disease, symptoms, cot, score}.
    Output: printed JSON summary.
    '''
    st_model = SentenceTransformer("intfloat/e5-large-v2")

    # 1) Download index/data
    idx_fused, idx_raw, ids, df = load_index(args.out_dir)

    # 2) Student Extraction → fused_query
    agent = TeachCOTAgent(mode="student")
    out = await agent.run(text=args.query_text)
    q_pos, q_neg = out.positive_symptoms, out.negative_symptoms
    q_ctx = extract_ctx_tokens(args.query_text)
    fused_q = build_fused_str(q_pos, q_neg, q_ctx)

    # 3) Encoding
    q_emb_fused = st_model.encode([fused_q], normalize_embeddings=True)
    q_emb_raw = st_model.encode([args.query_text], normalize_embeddings=True)

    # 4) Bidirectional search
    k = args.topk
    Df, If = idx_fused.search(q_emb_fused.astype(np.float32), k)
    Dr, Ir = idx_raw.search(q_emb_raw.astype(np.float32), k)

    # 5) Fusion (simple linear): score = 0.8*fused + 0.2*raw
    cand = {}
    for i, s in zip(If[0], Df[0]): cand[i] = cand.get(i, 0.0) + 0.8 * float(s)
    for i, s in zip(Ir[0], Dr[0]): cand[i] = cand.get(i, 0.0) + 0.2 * float(s)

    # 6) Hard constraint: Positive symptom exact hit count < 2 → 0.8 coefficient
    def exact_hits(query_pos, cand_pos) -> int:
        if query_pos is None:
            query_pos = []
        if cand_pos is None:
            cand_pos = []
        query_pos = _tolist(query_pos)
        cand_pos = _tolist(cand_pos)

        qs = {str(x).lower().strip() for x in query_pos if x}
        cs = {str(x).lower().strip() for x in cand_pos if x}
        return len(qs & cs)

    items = []
    for idx_i, sc in cand.items():
        rid = ids[idx_i]
        row = df[df["id"] == rid].iloc[0].to_dict()
        hits = exact_hits(q_pos, row.get("sym_pos", []))
        adj = float(sc) * (0.8 if hits < 2 else 1.0)
        items.append((adj, row))
    items.sort(key=lambda x: x[0], reverse=True)

    # 7) Output
    result = []
    q_pos = _tolist(q_pos)
    qset = {str(x).lower() for x in q_pos if x}

    for adj, row in items[:args.topk]:
        sym_pos = _tolist(row.get("sym_pos", []))
        ctx = _tolist(row.get("ctx", []))
        cot = _tolist(row.get("cot", []))

        matched = sorted({s for s in sym_pos if str(s).lower() in qset})

        result.append({
            "source": str(row.get("source", "")),
            "disease": str(row.get("disease", "")),
            "symptoms": sym_pos[:8],
            "cot": cot[:4],
            "score": float(round(float(adj), 4)),
        })

    # The top level also ensures serializability
    print(json.dumps({
        "query_symptoms": _tolist(q_pos),
        "candidates": result
    }, indent=2, ensure_ascii=False))

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build/rebuild or incrementally update the historical case index")
    b.add_argument("--symptom_csv", default="data/symptom2disease/train.csv")
    b.add_argument("--conversation_csv", default="data/conversation2disease/train.csv")
    b.add_argument("--out_dir", default="case_out/history_case_rag")
    b.add_argument("--append", action="store_true", help="Incremental: keep existing samples and only append new samples")
    b.add_argument("--concurrency", type=int, default=8)
    b.add_argument("--batch_size", type=int, default=32)

    q = sub.add_parser("query", help="Query demo (Student extraction → DPR-Q → retrieval)")
    q.add_argument("--out_dir", default="data/history_case_rag")
    q.add_argument("--query_text", required=True)
    q.add_argument("--topk", type=int, default=5)

    args = p.parse_args()
    if args.cmd == "build":
        asyncio.run(build_or_update(args))
    elif args.cmd == "query":
        asyncio.run(query_demo(args))

if __name__ == "__main__":
    main()