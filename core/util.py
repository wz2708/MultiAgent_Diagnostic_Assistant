from __future__ import annotations
import asyncio
from typing import Optional, Dict, Any, List

from sentence_transformers import SentenceTransformer
from agents.Extraction_agent import TeachCOTAgent

from case_core.build_case_rag import build_fused_str, load_index
from case_core.case_rag import RAGContext, api_depthead
from umls_core.umls_rag import EnhancedWhitelistRAG

class ResourcePool:
    """
    A lightweight singleton-style pool that owns all heavy resources:
    - TeachCOTAgent (for symptom extraction)
    - SentenceTransformer (for CaseRAG query encoding)
    - CaseRAG indices and dataframe
    - UMLS EnhancedWhitelistRAG
    The pool exposes convenience retrieval methods used by multiple agents.
    """
    _instance: Optional["ResourcePool"] = None

    @classmethod
    def get(cls) -> "ResourcePool":
        if cls._instance is None:
            raise RuntimeError("ResourcePool has not been initialized. Call ResourcePool.initialize(...) first.")
        return cls._instance

    @classmethod
    def initialize(
        cls,
        case_out_dir: str,
        st_model_name: str = "intfloat/e5-large-v2",
        umls_sqlite_path: str = "umls_out/mini_kit.sqlite"
    ) -> "ResourcePool":
        if cls._instance is not None:
            return cls._instance

        # TeachCOT is reused across the system (no filtering here; we pass everything "as is")
        student = TeachCOTAgent(mode="student")

        # Sentence-Transformer encoder (shared by CaseRAG)
        st_model = SentenceTransformer(st_model_name)

        # CaseRAG indices and frame
        idx_fused, idx_raw, ids, df = load_index(case_out_dir)
        case_ctx = RAGContext(
            out_dir=case_out_dir,
            st_model=st_model,
            idx_fused=idx_fused,
            idx_raw=idx_raw,
            ids=ids,
            df=df
        )

        # UMLS base-knowledge RAG
        umls_rag = EnhancedWhitelistRAG(umls_sqlite_path)

        cls._instance = cls(
            student=student,
            case_ctx=case_ctx,
            umls_rag=umls_rag
        )
        return cls._instance

    def __init__(self, student: TeachCOTAgent, case_ctx: RAGContext, umls_rag: EnhancedWhitelistRAG):
        self.student = student
        self.case_ctx = case_ctx
        self.umls_rag = umls_rag

    # ---------- High-level retrieval utilities ----------

    async def extract_symptoms(self, free_text: str) -> Dict[str, Any]:
        """
        Use TeachCOT to extract structured symptoms. We do not filter the outputs.
        Returns:
            {
              "q_pos": List[str],
              "q_neg": List[str],
              "fused_q": str,
              "q_emb_fused": np.ndarray[1, D],
              "q_emb_raw": np.ndarray[1, D]
            }
        """

        fx = await self.student.run(text=free_text)
        q_pos, q_neg = fx.positive_symptoms, fx.negative_symptoms
        fused_q = build_fused_str(q_pos, q_neg, ctx=[])
        q_emb_fused = self.case_ctx.st_model.encode([fused_q], normalize_embeddings=True).astype(np.float32)
        q_emb_raw = self.case_ctx.st_model.encode([raw_text], normalize_embeddings=True).astype(np.float32)
        feat = {"q_pos": q_pos, "q_neg": q_neg, "fused_q": fused_q, "q_emb_fused": q_emb_fused, "q_emb_raw": q_emb_raw}
        return feat

    async def case_candidates(self, symptoms_text: str, topk: int = 5) -> Dict[str, Any]:
        """
        CaseRAG department-head style retrieval (already deduplicates per disease).
        Returns a dict with "candidates": List[...], including disease, matched_symptoms, cot, raw_text, score
        """
        feat = await self.extract_symptoms(symptoms_text)
        q_pos, q_emb_fused, q_emb_raw = feat["q_pos"], feat["q_emb_fused"], feat["q_emb_raw"]
        resp = await api_depthead(self.case_ctx, q_pos, q_emb_fused, q_emb_raw, topk=topk)
        return resp

    def umls_candidates(self, symptom_phrases: List[str], topk: int = 2) -> Dict[str, Any]:
        """
        UMLS whitelist RAG over already-extracted symptoms.
        Returns a dict with "candidates": List[...], including name/definition/codes/matched/missing/score
        """
        out = self.umls_rag.symptoms_to_diseases(symptom_phrases, topk=topk)
        return out
