from __future__ import annotations
import asyncio
import sys
import json
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from autogen_core.memory import ListMemory, MemoryContent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from config.settings import llm_settings

from core.util import ResourcePool

from case_core.case_rag import (
    api_critic_single_round1,
    api_critic_single_round2,
)

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# Critic Schema
class CriticResult(BaseModel):
    diagnose: List[str] = Field(..., description="The single diagnosis under critique.")
    critique: List[str] = Field(..., description="New lever-symptom questions prefixed with Q#.")
    review: List[str] = Field(..., description="Status for last round Q&A, each prefixed with A#.")

# Department Critic Agent (RAG-injected)
class DeptCriticAgent(AssistantAgent):
    """
    DeptCriticAgent that:
      - Reuses TeachCOT and CaseRAG/UMLS from ResourcePool.
      - Round 1: retrieve using symptoms + candidate.
      - Later rounds: retrieve using candidate only (disease-focused).
      - Keeps original prompt style and JSON output schema (CriticResult).
    """

    def __init__(self,
                 department: str,
                 candidate: str,
                 symptoms: str,
                 history: str):
        self.department = department
        self.candidate = candidate.strip()
        self.symptoms = symptoms
        self.history = history

        # Shared resources (initialized at process start)
        self.pool = ResourcePool.get()

        # System prompt kept as close to your original as possible
        system_message = f"""
            You are the dedicated {department} Critic Agent for one specific patient.
            Immutable Patient information:
              • Patient Symptoms: “{symptoms}”
              • Patient History: “{history}”
              • Candidate Diagnosis: “{self.candidate}”
              • Previous rounds’ Q&A for traceability.

            Global Rules:
            1. Do NOT propose new diagnoses and ask the doctor questions cannot be solved based on the immutable inputs.
            2. Always extract 2–3 lever symptoms that best distinguish this diagnosis given the patient’s symptoms.
            3. For each lever symptom NOT addressed by the doctor’s reasoning, generate one question prefixed “Q#:”.
            4. Number new questions per round starting from Q1.
            5. Review last Expert responses: for each previous Q#:
                 – mark as “A# fixed” if adequately answered.
                 – “A# not_fixed” if answered but cannot convince you.
                 – “A# cannot_fix” if the expert indicates inability to answer based on the limited knowledge.
                 - please do not force expert give answer to questions they cannot answer based on the current information.
            6. Output ONLY valid JSON matching CriticResult; no extra text.
                diagnose: List[str] = Field(..., description="The single diagnosis under critique.")
                critique: List[str] = Field(..., description="New lever-symptom questions prefixed with Q#.")
                review: List[str] = Field(..., description="Status for last round Q&A, each prefixed with A#.")
            """

        client = OpenAIChatCompletionClient(
            model=llm_settings.openai_model,
            api_key=llm_settings.openai_api_key
        )
        self.memory = ListMemory(name=f"{department}_critic_memory")
        super().__init__(
            name="Critic",
            description=f"Critic Agent for {department}",
            model_client=client,
            system_message=system_message,
            memory=[self.memory],
            output_content_type=CriticResult
        )
        self.round = 1

    async def _build_rag_payload(self, q_pos: List[str]) -> Dict[str, Any]:
        """
        Build RAG payload for the current round:
          - CaseRAG: disease-specific critic API (Round1 vs Round2).
          - UMLS: disease→info; we also compute simple 'missing' = typical - observed (q_pos).
        Returns minimal, clean JSON to feed into the prompt.
        """
        disease = self.candidate

        # CaseRAG (disease-focused)
        if self.round == 1:
            # Use embeddings computed from symptoms + TeachCOT (already done in extract_symptoms)
            feat = await self.pool.extract_symptoms(self.symptoms)
            q_emb_fused, q_emb_raw = feat["q_emb_fused"], feat["q_emb_raw"]
            case_resp = await api_critic_single_round1(
                self.pool.case_ctx, disease, q_pos, q_emb_fused, q_emb_raw, topk=5
            )
        else:
            # Later rounds: focus more narrowly on disease; embeddings unnecessary
            case_resp = await api_critic_single_round2(
                self.pool.case_ctx, disease, q_pos, topk=5
            )

        # Normalize case lines for readability and compactness
        case_lines: List[str] = []
        for c in case_resp.get("candidates", []):
            dx = c.get("disease", "")
            sym = ", ".join(c.get("symptoms", [])[:6])
            cot = "; ".join(c.get("cot", [])[:])  # keep key lines compact
            sc = c.get("score", 0.0)
            case_lines.append(
                f"[History RAG] disease={dx} | score={sc} | typical=[{sym}] | cot=[{cot}]"
            )

        # UMLS disease info for this candidate
        umls_info = self.pool.umls_rag.diseases_to_info([disease], topk_symptoms=8)
        umls_lines: List[str] = []
        for item in umls_info.get("items", []):
            dx = item.get("name", disease)
            typ = item.get("typical_symptoms", []) or []
            typ_str = ", ".join(typ[:8])
            # Compute simple "missing" as typical - observed
            observed = {s.strip().lower() for s in q_pos}
            missing = [s for s in typ if s.strip().lower() not in observed][:4]
            miss_str = ", ".join(missing)
            defin = (item.get("definition", "") or "")[:220]
            umls_lines.append(
                f"[BaseKnowledge RAG] disease={dx} | typical=[{typ_str}] | missing_keys=[{miss_str}] | def={defin}"
            )

        return {
            "History_RAG": case_lines,  # keep original key name in your prompt payload
            "MedQuAD_RAG": umls_lines  # reuse key name; content is from UMLS
        }

    async def run(self, *, doctor_result: Dict) -> CriticResult:
        """
        doctor_result follows your DeptHead/GeneralDoctor output for a single candidate:
            {
              "diagnose": ["..."],  # or str; we use self.candidate anyway
              "confidence": [0.3],
              "reasoning": [...],
              "reference": [...]
            }
        """
        # TeachCOT extraction once (we do not filter; pass all positives)
        feat = await self.pool.extract_symptoms(self.symptoms)
        q_pos = feat["q_pos"]

        # Build RAG payload for this round
        rag_payload = await self._build_rag_payload(q_pos)

        # Assemble the LLM task (stay close to your original prompt)
        payload = {
            "Round": self.round,
            **rag_payload,
            "Candidate": self.candidate,
            "DeptHead_Diagnose": doctor_result.get("diagnose", []),
            "DeptHead_Reasoning": doctor_result.get("reasoning", []),
            # "Unresolved_QIDs": unresolved  # can be wired in your outer loop
        }

        task = (
            f"--- Critic Round {self.round} ---\n"
            f"Context JSON:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
            "Tasks:\n"
            "1. Extract 2–3 lever symptoms that differentiate this candidate.\n"
            "2. For each lever NOT covered in DeptHead_Reasoning, generate question 'Q#:' referencing lever.\n"
            "3. Review Unresolved_QIDs: for each, check Expert's A# response and mark 'A# fixed'/'A# not_fixed'/'A# cannot_fix'.\n"
            "4. All critics you give must be based on the patient's Symptoms, the RAG information only for your reference, but you only responsible for the one patient. \n"
            "Output only CriticResult JSON matching schema."
        )

        # Save round input to memory
        await self.memory.add(MemoryContent(
            content=json.dumps({f"Round{self.round}_Input": payload}, ensure_ascii=False),
            mime_type="application/json"
        ))

        # Call the LLM
        result = await super().run(task=task)
        structured: CriticResult = CriticResult.model_validate(result.messages[-1].content)

        # Save round output to memory
        await self.memory.add(MemoryContent(
            content=json.dumps({f"Round{self.round}_Critic": structured.model_dump()}, ensure_ascii=False),
            mime_type="application/json"
        ))

        # Increment round counter
        self.round += 1
        return structured



# async def run_real():
#     department = "Gastroenterology"
#     # Candidate should be a single disease name
#     candidate = "Peptic Ulcer Disease"
#     symptoms = (
#         "I have a burning sensation in my stomach that comes and goes. "
#         "It's worse when I eat and when I lie down. I also have heartburn and indigestion."
#     )
#     history = "No significant past medical history."
#     doctor_result = {
#         "diagnose": [candidate],
#         "confidence": [0.3],
#         "reasoning": [
#             "1. Burning epigastric pain postprandially consistent with mucosal ulceration.",
#             "2. Heartburn and indigestion often co-occur with PUD.",
#             "3. Lack of alarm features in the provided history."
#         ],
#         "reference": ["[History RAG] typical epigastric burning; [BaseKnowledge RAG] classic presentation"]
#     }
#
#     agent = DeptCriticAgent(
#         department=department,
#         candidate=candidate,
#         symptoms=symptoms,
#         history=history
#     )
#     res: CriticResult = await agent.run(doctor_result=doctor_result)
#     print(json.dumps(res.model_dump(), indent=2, ensure_ascii=False))
#
# if __name__ == "__main__":
#     CASE_OUT_DIR = "case_out/history_case_rag"  # contains history_fused.index, history_raw.index, ids.pkl, history_cases.parquet
#     UMLS_SQLITE = "umls_out/mini_kit.sqlite"  # UMLS mini db path
#
#     ResourcePool.initialize(
#         case_out_dir=CASE_OUT_DIR,
#         st_model_name="intfloat/e5-large-v2",
#         umls_sqlite_path=UMLS_SQLITE
#     )
#     asyncio.run(run_real())