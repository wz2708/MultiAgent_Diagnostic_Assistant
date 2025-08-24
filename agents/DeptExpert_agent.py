from __future__ import annotations
import asyncio
import sys
import json
from typing import List, Dict, Any

from pydantic import BaseModel, Field
from autogen_core.memory import ListMemory, MemoryContent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from config.settings import llm_settings

from core.util import ResourcePool

from case_core.case_rag import (
    api_expert_single_round1,
    api_expert_single_round2,
)

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# DeptExpert Schema
class DeptExpertResult(BaseModel):
    diagnose: List[str] = Field(..., description="Single diagnosis name")
    confidence: List[float] = Field(..., description="Confidence for this diagnosis")
    response: List[str] = Field(..., description="Answers to Critic Q#, each starting with 'A#:'")
    reference: List[str] = Field(..., description="Evidence tags and snippets for each answer")


# DeptExpertAgent (RAG-injected)
class DeptExpertAgent(AssistantAgent):
    """
    Expert Agent refactored to use shared RAG resources:
      - Round 1: CaseRAG api_expert_single_round1(symptoms embeddings + disease)
      - Later rounds: api_expert_single_round2(disease only + q_pos)
      - UMLS: diseases_to_info for supportive typical/missing-key symptoms
    Prompts are minimally changed; output schema remains DeptExpertResult.
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
        self.pool = ResourcePool.get()

        system_message = f"""
            You are a {department} specialist responsible for one specific patient.
            Immutable Patient information:
             • Symptoms: “{symptoms}”
             • History: “{history}”
             • Candidate Diagnosis: “{self.candidate}”
             • CriticFeedback stored as JSON under key "Round{{n}}_Critic"

            Rules:
            1. Do NOT propose new diagnoses.
            2. For each Critic question "Q#...", respond with "A#:" in the same order.
            3. Optionally adjust confidence (single float, sum across candidates = 1.0).
            4. For each answer, add a reference tagged with [History RAG], [Knowledge RAG].
            5. Output ONLY JSON matching ExpertResult schema.
                diagnose: List[str] = Field(..., description="Single diagnosis name")
                confidence: List[float] = Field(..., description="Confidence for this diagnosis")
                response: List[str] = Field(..., description="Answers to Critic Q#, each starting with 'A#:'")
                reference: List[str] = Field(..., description="Evidence tags and snippets for each answer")
            """

        client = OpenAIChatCompletionClient(
            model=llm_settings.openai_model,
            api_key=llm_settings.openai_api_key
        )
        self.memory = ListMemory(name=f"{department}_expert_memory")
        super().__init__(
            name=f"{department.replace(' ', '_')}_DeptExpert",
            description=f"Expert Agent for {self.candidate}",
            model_client=client,
            system_message=system_message,
            memory=[self.memory],
            output_content_type=DeptExpertResult
        )
        self.round = 1

    async def _build_rag_payload(self, q_pos: List[str]) -> Dict[str, List[str]]:
        """
        Prepare compact, labeled evidence lines for the LLM:
          - HistoryRAG: CaseRAG expert APIs (round1 vs round2)
          - MedQuADRAG: from UMLS diseases_to_info(candidate)
        Returns a dict with keys 'HistoryRAG' and 'MedQuADRAG' (to minimize prompt changes).
        """
        disease = self.candidate

        # CaseRAG — expert view per round
        if self.round == 1:
            feat = await self.pool.extract_symptoms(self.symptoms)
            q_emb_fused, q_emb_raw = feat["q_emb_fused"], feat["q_emb_raw"]
            case_resp = await api_expert_single_round1(
                self.pool.case_ctx, disease, q_pos, q_emb_fused, q_emb_raw, topk=5
            )
        else:
            case_resp = await api_expert_single_round2(
                self.pool.case_ctx, disease, q_pos, topk=5
            )

        history_lines: List[str] = []
        for c in case_resp.get("candidates", []):
            dx = c.get("disease", "")
            typical = ", ".join(c.get("symptoms", [])[:6])
            matched = ", ".join(c.get("matched_symptoms", [])[:6])
            cot = "; ".join(c.get("cot", [])[:])  # keep key lines concise
            sc = c.get("score", 0.0)
            history_lines.append(
                f"[History RAG] disease={dx} | score={sc} | matched=[{matched}] | typical=[{typical}] | cot=[{cot}]"
            )

        # UMLS — supportive typical & definition; compute missing keys relative to q_pos
        umls_info = self.pool.umls_rag.diseases_to_info([disease], topk_symptoms=8)
        medquad_lines: List[str] = []
        observed = {s.strip().lower() for s in q_pos}
        for item in umls_info.get("items", []):
            dx = item.get("name", disease)
            typ = item.get("typical_symptoms", []) or []
            miss = [s for s in typ if s.strip().lower() not in observed][:4]
            typ_str = ", ".join(typ[:8])
            miss_str = ", ".join(miss)
            defin = (item.get("definition", "") or "")[:220]
            medquad_lines.append(
                f"[Knowledge RAG] disease={dx} | typical=[{typ_str}] | missing_keys=[{miss_str}] | def={defin}"
            )

        return {"HistoryRAG": history_lines, "KnowledgeRAG": medquad_lines}

    async def run(self, *, critic_feedback: Dict) -> DeptExpertResult:
        """
        Args:
            critic_feedback: dict with keys
              - "diagnose": str
              - "critique": List[str]  # Q#...
              - "review":   List[str]  # A# fixed/not_fixed/cannot_fix
        Returns:
            DeptExpertResult JSON.
        """
        # Extract q_pos once (no filtering)
        feat = await self.pool.extract_symptoms(self.symptoms)
        q_pos = feat["q_pos"]

        # Build RAG payload
        rag_payload = await self._build_rag_payload(q_pos)

        # Build task input structure for the LLM
        prompt_struct = {
            "Round": self.round,
            "Candidate": self.candidate,
            "HistoryRAG": rag_payload["HistoryRAG"],
            "KnowledgeRAG": rag_payload["KnowledgeRAG"],
            "CriticQuestions": critic_feedback.get("critique", []),
            "CriticFeedBack": critic_feedback.get("review", []),
        }

        # Save input to memory
        await self.memory.add(MemoryContent(
            content=json.dumps({f"Round{self.round}_Input": prompt_struct}, ensure_ascii=False),
            mime_type="application/json"
        ))

        # Compose task (keep your original instructions)
        task = (
            f"Task Input (JSON):\n{json.dumps(prompt_struct, ensure_ascii=False)}\n\n"
            "Instructions:\n"
            "For each CriticQuestions[i] (Q#), produce a response A# in response field.\n"
            "All response you give must be based on the patient's Symptoms, the RAG information only for your reference, but you only responsible for the one patient. \n"
            "If you lack evidence, respond: \"A#: I do not have evidence regarding Q#.\"\n"
            "Optionally revise the confidence for this candidate (float between 0.1–0.9).\n"
            "You should continue to improve answer or respond 'Cannot respond within current information' if you find any answer with tag 'not_fixed' in CriticFeedBack."
            "Output ONLY JSON matching ExpertResult."
        )

        # Call LLM
        resp = await super().run(task=task)
        expert: DeptExpertResult = DeptExpertResult.model_validate(resp.messages[-1].content)

        # Save output to memory
        await self.memory.add(MemoryContent(
            content=json.dumps({f"Round{self.round}_Response": expert.model_dump()}, ensure_ascii=False),
            mime_type="application/json"
        ))

        self.round += 1
        return expert


# async def run_real():
#     department = "Gastroenterology"
#     candidate = "Peptic Ulcer Disease"
#     symptoms = (
#         "Burning epigastric pain worsened by meals and when lying down; heartburn; nausea; nocturnal symptoms."
#     )
#     history = "No significant past medical history."
#
#     critic_feedback = {
#         "diagnose": candidate,
#         "critique": [
#             "Q1: Does the patient experience loss of appetite or fatigue associated with the stomach pain?",
#             "Q2: Is there any relief of the burning stomach pain after eating, or any timing pattern such as morning pain relieved by food?",
#             "Q3: Does the patient have symptoms of increased hunger, stomach cramps, or postprandial bloating and gas?"
#
#         ],
#         "review": []
#     }
#
#     agent = DeptExpertAgent(
#         department=department,
#         candidate=candidate,
#         symptoms=symptoms,
#         history=history
#     )
#     res: DeptExpertResult = await agent.run(critic_feedback=critic_feedback)
#     print(json.dumps(res.model_dump(), ensure_ascii=False, indent=2))
#
# if __name__ == "__main__":
#     CASE_OUT_DIR = "case_out/history_case_rag"  # history_fused.index, history_raw.index, ids.pkl, history_cases.parquet
#     UMLS_SQLITE = "umls_out/mini_kit.sqlite"  # UMLS mini DB path
#
#     ResourcePool.initialize(
#         case_out_dir=CASE_OUT_DIR,
#         st_model_name="intfloat/e5-large-v2",
#         umls_sqlite_path=UMLS_SQLITE
#     )
#
#     asyncio.run(run_real())