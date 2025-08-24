from __future__ import annotations
import asyncio
import json
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field

from autogen_core.memory import ListMemory, MemoryContent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from config.settings import llm_settings

from core.util import ResourcePool

# General Doctor Schema
class DeptDiagnosisItem(BaseModel):
    department: List[str] = Field(..., description="Department names.")
    diagnose: List[str] = Field(..., description="List of single disease names; each item contains exactly one name.")
    confidence: List[float] = Field(..., description="A single confidence per item; all confidences sum to 1.0 across candidates.")
    reasoning: List[str] = Field(..., description="Numbered Chain-of-Thought style reasoning steps leveraging RAG evidence.")
    reference: List[str] = Field(..., description="Tagged citations, e.g., '[History RAG] ...', '[BaseKnowledge RAG] ...'")

class DeptDiagnosisResult(BaseModel):
    diagnoses: List[DeptDiagnosisItem] = Field(..., description="We will output up to 5 candidates as requested.")

# General Doctor Agent
class GeneralDoctorAgent(AssistantAgent):
    """
    GeneralDoctor replaces recruiter->deptHead. It:
      - Uses TeachCOT to extract symptoms (no filtering).
      - Retrieves CaseRAG top-5 and UMLS_RAG top-2.
      - Instructs the LLM to prioritize CaseRAG while treating UMLS as auxiliary.
      - Emits DeptDiagnosisResult JSON with up to 5 diagnoses.

    Heavy resources (encoders, indices, agents) are shared via ResourcePool (initialized once).
    """

    def __init__(self):

        self.pool = ResourcePool.get()

        # System message: role, constraints, and output schema
        system_message = f"""
                            You are the General Doctor Agent in a multi-agent medical diagnostic system.
                            Your job is to produce a differential diagnosis list that maximizes recall while remaining clinically coherent.
                            
                            Global constraints:
                            1) You MUST output only valid JSON strictly matching the DeptDiagnosisResult schema shown below.
                            2) Produce up to FIVE candidate diagnoses (≤5). Each 'diagnose' must be a list with exactly one disease name and one corresponding department.
                            3) Provide a single confidence per candidate; ensure that confidences across all candidates sum to 1.0.
                            4) Provide numbered Chain-of-Thought style 'reasoning' steps referencing the specific evidence lines you were given.
                            5) Provide 'reference' lines prefixed with:
                               - "[History RAG]" for items derived from retrieved past cases (CaseRAG)
                               - "[BaseKnowledge RAG]" for items derived from UMLS knowledge
                            6) DO NOT ask for more information or suggest ordering tests; work strictly with the given evidence and inputs.
                            
                            DeptDiagnosisResult {{
                              "diagnoses": [
                                {{
                                  "department": [String],
                                  "diagnose": ["DiseaseName"],
                                  "confidence": [0.42],
                                  "reasoning": ["1. ...", "2. ..."],
                                  "reference": ["[History RAG] ...", "[BaseKnowledge RAG] ..."]
                                }},
                                ...
                              ]
                            }}
                            """

        model_client = OpenAIChatCompletionClient(
            model=llm_settings.openai_model,
            api_key=llm_settings.openai_api_key
        )

        self.memory = ListMemory(name="general_doctor_memory")

        super().__init__(
            name="GeneralDoctor",
            description="Generalist physician using CaseRAG (primary) + UMLS RAG (aux) and CoT.",
            model_client=model_client,
            system_message=system_message,
            memory=[self.memory],
            output_content_type=DeptDiagnosisResult
        )

    async def run(self, *, symptoms_text: str, patient_history: str = "") -> DeptDiagnosisResult:
        """
        Execute the general-doctor workflow:
          1) TeachCOT extraction (no filtering).
          2) CaseRAG top-5 + UMLS top-2 retrieval.
          3) LLM synthesis into DeptDiagnosisResult (<=5 diagnoses).
        """
        # 1) TeachCOT extraction (we do not filter or shrink; whatever we get, we pass along)
        feat = await self.pool.extract_symptoms(symptoms_text)
        q_pos = feat["q_pos"]  # positive symptoms as a list of strings

        # 2) Parallel retrieval: CaseRAG (top-5) and UMLS (top-2)
        case_task = asyncio.create_task(self.pool.case_candidates(symptoms_text, topk=5))
        umls_task = asyncio.create_task(asyncio.to_thread(self.pool.umls_candidates, q_pos, 2))
        case_resp, umls_resp = await asyncio.gather(case_task, umls_task)

        # Prepare compact evidence strings for the LLM (flattened, readable, and labeled)
        case_lines: List[str] = []
        for c in case_resp.get("candidates", []):
            dx = c.get("disease", "")
            sx = ", ".join(c.get("symptoms", [])[:6])
            cot = "; ".join(c.get("cot", [])[:])
            raw = c.get("raw_text", "")
            sc = c.get("score", 0.0)
            case_lines.append(f"[History RAG] disease={dx} | score={sc} | typical=[{sx}] | cot=[{cot}] | raw_snippet={raw[:220]}")

        umls_lines: List[str] = []
        for u in umls_resp.get("candidates", []):
            dx = u.get("name", "")
            m = ", ".join(u.get("matched_symptoms", [])[:6])
            typ = ", ".join(u.get("typical_symptoms", [])[:6])
            miss = ", ".join(u.get("missing_key_symptoms", [])[:4])
            defin = (u.get("definition", "") or "")[:220]
            sc = u.get("score", 0.0)
            umls_lines.append(f"[BaseKnowledge RAG] disease={dx} | score={sc} | matched=[{m}] | typical=[{typ}] | missing_keys=[{miss}] | def={defin}")

        # 3) Build the task for the LLM
        task_prompt = f"""
            --- General Doctor Task ---
            Primary objective: maximize candidate recall while staying clinically coherent.
            
            Patient symptoms (free text):
            {symptoms_text}
            
            Patient history (if any):
            {patient_history}
            
            Extracted positive symptoms (no filtering):
            {json.dumps(q_pos, ensure_ascii=False)}
            
            == Evidence to use ==
            IMPORTANT: Prioritize HISTORY CASE RAG. Treat UMLS knowledge as auxiliary context.
            
            [History Case RAG] (top-5):
            {json.dumps(case_lines, ensure_ascii=False, indent=2)}
            
            [UMLS / Base Knowledge RAG] (top-2):
            {json.dumps(umls_lines, ensure_ascii=False, indent=2)}
            
            Output requirements:
            - RAG information just for you reference, but you need to give most possible diagnostic candidates after deep thinking.
            - Output up to FIVE diagnoses (≤5). Each 'diagnose' contains exactly one disease name and its corresponding department.
            - Provide a single confidence per candidate; ensure confidences across all candidates sum to 1.0.
            - Provide concise numbered reasoning lines referencing explicit evidence above:
              e.g., "1) Typical triad ... [History RAG] disease=..., matched=[...]" or "[BaseKnowledge RAG] ..."
            - Provide 'reference' lines that are literal citations from the evidence (copy small snippets).
            - Return JSON ONLY; no extra commentary.
            """

        # 4) Call the LLM and parse the structured output
        result_msg = await super().run(task=task_prompt)
        structured: DeptDiagnosisResult = DeptDiagnosisResult.model_validate(result_msg.messages[-1].content)

        return structured

# if __name__ == "__main__":
#
#     ResourcePool.initialize(
#         case_out_dir="case_out/history_case_rag",
#         st_model_name="intfloat/e5-large-v2",
#         umls_sqlite_path="umls_out/mini_kit.sqlite"
#     )
#
#     general_doctor = GeneralDoctorAgent()
#
#     result = asyncio.run(general_doctor.run(
#         symptoms_text="I have burning chest pain worse when lying down. Chronic cough. Regurgitation at night.",
#         patient_history="No prior surgeries."
#     ))
#     print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

