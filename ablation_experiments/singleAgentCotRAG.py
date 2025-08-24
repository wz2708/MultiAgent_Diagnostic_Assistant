import os
import asyncio
import json
from pathlib import Path
from typing import List, Dict

import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_core.memory import ListMemory, MemoryContent
from config.settings import llm_settings
from agents.Decision_agent import DiseaseMatchAgent
from rag_utils.rag_utils import (
    load_faiss_index_and_mapping,
    retrieve_history_filtered,
    retrieve_medquad_filtered,
)
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

# ---------------- Schema ----------------
class SingleAgentRAGCoTItem(BaseModel):
    diagnose: List[str] = Field(..., description="A list of single diagnosis names, one for each disease")
    confidence: List[float] = Field(..., description="A single confidence list, summing to 1.0")
    reasoning: List[str] = Field(..., description="Chain-of-Thought Reasoning (Numbered Points)")
    reference: List[str] = Field(..., description="RAG & clinical knowledge citations")


class SingleAgentRAGCoTResult(BaseModel):
    diagnoses: List[SingleAgentRAGCoTItem]


# ---------------- Agent ----------------
class SingleAgentRAGCoT(AssistantAgent):
    def __init__(self, llm_model: str, history_index_path: str, history_map_path: str,
                 medquad_index_path: str, medquad_map_path: str,
                 symptoms: str, patient_history: str):
        self.symptoms = symptoms
        self.patient_history = patient_history

        # Load RAG indices
        self.load_index = load_faiss_index_and_mapping
        self.retrieve_history = retrieve_history_filtered
        self.retrieve_medquad = retrieve_medquad_filtered
        self.history_index, self.history_pairs = self.load_index(history_index_path, history_map_path)
        self.medquad_index, self.medquad_pairs = self.load_index(medquad_index_path, medquad_map_path)

        system_message = f"""
                            You are a generalist diagnostic physician. Given a patient's symptoms and history, use both retrieved past cases (History RAG) and medical knowledge QA (MedQuAD RAG) to inform your differential diagnosis. 
                            
                            Instructions:
                            1. Retrieve relevant RAG evidence internally from History and MedQuAD and incorporate it; do not ask for extra data.
                            2. Provide up to three candidate diagnoses in descending likelihood order.
                            3. For each candidate diagnosis, output:
                               - diagnose: single disease name in list
                               - confidence: one number; all confidences across candidates must sum to 1.0
                               - reasoning: a numbered chain-of-thought linking key symptoms to pathophysiology, typical presentation, and distinguishing features.
                               - reference: cite which parts came from RAG (prefix with [History RAG] or [MedQuAD RAG]) versus clinical intuition ([Clinical Knowledge]).
                            4. Emphasize the single most discriminative symptom driving the top diagnosis.
                            5. Output only valid JSON matching this schema:
                            SingleAgentRAGCoTResult {{
                              "diagnoses": [
                                {{
                                  "diagnose": ["Disease A"],
                                  "confidence": [0.6],
                                  "reasoning": ["1. ...", "2. ..."],
                                  "reference": ["[History RAG] ...", "[Clinical Knowledge] ..."]
                                }},
                                ...
                              ]
                            }}
                            No extra fields or commentary outside the JSON.
                            """
        model_client = OpenAIChatCompletionClient(model=llm_model, api_key=llm_settings.openai_api_key)
        self.memory = ListMemory(name="single_agent_rag_cot_memory")
        super().__init__(
            name="SingleAgentRAGCoT",
            description="Single-agent diagnostic with RAG + CoT",
            model_client=model_client,
            system_message=system_message,
            memory=[self.memory],
            output_content_type=SingleAgentRAGCoTResult,
        )
        # For retrieval encoding
        self._q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self._q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

    async def diagnose(self):
        # 1. Retrieve RAG candidates (use symptoms + history)
        query = self.symptoms
        hist_cands = self.retrieve_history(query, self.history_index, self.history_pairs,
                                           self._q_encoder, self._q_tokenizer, top_k=3)
        mq_cands = self.retrieve_medquad(query, self.medquad_index, self.medquad_pairs,
                                         self._q_encoder, self._q_tokenizer, top_k=3)

        # 2. Build task prompt supplement (embedding the retrieved items)
        task = f"""
                    History RAG candidates:
                    {json.dumps(hist_cands, ensure_ascii=False, indent=2)}
                    
                    MedQuAD RAG candidates:
                    {json.dumps(mq_cands, ensure_ascii=False, indent=2)}
                    
                    Patient Symptoms: {self.symptoms}
                    Patient History: {self.patient_history}
                    
                    Using the above, follow your system instructions to produce differential diagnoses with CoT reasoning, confidences, and references.
                    """
        result = await super().run(task=task)
        structured: SingleAgentRAGCoTResult = SingleAgentRAGCoTResult.model_validate(result.messages[-1].content)

        # 3. Normalize confidences to sum to 1.0
        total_conf = sum(item.confidence[0] for item in structured.diagnoses) if structured.diagnoses else 1.0
        if total_conf <= 0:
            n = len(structured.diagnoses)
            for item in structured.diagnoses:
                item.confidence = [1.0 / n]
        else:
            for item in structured.diagnoses:
                item.confidence = [item.confidence[0] / total_conf]

        # 4. Persist to memory
        await self.memory.add(MemoryContent(
            content=json.dumps(structured.model_dump(), ensure_ascii=False),
            mime_type="application/json"
        ))
        return structured


# ---------- Evaluation Utils ----------
async def evaluate_ranked(preds: List[str], truth: str, decision_agent):
    rank = None
    for i, p in enumerate(preds):
        res = await decision_agent.run(pred=p, truth=truth)
        is_same = getattr(res, "is_same", False)
        if is_same:
            rank = i + 1
            break
    top1_hit = rank == 1
    top2_hit = rank is not None and rank <= 2
    top3_hit = rank is not None and rank <= 3
    reciprocal_rank = 1.0 / rank if rank is not None else 0.0
    return top1_hit, top2_hit, top3_hit, rank, reciprocal_rank


def aggregate_from_records(jsonl_path: Path):
    total_cases = 0
    top1_hits = 0
    top2_hits = 0
    top3_hits = 0
    mrr_sum = 0.0

    with jsonl_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            metrics = record.get("metrics", {})
            top1_hits += int(bool(metrics.get("top1_hit", False)))
            top2_hits += int(bool(metrics.get("top2_hit", False)) and not bool(metrics.get("top1_hit", False)))
            top3_hits += int(bool(metrics.get("top3_hit", False))
                             and not bool(metrics.get("top1_hit", False))
                             and not bool(metrics.get("top2_hit", False)))
            mrr_sum += float(metrics.get("mrr", 0.0))
            total_cases += 1

    top1_acc = top1_hits / total_cases if total_cases else 0.0
    top2_acc = top2_hits / total_cases if total_cases else 0.0
    top3_acc = top3_hits / total_cases if total_cases else 0.0
    mean_mrr = mrr_sum / total_cases if total_cases else 0.0

    return {
        "num_cases": total_cases,
        "top1_accuracy": round(top1_acc, 4),
        "top2_accuracy": round(top2_acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "mean_mrr": round(mean_mrr, 4),
    }


# ---------- Main ----------
async def main():
    # Choose dataset: symptom2diagnosis or conversation2disease
    csv_path = "data/conversation2disease/test/test.csv" # "data/symptom2diagnosis/test/test.csv"
    df = pd.read_csv(csv_path)

    # RAG index paths (original traffic RAG pipeline)
    history_index_path = "data/symptom2diagnosis/history_dpr.index"
    history_map_path = "data/symptom2diagnosis/history_dpr.pkl"
    medquad_index_path = "data/conversation2disease/conversation2disease.index"
    medquad_map_path   = "data/conversation2disease/conversation2disease_pairs.pkl"

    output_dir = Path("output/singleAgent_rag_cot/conversation2disease/test_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "all_cases.jsonl"

    # Initialize agent
    decision = DiseaseMatchAgent()
    num = 0

    with out_file.open("w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            # if num < 142:
            #     num += 1
            #     continue
            if "input_text" in row and "output_text" in row:
                symptoms = row["input_text"]
                truth = row["output_text"]
            elif "symptom" in row and "disease" in row:
                symptoms = row["symptom"]
                truth = row["disease"]
            else:
                raise ValueError("Unexpected schema in test CSV")

            patient_history = ""  # modify if available

            # 1. Diagnose
            agent = SingleAgentRAGCoT(
                llm_model=llm_settings.openai_model,
                history_index_path=history_index_path,
                history_map_path=history_map_path,
                medquad_index_path=medquad_index_path,
                medquad_map_path=medquad_map_path,
                symptoms=symptoms,
                patient_history=patient_history
            )
            cot_result = await agent.diagnose()
            preds = [item.diagnose[0] for item in cot_result.diagnoses]

            # 2. Evaluate
            top1_hit, top2_hit, top3_hit, rank, mrr = await evaluate_ranked(preds, truth, decision)

            # 3. Save record
            case_record = {
                "symptom": symptoms,
                "ground_truth": truth,
                "cot_output": cot_result.model_dump(),
                "predictions": preds,
                "metrics": {
                    "top1_hit": top1_hit,
                    "top2_hit": top2_hit,
                    "top3_hit": top3_hit,
                    "rank": rank,
                    "mrr": mrr
                }
            }
            num += 1
            print(f"case {num}: truth={truth}, preds={preds}, top1={top1_hit}, top2={top2_hit}, top3={top3_hit}, rank={rank}")
            fout.write(json.dumps(case_record, ensure_ascii=False) + "\n")

    # Aggregate and save summary
    aggregated = aggregate_from_records(out_file)
    summary_path = output_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Aggregated metrics:", aggregated)


if __name__ == "__main__":
    asyncio.run(main())
