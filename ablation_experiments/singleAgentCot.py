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
from config.settings import llm_settings
from agents.Decision_agent import DiseaseMatchAgent  


# ---------------- Schema ----------------
class DeptDiagnosisItem(BaseModel):
    diagnose: List[str] = Field(..., description="Single diagnosis name, each item has only one disease")
    confidence: List[float] = Field(..., description="A single-element list of the confidence level for this diagnosis")
    reasoning: List[str] = Field(..., description="Chain-of-Thought Reasoning (Numbered Points)")


class SingleAgentCoTResult(BaseModel):
    diagnoses: List[DeptDiagnosisItem]


# ---------------- Agent ----------------
class SingleDoctorCoTAgent(AssistantAgent):
    def __init__(self):
        system_message = """
                            You are a generalist diagnostic physician. Given a patient's symptoms or conversation, you will:
                            1. Think step by step (Chain-of-Thought) about the differential diagnoses.
                            2. Propose up to three candidate diseases in descending likelihood.
                            3. For each candidate, provide:
                               - diagnose: single disease name
                               - confidence: one number (the three confidences must sum to 1.0)
                               - reasoning: a numbered CoT list explaining why that diagnosis fits, including key symptoms mapping to pathophysiology and discriminative features.
                            Output only valid JSON matching this schema:
                            
                            {
                              "diagnoses": [
                                {
                                  "diagnose": ["Disease A"],
                                  "confidence": [0.6],
                                  "reasoning": [
                                     "1. Key symptom X suggests underlying mechanism Y.",
                                     "2. Epidemiology / typical presentation matches patient.",
                                     "3. Most discriminative feature is Z."
                                  ]
                                }
                              ]
                            }
                            
                            Do NOT output any extra fields or commentary outside the JSON.
                            """
        model_client = OpenAIChatCompletionClient(
            model=llm_settings.openai_model,
            api_key=llm_settings.openai_api_key
        )
        super().__init__(
            name="SingleDoctorCoT",
            description="Single-agent with chain-of-thought diagnostic reasoning",
            model_client=model_client,
            system_message=system_message,
            output_content_type=SingleAgentCoTResult,
        )

    async def diagnose(self, patient_input: str) -> SingleAgentCoTResult:
        task = f"Patient case: {patient_input}\nProvide your reasoning and top 3 candidate diagnoses as described."
        result = await super().run(task=task)
        structured = SingleAgentCoTResult.model_validate(result.messages[-1].content)
        # Normalize confidences to sum to 1.0 if needed
        # Extract raw and adjust
        total = sum(item.confidence[0] for item in structured.diagnoses) if structured.diagnoses else 1.0
        if total <= 0:
            # fallback uniform
            n = len(structured.diagnoses)
            for item in structured.diagnoses:
                item.confidence = [1.0 / n]
        else:
            for item in structured.diagnoses:
                item.confidence = [item.confidence[0] / total]
        return structured


# ---------- Evaluation Utils ----------
async def evaluate_ranked(preds: List[str], truth: str, decision_agent):
    """
    Return: top1_hit (bool), top2_hit (bool), top3_hit (bool), rank (1-based or None), reciprocal_rank (float)
    Uses decision_agent.run(pred=..., truth=...) which returns an object with attribute `is_same`.
    """
    rank = None
    for i, p in enumerate(preds):
        result = await decision_agent.run(pred=p, truth=truth)
        is_same = getattr(result, "is_same", False)
        if is_same:
            rank = i + 1  # 1-based
            break

    top1_hit = rank == 1
    top2_hit = rank is not None and rank <= 2
    top3_hit = rank is not None and rank <= 3
    reciprocal_rank = 1.0 / rank if rank is not None else 0.0

    return top1_hit, top2_hit, top3_hit, rank, reciprocal_rank


def extract_candidate_list(cot_result: SingleAgentCoTResult) -> List[str]:
    return [item.diagnose[0] for item in cot_result.diagnoses]


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
            # only count each case once: hierarchy top1 > top2 > top3
            top1_hits += int(bool(metrics.get("top1_hit", False)))
            top2_hits += int(bool(metrics.get("top2_hit", False)) and not bool(metrics.get("top1_hit", False)))
            top3_hits += int(bool(metrics.get("top3_hit", False)) and not bool(metrics.get("top1_hit", False)) and not bool(metrics.get("top2_hit", False)))
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


# ---------- Main Experiment ----------
async def main():
    # output locations
    output_dir = Path("output/singleAgent_cot/conversation2disease/test_results")
    # output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "all_cases1.jsonl"

    # initialize agents
    doctor = SingleDoctorCoTAgent()
    decision = DiseaseMatchAgent()

    num = 0
    with out_file.open("w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            if num < 172:
                num += 1
                continue
            # depending on dataset columns
            if "input_text" in row and "output_text" in row:
                symptom = row["input_text"]
                truth = row["output_text"]
            elif "symptom" in row and "disease" in row:
                symptom = row["symptom"]
                truth = row["disease"]
            else:
                raise ValueError("Unexpected dataframe schema")

            # 1. Single-agent CoT diagnosis
            cot_result = await doctor.diagnose(symptom)  # SingleAgentCoTResult
            preds = extract_candidate_list(cot_result)

            # 2. Evaluate
            top1_hit, top2_hit, top3_hit, rank, mrr = await evaluate_ranked(preds, truth, decision)

            # 3. Build record
            case_record = {
                "symptom": symptom,
                "ground_truth": truth,
                "cot_output": cot_result.model_dump(),  # full structured reasoning & confidences
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

    # 4. Aggregate metrics
    aggregated = aggregate_from_records(out_file)
    summary_path = output_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Aggregated metrics:", aggregated)


if __name__ == "__main__":
    asyncio.run(main())
