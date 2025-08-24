import os
import asyncio
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
from pydantic import BaseModel, Field

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from config.settings import llm_settings
from agents.Decision_agent import DiseaseMatchAgent
from tqdm import tqdm

class SingleAgentResult(BaseModel):
    """Container for single-agent diagnostic output.

    Attributes:
        diagnoses: Disease names sorted by likelihood in descending order.
    """
    diagnoses: List[str] = Field(..., description="Disease names based on user input")

class SingleDoctorAgent(AssistantAgent):
    """
        A minimalist single-agent diagnostic LLM without CoT or RAG.

        This agent takes a free-form patient case description and returns up to
        three candidate diagnoses as a JSON list, ranked by likelihood.

        The class configures the LLM system prompt and exposes a convenience
        coroutine `diagnose` to run one-shot inference.

        Output schema is validated with `SingleAgentResult`.
    """

    def __init__(self):
        """Initialize the single diagnostic agent with system prompt and model."""

        system_message = """
        You are a generalist diagnostic physician. Given a patient's symptoms or conversation, propose up to three most likely disease diagnoses.
        Output a JSON object: { "diagnoses": ["disease1", "disease2", "disease3"] } in descending likelihood order. Do not include any extral information.
        """
        model_client = OpenAIChatCompletionClient(model=llm_settings.openai_model,
                                                  api_key=llm_settings.openai_api_key)
        super().__init__(
            name="SingleDoctor",
            description="Single LLM diagnostic agent without CoT or RAG",
            model_client=model_client,
            system_message=system_message,
            output_content_type=SingleAgentResult,
        )

    async def diagnose(self, patient_input: str) -> List[str]:
        """
            Run one-shot diagnosis on raw patient text.

            The agent is instructed to return a JSON with a `diagnoses` list of
            up to 3 items, sorted by likelihood in descending order.

            Args:
                patient_input: Raw patient description (symptoms / conversation).

            Returns:
                SingleAgentResult: Validated object with `diagnoses: List[str]`.

            Example:
                >>> res = await doctor.diagnose("Fever, cough, chest pain.")
                >>> res.diagnoses
                ['Pneumonia', 'Bronchitis', 'Common Cold']
        """
        task = f"Patient case: {patient_input}\nProvide top 3 candidate diagnoses as JSON."
        result = await super().run(task=task)

        return SingleAgentResult.model_validate(result.messages[-1].content)

# ---------- Evaluation Utils ----------
async def evaluate_ranked(preds: List[str], truth: str, decision_agent):
    """Evaluate a ranked list of predictions against ground truth.

    Uses a decision agent to determine semantic equivalence between each
    predicted disease name and the ground truth. The first match determines
    the rank.

    Args:
        preds: Ranked predictions from the agent (best → worst).
        truth: Ground-truth disease name string.
        decision_agent: A disease matching agent that exposes
            `await decision_agent.run(pred=..., truth=...)` and returns an
            object with boolean attribute `is_same`.

    Returns:
        A 5-tuple:
            - top1_hit (bool): True if truth == preds[0]
            - top2_hit (bool): True if truth in top 2 (and not top1)
            - top3_hit (bool): True if truth in top 3 (and not top1/top2)
            - rank (Optional[int]): 1-based rank of the first match, or None
            - reciprocal_rank (float): 1/rank if matched else 0.0

    Notes:
        - Top2/Top3 here are *exclusive* increments for reporting a histogram
          later; top1, top2, top3 can then be summed to cumulative accuracy.
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

def aggregate_from_records(jsonl_path: Path):
    """
        Aggregate per-case JSONL metrics into dataset-level figures.

       Reads line-delimited JSON (`all_cases*.jsonl`) where each record contains
       a `metrics` dict with keys: `top1_hit`, `top2_hit`, `top3_hit`, `mrr`.

       Args:
           jsonl_path: Path to the JSONL file with per-case metrics.

       Returns:
           dict: Aggregated metrics including:
               - num_cases
               - top1_accuracy
               - top2_accuracy
               - top3_accuracy
               - mean_mrr
    """
    total_cases = 0
    top1_hits = 0
    top2_hits = 0
    top3_hits = 0
    mrr_sum   = 0.0

    with jsonl_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            metrics = record.get("metrics", {})
            # Count once: if top1_hit true, top2/top3 should not be double counted
            top1_hits += int(bool(metrics.get("top1_hit", False)))
            top2_hits += int(bool(metrics.get("top2_hit", False)) and not bool(metrics.get("top1_hit", False)))
            top3_hits += int(bool(metrics.get("top3_hit", False)) and not bool(metrics.get("top2_hit", False)) and not bool(metrics.get("top1_hit", False)))
            mrr_sum   += float(metrics.get("mrr", 0.0))
            total_cases += 1

    top1_acc = top1_hits / total_cases if total_cases else 0.0
    top2_acc = top2_hits / total_cases if total_cases else 0.0  # cumulative
    top3_acc = top3_hits / total_cases if total_cases else 0.0
    mean_mrr = mrr_sum   / total_cases if total_cases else 0.0

    return {
        "num_cases":      total_cases,
        "top1_accuracy":  round(top1_acc, 4),
        "top2_accuracy":  round(top2_acc, 4),
        "top3_accuracy":  round(top3_acc, 4),
        "mean_mrr":       round(mean_mrr,   4),
    }


# ---------- Main Experiment ----------
async def main():
    """
        Run the SingleAgent ablation experiment end-to-end.

        Pipeline:
            1) Read test CSV (symptom, disease).
            2) For each case, run SingleDoctorAgent to get predictions.
            3) Compare with ground truth using DiseaseMatchAgent.
            4) Write per-case JSONL and aggregate metrics to JSON.

        Side-effects:
            Creates output directories and writes:
              - `output/singleAgent/symptom2diagnosis/test_results/all_cases1.jsonl`
              - `output/singleAgent/symptom2diagnosis/test_results/summary_metrics.json`
    """
    df = pd.read_csv("data/symptom2diagnosis/test.csv")

    output_path = Path("output/singleAgent/conversation2disease/test_results")
    os.makedirs("output/singleAgent/symptom2diagnosis/test_results", exist_ok=True)
    out_file = Path("output/singleAgent/conversation2disease/test_results/all_cases1.jsonl")
    fout = open("output/singleAgent/symptom2diagnosis/test_results/all_cases1.jsonl", "w", encoding="utf8")

    # Initialize agents
    doctor = SingleDoctorAgent()
    decision = DiseaseMatchAgent()

    num = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        symptom = row["symptom"]
        truth   = row["disease"]

        # 1. Single-agent diagnosis
        result = await doctor.diagnose(symptom)  # list of up to 3 disease strings
        preds = result.diagnoses
        
        # 2. Evaluate with decision agent
        top1_hit, top2_hit, top3_hit, rank, mrr = await evaluate_ranked(preds, truth, decision)

        case_record = {
            "symptom": symptom,
            "ground_truth": truth,
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

    # Aggregate
    aggregated = aggregate_from_records(out_file)
    summary_path = output_path / "summary_metrics.json"
    summary_path.write_text(json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Aggregated metrics:", aggregated)


if __name__ == "__main__":
    asyncio.run(main())
