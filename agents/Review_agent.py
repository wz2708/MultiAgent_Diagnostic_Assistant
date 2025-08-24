import asyncio
import sys
import json
from typing import List, Dict
from pydantic import BaseModel, Field
from autogen_core.memory import ListMemory, MemoryContent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from config.settings import llm_settings

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Reviewer Schema
class ReviewResult(BaseModel):
    diagnosis: List[str]
    scores: List[float]
    references: List[str]

# Reviewer Doctor Agent
class ReviewAgent(AssistantAgent):
    def __init__(self, perspective: str, symptoms: str, candidates: list, reports: list):
        self.perspective = perspective
        self.symptoms = symptoms
        self.candiates = candidates
        self.reports = reports

        system_message = f"""
            You are ReviewAgent specializing in **{self.perspective}** evaluation in a multi-agent medical diagnostic system.
            Your sole responsibility is to assess the quality of diagnoses from all departments based on Patient Symptoms and Department Discussion Summary.
            Focus exclusively on the **{self.perspective}** criteria provided.
            
            You are given:
              • Patient Symptoms: {self.symptoms},
              • Candidate Diagnoses: {self.candiates},
              • Candidate Diagnoses Reports: {self.reports}
            
            Rules:
            1. Do NOT propose new diagnoses.  
            2. Score given candidates mainly reference the Patient Symptoms and Candidate Diagnoses Reports.
            3. Output **only** valid JSON matching ReviewResult:
               {{
                 "diagnosis":    [str,…], the order should be strictly according to the Candidate Diagnoses: {self.candiates}
                 "scores":       [float,…], 
                 "references":   [str,…] numbered entries, each number according to each diagnosis，make sure the number of reference must be equal to the number of diagnosis or Candidate Diagnoses.
               }}
            4. Do NOT output any additional fields or free‑text commentary.
            """
        model_client = OpenAIChatCompletionClient(
            model=llm_settings.openai_model,
            api_key=llm_settings.openai_api_key
        )

        self.memory = ListMemory(name=f"review_memory_{self.perspective.replace(' ', '_')}")
        super().__init__(
            name = f"ReviewAgent_{self.perspective.replace(' ', '_')}",
            description="Review agent assess diagnosis quality based on user's prompt and candidates' reports",
            model_client=model_client,
            system_message=system_message,
            memory=[self.memory],
            output_content_type=ReviewResult
        )

    async def run(self) -> ReviewResult:
        snippet_map = {
            "Accuracy": """
                As the Accuracy reviewer, your mission is to confirm that each candidate diagnosis fully addresses 
                the critic’s unresolved questions.  
                • For each candidate, list which critic questions remain unanswered.  
                • Score = 1 – (#unanswered_questions / total_questions).  
                • In your reference note, explicitly call out which questions were resolved (“fixed”) and which remain open.
            """,

            "Coverage": """
                As the Coverage reviewer, your mission is to measure how completely each candidate’s evidence set 
                covers the core lever symptoms.  
                • For each candidate, enumerate which lever symptoms are supported by its rationale_evidence.  
                • Score = (#covered_lever_symptoms / total_lever_symptoms).  
                • In your reference note, highlight any missing lever symptoms that weaken the candidate’s coverage.
            """,

            "Interpretability": """
                As the Interpretability reviewer, your mission is to assess how clear, coherent, and standalone each 
                candidate’s reasoning is.  
                • For each candidate, count the number of self‑contained reasoning statements that require no outside context.  
                • Score = (count_of_standalone_statements / max_standalone_statements_across_candidates).  
                • In your reference note, point to one exemplar statement that best illustrates clarity and one that is confusing.
            """,

            "Specificity": """
                As the Specificity reviewer, your mission is to judge how well each candidate diagnosis is uniquely 
                distinguished from the others by its evidence.  
                • For each candidate, identify which lever symptoms it supports that no other candidate does.  
                • Score = (#unique_lever_symptoms / total_lever_symptoms).  
                • In your reference note, call out those uniquely supported symptoms that set this candidate apart.
            """
        }

        prompt_snippet = snippet_map.get(self.perspective)

        task_prompt = f"""
            --- Review Task ({self.perspective}) ---
            {prompt_snippet}
            """
        review_result = await super().run(task=task_prompt)
        review_structured: ReviewResult = ReviewResult.model_validate(review_result.messages[-1].content)
        return review_structured
