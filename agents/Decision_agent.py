import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from pydantic import BaseModel, Field

from config.settings import llm_settings

# 1. schema
class MatchResult(BaseModel):
    is_same: bool = Field(..., description="True if the two disease names refer to the same condition")

# 2. define Match Agent
class DiseaseMatchAgent(AssistantAgent):
    def __init__(self):
        system_message = """
            You are a medical equivalence adjudicator.
            • Task: Given two disease names (predicted vs. ground truth), decide whether they refer to the same underlying medical condition for diagnostic evaluation.
            • Instructions:
            Normalize names internally before comparing (case-insensitive, remove punctuation, expand common abbreviations, handle minor spelling variations).
            Match criteria (return is_same: true) if:
            They are exact synonyms (e.g., “GERD” vs. “Gastroesophageal reflux disease”).
            One is a common alias of the other (“Peptic ulcer disease” vs. “PUD”).
            They denote the same disease even if expressed at slightly different granularity, provided the difference does not change the core diagnosis (e.g., “Type 2 diabetes mellitus” vs. “diabetes mellitus” — accept unless explicit subtype mismatch matters in context).
            
            • Do NOT match if:
            They are distinct diseases with overlapping symptoms (e.g., “Gastritis” vs. “Peptic ulcer disease”).
            One is a broader category and the other a specific different entity where precision matters (e.g., “Infectious gastroenteritis” vs. “Typhoid fever” — do not match unless contextual evidence says they treated them as interchangeable).
            They represent complications, causes, or related but not equivalent conditions (e.g., “GERD” vs. “Barrett’s esophagus”).
            
            • Output only valid JSON matching the schema:
              {
                "is_same": true|false
              }
            • Do NOT output redundant fields or explanations.
            """
        client = OpenAIChatCompletionClient(
            model=llm_settings.openai_model,
            api_key=llm_settings.openai_api_key
        )
        super().__init__(
            name="DiseaseMatchAgent",
            description="A lightweight agent to determine whether two disease names are the same",
            model_client=client,
            system_message=system_message,
            memory=[],
            output_content_type=MatchResult
        )

    async def run(self, *, pred: str, truth: str) -> MatchResult:
        task = f'''
            Compare these two disease names:
            1) "{pred}"
            2) "{truth}"
            
            Answer in JSON as per the schema.
            '''
        result = await super().run(task=task)
        return MatchResult.model_validate(result.messages[-1].content)

# # 3. test case
# async def test_match():
#     agent = DiseaseMatchAgent()
#     cases = [
#         ("Gastroesophageal reflux disease", "GERD"),
#         ("Peptic ulcer disease", "Gastric ulcer"),
#         ("Functional dyspepsia", "Gastroparesis")
#     ]
#     for pred, truth in cases:
#         match = await agent.run(pred=pred, truth=truth)
#         print(f"{pred!r} ≅ {truth!r}? → {match.is_same}")
#
# if __name__ == "__main__":
#     asyncio.run(test_match())
