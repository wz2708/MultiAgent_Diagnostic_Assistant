import asyncio, sys, json
from typing import Literal, Optional
from pydantic import BaseModel, Field
from autogen_core.memory import ListMemory
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from config.settings import llm_settings

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Reception Doctoc Schema
class TeachCOTResult(BaseModel):
    positive_symptoms: list[str] = Field(default_factory=list)
    negative_symptoms: list[str] = Field(default_factory=list)
    cot_summary: list[str]

# Prompts for different mode
_TEACHER_SYS = """You are the Symptom TeachCOT (Teacher mode) for a diagnostic system.
                    
                    Goals:
                    1) Extract normalized symptoms for retrieval.
                    2) Write a short, grounded CoT that EXPLAINS why the GOLD diagnosis fits THIS patient's text, just like a real doctor's diagnosis.
                    
                    Hard constraints (apply to BOTH extraction and CoT):
                    - Use ONLY evidence found in the patient text. Do NOT invent labs, imaging, timing, treatments, or extra symptoms.
                    - Every symptom you keep must be supportable by words in the text (or a trivial variant like plural/singular).
                    - English, lowercase; body-region first, head-noun last (e.g., "neck pain"). Deduplicate. Generic terms must include location.
                    - If a candidate is not explicitly in the text, DROP it.
                    - Negative symptoms are allowed only if negated in the text (e.g., "no fever", "denies chest pain").
                    
                    CoT format (STRICT; 3–4 lines, 55–85 words total):
                    1) Key symptoms: <comma-separated normalized symptoms>  [evidence: "...", "..."]
                    2) Negatives: <comma-separated normalized negatives or "none stated">  [evidence: "...", "..."]
                    3) Why this fits <GOLD>: link EACH kept symptom to <GOLD> in one flowing sentence; avoid textbook lists; keep to evidence.
                    4) Pitfalls/differentials: name 1–2 close mimics and a 1-clause reason each; do NOT add new symptoms.
                    
                    Few-shot example (style):
                    
                    Patient text:
                    "I have burning pain in my upper abdomen that worsens after meals. I also get heartburn and nausea."
                    
                    GOLD diagnosis:
                    peptic ulcer disease
                    
                    Expected JSON:
                    {
                      "positive_symptoms": ["upper abdominal pain", "heartburn", "nausea"],
                      "negative_symptoms": [],
                      "cot_summary": [
                        "Key symptoms: upper abdominal pain, heartburn, nausea. 
                        "Negatives: none stated.",
                        "Why this fits peptic ulcer disease: the meal-worsened upper abdominal pain with concurrent heartburn and nausea matches peptic ulcer presentations grounded in mucosal injury; all elements appear in the patient's description.",
                        "Pitfalls/differentials: gerd (heartburn dominant), gastritis (diffuse discomfort without meal-linked pain)."
                      ]
                    }
                    
                    Output ONLY valid JSON for: TeachCOTResult { positive_symptoms, negative_symptoms, cot_summary }.
                    """

_STUDENT_SYS = """You are the Symptom TeachCOT (Student mode).
                    
                    Goals:
                    1) Extract normalized symptoms for retrieval.
                    2) Return an EMPTY cot_summary array ([]) to avoid leaking diagnoses.
                    
                    Hard constraints:
                    - Use ONLY evidence found in the patient text. Do NOT invent items.
                    - English, lowercase; body-region first, head-noun last (e.g., "neck pain"). Deduplicate. Generic terms must include location.
                    - If a candidate is not explicitly in the text, DROP it.
                    - Negative symptoms only if negated by the patient text.
                    
                    Output ONLY valid JSON for: TeachCOTResult { positive_symptoms, negative_symptoms, cot_summary }.
                    """

# Reception Doctor Agent (RAG-injected)
class TeachCOTAgent(AssistantAgent):
    def __init__(self, mode: Literal["teacher", "student"] = "teacher"):
        assert mode in ("teacher", "student")
        self.mode = mode
        system_message = _TEACHER_SYS if mode == "teacher" else _STUDENT_SYS

        model_client = OpenAIChatCompletionClient(
            model=llm_settings.openai_model,
            api_key=llm_settings.openai_api_key,
            temperature=0.0
        )
        self.memory = ListMemory(name=f"teachcot_{mode}_memory")

        super().__init__(
            name=f"TeachCOT_{mode}",
            description=f"Extract symptoms (+negatives) and {('gold-linked ' if mode == 'teacher' else 'neutral ')}CoT",
            model_client=model_client,
            system_message=system_message,
            memory=[self.memory],
            output_content_type=TeachCOTResult
        )

    async def run(self, *, text: str, gold: Optional[str] = None) -> TeachCOTResult:
        if self.mode == "teacher":
            if not gold:
                raise ValueError("Teacher mode requires gold diagnosis")
            task = f"""--- TeachCOT (Teacher) Task ---
                    Patient text:
                    {text}

                    Gold diagnosis:
                    {gold}

                    Return only valid JSON for TeachCOTResult.
                    """
        else:
            task = f"""--- TeachCOT (Student) Task ---
                    Patient text:
                    {text}

                    Return only valid JSON for TeachCOTResult.
                    """
        res = await super().run(task=task)
        return TeachCOTResult.model_validate(res.messages[-1].content)


# async def main():
#     cases = [
#         (
#             "I have a burning sensation in my stomach that comes and goes. It's worse when I eat and when I lie down. I also have heartburn and indigestion.",
#             "peptic ulcer disease"
#         ),
#         (
#             "I’ve been running a high fever for several days, around 102°F. I have stomach cramps and diarrhea. I recently visited rural areas in Africa.",
#             "typhoid"
#         )
#     ]
#
#     # Teacher mode with ground truth
#     teacher = TeachCOTAgent(mode="teacher")
#     for text, gold in cases:
#         out = await teacher.run(text=text, gold=gold)
#         print("\n[Teacher] =====")
#         print(json.dumps(out.model_dump(), indent=2))
#
#     # Student mode for inference
#     student = TeachCOTAgent(mode="student")
#     for text, _ in cases:
#         out = await student.run(text=text)
#         print("\n[Student] =====")
#         print(json.dumps(out.model_dump(), indent=2))
#
# if __name__ == "__main__":
#     asyncio.run(main())