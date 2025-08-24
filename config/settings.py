from pydantic_settings import BaseSettings
from pydantic import Field

class LLMSettings(BaseSettings):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_api_base: str = Field(default="https://api.openai.com/v1", env="OPENAI_API_BASE")
    openai_model: str = "gpt-4.1-mini"

    class Config:
        env_file = ".env4"
        env_file_encoding = "utf-8"

llm_settings = LLMSettings()
