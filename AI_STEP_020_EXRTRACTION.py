import json
import math
from openai import OpenAI
import os
import sys
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
import httpx
from openai import OpenAI, LengthFinishReasonError
from tenacity import retry, stop_after_attempt, wait_exponential
from enum import Enum
client = OpenAI()

from pydantic import BaseModel, Field, field_validator, ValidationError
from pydantic_settings import BaseSettings 
from openai import (
    OpenAI, 
    LengthFinishReasonError, 
    APIConnectionError, 
    APITimeoutError, 
    RateLimitError, 
    InternalServerError
)
from tenacity import retry, stop_after_attempt, wait_exponential

class Settings(BaseSettings):
    openai_api_key: str
    model: str = "gpt-4o-2024-08-06"
    max_retries: int = 3
    timeout: float = 30.0
    max_input_chars: int = 15000
    max_output_tokens: int = 1000
    min_time_retry: int = 2
    max_time_retry: int = 10


settings = Settings()

# ---------------------------------------------------------
# 1. Schema Definitions (Optimized)
# ---------------------------------------------------------
class ActionCategory(str, Enum):
    MIGRATION = "migration"
    DEBUG = "debug"
    ENHANCE = "enhance"
    REFACTOR = "refactor"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    RESEARCH = "research"
    OTHER = "other"
    NOT_FOUND = "N" 

class Who(BaseModel):
    identified: bool = Field(description="True if a specific person, role, or team is mentioned.")
    evidence: str = Field(default="N", description="The exact phrase found. If no info, return 'N'.")

class What(BaseModel):
    identified: bool = Field(description="True if an action or intent is clearly defined.")
    category: ActionCategory = Field(default=ActionCategory.NOT_FOUND, description="Classify main intent of jira. If no info, return 'N'.")
    intent_evidence: str = Field(default="N", description="Specific action mentioned. If no info, return 'N'.")

class Why(BaseModel):
    identified: bool = Field(description="True if a business value/reason is provided.")
    value_evidence: str = Field(default="N", description="The reason. If no info, return 'N'.")

class CustomerImpact(BaseModel):
    identified: bool = Field(description="True if customer experience is mentioned")
    impact_evidence: str = Field(default="N", description="Specific impact. If no info, return 'N'.")

class Technologies(BaseModel):
    identified: bool = Field(description="True if tech tools/languages are mentioned.")
    tools: List[str] = Field(default_factory=list, description="List of technologies. Empty list if none.")

    @field_validator('tools', mode='before')
    @classmethod
    def normalize_tools(cls, v):
        if not v or not isinstance(v, list):
            return []
        return list(set(str(tool).strip() for tool in v))

class JiraAnalysis(BaseModel):
    reasoning: str = Field(description="Chain-of-thought analysis of the description.")
    who: Who
    what: What
    why: Why
    customer_impact: CustomerImpact
    technologies: Technologies

# ---------------------------------------------------------
# 2. Client Setup & Safety Checks
# ---------------------------------------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("CRITICAL ERROR: OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)

client = OpenAI(                   
    api_key=api_key,
    timeout=httpx.Timeout(30.0, connect=5.0),
    max_retries=0,  
)

# ---------------------------------------------------------
# 3. Extraction Logic
# ---------------------------------------------------------

@retry(stop=stop_after_attempt(settings.max_retries), wait=wait_exponential(multiplier=1, min=settings.min_time_retry, max=settings.max_time_retry))
def extract_jira_metadata(jira_description: str) -> Optional[JiraAnalysis]:

    if len(jira_description) > settings.max_input_chars:
            print(f"Error: Input too long ({len(jira_description)} chars)")
            return None

    system_prompt = (
        """You are a deterministic Jira metadata extractor. Extract structured metadata from ticket descriptions.\n"
        "STEPS:\n"
        "1. Analyze the text contextually in the 'reasoning' field.\n"
        "2. Extract Who, What, Why, Impact, and Tech.\n"
        "3. Be strict: If implicit, mark identified=False."
        Rules:
        - Only mark identified=True if explicitly stated in the text.
        - Do NOT infer unstated actors or intent.
        - If information is ambiguous, set identified=False.
        - Do NOT hallucinate technologies.
        - Keep text evidence EXACTLY as written in the input.
        - Do not paraphrase evidence.
        You must produce valid JSON matching the schema exactly.
""")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": jira_description}
    ]

    
        
    try:
      completion = client.beta.chat.completions.parse(
        model=settings.model,
        messages=messages,
        temperature=0.0,
        max_output_tokens = settings.max_output_tokens,
        response_format=JiraAnalysis,)
      
      if completion.choices[0].message.refusal:
          print("[MODEL_REFUSAL] Model refused to process the input.")
          print(f"Reason: {completion.choices[0].message.refusal}")
          return None

      return completion.choices[0].message.parsed

# --------------------------------------------------
# Structured Error Classification
# --------------------------------------------------

    except LengthFinishReasonError as e:
      print("[TOKEN_LIMIT_EXCEEDED]")
      print("Cause: Model output exceeded maximum token limit.")
      print(f"Details: {repr(e)}")
      return None

    except ValidationError as e:
      print("[SCHEMA_VALIDATION_ERROR]")
      print("Cause: Model response did not match JiraAnalysis schema.")
      print("Validation details:")
      print(e.json())
      return None

    except (APIConnectionError, APITimeoutError) as e:
      print("[NETWORK_ERROR]")
      print("Cause: Connection issue or timeout while calling OpenAI API.")
      print(f"Details: {repr(e)}")
      raise  # retry

    except RateLimitError as e:
      print("[RATE_LIMIT_ERROR]")
      print("Cause: API rate limit exceeded.")
      print(f"Details: {repr(e)}")
      raise  # retry

    except InternalServerError as e:
      print("[OPENAI_SERVER_ERROR]")
      print("Cause: OpenAI internal server error (5xx).")
      print(f"Details: {repr(e)}")
      raise  # retry

    except Exception as e:
      print("[UNEXPECTED_FATAL_ERROR]")
      print("Cause: Unhandled exception type.")
      print(f"Exception Type: {type(e).__name__}")
      print(f"Details: {repr(e)}")
      raise

# ---------------------------------------------------------
# Usage Example
# ---------------------------------------------------------
if __name__ == "__main__":
    description = "We need to move Redis to AWS ElastiCache..."

  
    result = extract_jira_metadata(description)
    print(result.model_dump_json(indent=2))
