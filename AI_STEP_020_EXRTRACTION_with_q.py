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
    NOT_FOUND = "not_found" 

class Who(BaseModel):
    identified: bool = Field(description="True if a specific person, role, or team is mentioned.")
    evidence: str = Field(default="not_found", description="The exact phrase found. If no info, return 'N'.")

class What(BaseModel):
    identified: bool = Field(description="True if an action or intent is clearly defined.")
    category: ActionCategory = Field(default=ActionCategory.NOT_FOUND, description="Classify main intent of jira. If no info, return 'N'.")
    intent_evidence: str = Field(default="not_found", description="Specific action mentioned. If no info, return 'N'.")

class Why(BaseModel):
    identified: bool = Field(description="True if a business value/reason is provided.")
    value_evidence: str = Field(default="not_found", description="The reason. If no info, return 'N'.")

class CustomerImpact(BaseModel):
    identified: bool = Field(description="True if customer experience is mentioned")
    impact_evidence: str = Field(default="not_found", description="Specific impact. If no info, return 'N'.")

class Technologies(BaseModel):
    identified: bool = Field(description="True if tech tools/languages are mentioned.")
    tools: List[str] = Field(default_factory=list, description="List of technologies. Empty list if none.")

    @field_validator('tools', mode='before')
    @classmethod
    def normalize_tools(cls, v):
        if not v or not isinstance(v, list):
            return []
        return list(set(str(tool).strip() for tool in v))

class BacklogQuestion(BaseModel):
    question: str = Field(description="The direct question to ask the ticket reporter.")
    reasoning: str = Field(description="Why is this question important? (e.g. 'Missing data schemas', 'Ambiguous deadline').")


class JiraAnalysis(BaseModel):
    reasoning: str = Field(description="Chain-of-thought analysis of the description.")
    who: Who
    what: What
    why: Why
    customer_impact: CustomerImpact
    technologies: Technologies

    grooming_questions: List[BacklogQuestion] = Field(
    description="Exactly 2 most critical questions to clarify missing information, edge cases, or unstated data requirements.",
    max_length=2,
    min_length=2)
  
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
        """You are an expert Technical Product Manager and Jira Analyst.\n"
           Your Goal:\n
        1. Extract structured metadata (Who, What, Why, Impact, Tech) from the ticket description.\n
        2. CRITICAL: Act as a 'Backlog Groomer'. Identify gaps, ambiguities, or MISSING DATA in the requirements.\n


         Rules for Questions:\n
        - Generate exactly 2 clarifying questions.\n
        - Focus on what is NOT written but essential for development (e.g., missing error handling, undefined data schemas, unknown load requirements).\n
        - Be specific. Do not ask generic questions like "Is this correct?".\n
        
        Rules for Extraction:
        - Extract Who, What, Why, Impact, and Tech.\n
        - Only mark identified=True if explicitly stated.
        - Do NOT hallucinate technologies.
        - Keep text evidence EXACTLY as written.

        You must produce valid JSON matching the schema exactly""")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": jira_description}
    ]

    
        
    try:
      completion = client.beta.chat.completions.parse(
        model=settings.model,
        messages=messages,
        temperature=0.0,
        response_format=JiraAnalysis,)
      
      if completion.choices[0].message.refusal:
          print("[MODEL_REFUSAL] Model refused to process the input.")
          print(f"Reason: {completion.choices[0].message.refusal}")
          return None

      return completion.choices[0].message.parsed


  # --------------------------------------------------
  # Error Handling
  # --------------------------------------------------
  except LengthFinishReasonError as e:
      print("[TOKEN_LIMIT_EXCEEDED]")
      return None
  except ValidationError as e:
      print("[SCHEMA_VALIDATION_ERROR]")
      print(e.json())
      return None
  except (APIConnectionError, APITimeoutError) as e:
      print(f"[NETWORK_ERROR] {e}")
      raise 
  except RateLimitError as e:
      print(f"[RATE_LIMIT_ERROR] {e}")
      raise 
  except InternalServerError as e:
      print(f"[OPENAI_SERVER_ERROR] {e}")
      raise 
  except Exception as e:
      print(f"[UNEXPECTED_ERROR] {type(e).__name__}: {e}")
      raise

# ---------------------------------------------------------
# Usage Example
# ---------------------------------------------------------
if __name__ == "__main__":
    description = "We need to move Redis to AWS ElastiCache..."

  
    result = extract_jira_metadata(description)
    print(result.model_dump_json(indent=2))
