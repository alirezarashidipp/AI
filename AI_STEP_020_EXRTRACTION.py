import os
import sys
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI, LengthFinishReasonError
from tenacity import retry, stop_after_attempt, wait_exponential
from enum import Enum 

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
# ---------------------------------------------------------
# 1. Schema Definitions (Optimized)
# ---------------------------------------------------------

class ExtractionReasoning(BaseModel):
    analysis: str = Field(description="Briefly analyze the ticket to identify key components before extraction.")

class Who(BaseModel):
    identified: bool = Field(description="True if a specific person, role, or team is mentioned.")
    evidence: Optional[str] = Field(default=None, description="The exact phrase found (e.g., 'Python Developer').")

class What(BaseModel):
    identified: bool = Field(description="True if an action or intent is clearly defined.")
    category: Optional[ActionCategory] = Field(default=None, description="Classify the intent into the single most relevant category.")
    intent_evidence: Optional[str] = Field(default=None, description="The specific action mentioned (e.g., 'migrate database', 'fix login bug').")

class Why(BaseModel):
    identified: bool = Field(description="True if a business value/reason is provided.")
    value_evidence: Optional[str] = Field(default=None, description="The reason (e.g., 'improve performance').")

class CustomerImpact(BaseModel):
    identified: bool = Field(description="True if customer experience is mentioned.")
    impact_evidence: Optional[str] = Field(default=None, description="Specific impact on the customer.")

class Technologies(BaseModel):
    identified: bool = Field(description="True if tech tools/languages are mentioned.")
    tools: List[str] = Field(default_factory=list, description="List of technologies.")

    # Validator: Clean up technologies (lowercase & remove duplicates)
    @field_validator('tools', mode='before')
    @classmethod
    def normalize_tools(cls, v):
        if not v:
            return []
        # Remove duplicates and strip whitespace, keeping original case usually preferred 
        # but here we capitalize for consistency (optional)
        return list(set(tool.strip() for tool in v))

class JiraAnalysis(BaseModel):
    # 'reasoning' field forces the model to think first -> Better accuracy
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

client = OpenAI(api_key=api_key)

# ---------------------------------------------------------
# 3. Extraction Logic
# ---------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def extract_jira_metadata(jira_description: str) -> Optional[JiraAnalysis]:
    
    system_prompt = (
        "You are an expert Jira Analyst. Extract structured metadata from ticket descriptions.\n"
        "STEPS:\n"
        "1. Analyze the text contextually in the 'reasoning' field.\n"
        "2. Extract Who, What, Why, Impact, and Tech.\n"
        "3. Be strict: If implicit, mark identified=False."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": jira_description}
    ]

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06", # Use a valid, latest model supporting Structured Outputs
            messages=messages,
            temperature=0.0, # Deterministic
            response_format=JiraAnalysis, 
        )

        # Check for refusal (safety filtering)
        if completion.choices[0].message.refusal:
            print(f"Model refused to process: {completion.choices[0].message.refusal}")
            return None

        return completion.choices[0].message.parsed

    except LengthFinishReasonError:
        print("Error: Output token limit reached.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise # Let tenacity retry handle network blips, or crash if it's a logic error

# ---------------------------------------------------------
# Usage Example
# ---------------------------------------------------------
if __name__ == "__main__":
    description = "We need to migrate the Redis cache to AWS ElastiCache to improve Python API performance for our enterprise users."
    
    try:
        result = extract_jira_metadata(description)
        if result:
            # Print reasoning to see how the model thought
            print(f"--- Analysis Logic ---\n{result.reasoning}\n") 
            print("--- Structured Data ---")
            print(result.model_dump_json(indent=2))
    except Exception as e:
        print(f"Extraction failed after retries: {e}")
