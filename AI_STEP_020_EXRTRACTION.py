import os
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------------------------------------------------
# 1. Schema Definitions (تعریف ساختار داده‌ها)
# ---------------------------------------------------------

class Who(BaseModel):
    identified: bool = Field(description="True if a specific person, role, or team is mentioned.")
    evidence: Optional[str] = Field(default=None, description="The exact phrase or role name found (e.g., 'Python Developer', 'Support Team').")


class What(BaseModel):
    identified: bool = Field(description="True if an action or intent is clearly defined.")
    intent_evidence: Optional[str] = Field(default=None, description="The specific action mentioned (e.g., 'migrate database', 'fix login bug').")

class Why(BaseModel):
    identified: bool = Field(description="True if a business value or reason is provided.")
    value_evidence: Optional[str] = Field(default=None, description="The reason or benefit mentioned (e.g., 'improve performance').")

class CustomerImpact(BaseModel):
    identified: bool = Field(description="True if there is a direct or indirect mention of customer experience.")
    impact_evidence: Optional[str] = Field(default=None, description="The specific impact on the customer.")

class Technologies(BaseModel):
    identified: bool = Field(description="True if any technical tools, languages, or frameworks are mentioned.")
    tools: List[str] = Field(default_factory=list, description="List of extracted technologies (e.g. ['Python', 'AWS']).")


class JiraAnalysis(BaseModel):
    who: Who
    what: What
    why: Why
    customer_impact: CustomerImpact
    technologies: Technologies 

# ---------------------------------------------------------
# 2. Client Setup
# ---------------------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------
# 3. Extraction Logic
# ---------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def extract_jira_metadata(jira_description: str) -> JiraAnalysis:
    """
    Extracts Who, What, Why, CustomerImpact, AND Technologies 
    from a Jira description using strict JSON Structured Outputs.
    """
    
    # 4. Prompt Engineering (System Instruction + Few-Shot)
    system_prompt = (
        "You are a strict Jira ticket parser. Analyze the description to extract structured data.\n"
        "RULES:\n"
        "1. Identify Who, What, Why, CustomerImpact, and Technologies used.\n"
        "2. If a field is not present, set identified=False and value=null (or empty list).\n"
        "3. Output strictly valid JSON."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        
        # Few-Shot Example 1: With Technologies
        {
            "role": "user", 
            "content": "As a python developer, I need to query BigQuery using the new API to generate reports."
        },
        {
            "role": "assistant", 
            "content": (
                '{"who": {"identified": true, "evidence": "python developer"}, '
                '"what": {"identified": true, "intent_evidence": "query BigQuery using the new API"}, '
                '"why": {"identified": true, "value_evidence": "to generate reports"}, '
                '"customer_impact": {"identified": false, "impact_evidence": null}, '
                '"technologies": {"identified": true, "tools": ["Python", "BigQuery", "API"]}}'
            )
        },

        # Actual Input
        {"role": "user", "content": jira_description}
    ]

    # 5. API Call
    completion = client.beta.chat.completions.parse(
        model="gpt-4o", 
        messages=messages,
        temperature=0.0,
        response_format=JiraAnalysis, 
    )

    return completion.choices[0].message.parsed

# ---------------------------------------------------------
# Usage Example
# ---------------------------------------------------------
if __name__ == "__main__":
    description = "We need to migrate the Redis cache to AWS ElastiCache to improve Python API performance."
    
    try:
        result = extract_jira_metadata(description)
        # نمایش خروجی به صورت JSON تمیز
        print(result.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error: {e}")
