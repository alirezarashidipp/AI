import os
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------------------------------------------------
# 1. Schema Definitions (تعریف ساختار داده‌ها)
# ---------------------------------------------------------

class Who(BaseModel):
    identified: bool
    evidence: str | None = Field(description="The specific actor or role mentioned")

class What(BaseModel):
    identified: bool
    intent_evidence: str | None = Field(description="The specific action or intent")

class Why(BaseModel):
    identified: bool
    value_evidence: str | None = Field(description="The business value or reason")

class CustomerImpact(BaseModel):
    identified: bool
    impact_evidence: str | None = Field(description="Explicit mention of impact on the customer")

# --- کلاس جدید اضافه شده ---
class Technologies(BaseModel):
    identified: bool
    tools: List[str] = Field(description="List of specific technologies, languages, databases, or APIs mentioned (e.g. ['Python', 'BigQuery', 'Redis'])")

# --- کلاس مادر به‌روزرسانی شده ---
class JiraAnalysis(BaseModel):
    who: Who
    what: What
    why: Why
    customer_impact: CustomerImpact
    technologies: Technologies  # <--- اضافه شد

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
        "2. For 'Technologies', list distinct tools, languages, or frameworks (e.g. Python, AWS, API).\n"
        "3. If a field is not present, set identified=False and value=null (or empty list).\n"
        "4. Output strictly valid JSON."
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
        
        # Few-Shot Example 2: No Technologies
        {
            "role": "user", 
            "content": "The login page is slow. Customers are complaining."
        },
        {
            "role": "assistant", 
            "content": (
                '{"who": {"identified": false, "evidence": null}, '
                '"what": {"identified": true, "intent_evidence": "login page is slow"}, '
                '"why": {"identified": false, "value_evidence": null}, '
                '"customer_impact": {"identified": true, "impact_evidence": "Customers are complaining"}, '
                '"technologies": {"identified": false, "tools": []}}'
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
