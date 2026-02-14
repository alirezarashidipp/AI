from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from extraction import extract_jira_metadata, JiraAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Jira Metadata Extraction API")


# -------------------------
# Request Model
# -------------------------
class JiraRequest(BaseModel):
    description: str


# -------------------------
# Endpoint
# -------------------------
@app.post("/analyze", response_model=JiraAnalysis)
def analyze_jira(request: JiraRequest):

    try:
        result = extract_jira_metadata(request.description)

        if result is None:
            raise HTTPException(
                status_code=422,
                detail="Extraction failed (model refusal, schema mismatch, or token limit)."
            )

        return result

    except Exception as e:
        logger.exception("Fatal extraction error")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {type(e).__name__}"
        )
