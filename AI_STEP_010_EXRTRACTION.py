# ultra_min_jira_analyzer_v4_impact_fix.py
from __future__ import annotations

import json, time, math
from typing import Any, Dict, Optional, Literal
from dataclasses import dataclass

from openai import OpenAI
from pydantic import BaseModel, Field

# ---- Types ----
IntentType = Literal["Create", "Modify", "Remove", "Migrate", "Integrate", "Investigate", "Enforce"]
ValueCategory = Literal["Customer", "Cost", "Risk", "Compliance", "Internal Efficiency"]

# تغییر ۱: حذف "No" از اینجا. اگر تاثیری نباشد، identified باید false باشد.
# اگر تاثیر باشد اما شدت معلوم نباشد، مقدار null می‌شود.
ImpactLevel = Literal["Low", "Medium", "High"]

# ---- Schema (Strict Enterprise Mode) ----

class Who(BaseModel):
    identified: bool
    confidence: int = Field(ge=0, le=100)
    actor: str | None
    evidence: str | None

class What(BaseModel):
    identified: bool
    confidence: int = Field(ge=0, le=100)
    intent_type: IntentType | None
    intent_evidence: str | None

class Why(BaseModel):
    identified: bool
    confidence: int = Field(ge=0, le=100)
    value_category: ValueCategory | None
    value_evidence: str | None

class CustomerImpact(BaseModel):
    identified: bool
    confidence: int = Field(ge=0, le=100)
    
    # تغییر ۲: اضافه کردن | None
    # این یعنی: یا یکی از مقادیر Low/Medium/High است، یا اگر مدل نتوانست تشخیص دهد، null است.
    impact_level: ImpactLevel | None 
    
    impact_evidence: str | None

class JiraAnalysis(BaseModel):
    who: Who
    what: What
    why: Why
    customer_impact: CustomerImpact

# ---- Config ----
@dataclass
class Cfg:
    model: str = "gpt-4.1" 
    temperature: float = 0.0
    max_tokens: int = 1000
    timeout: float = 30.0
    retries: int = 2
    backoff: float = 1.2
    lp_min: float = -5.0
    lp_max: float = 0.0

SYSTEM = (
    "You are a strict enterprise Jira Story Analyzer. "
    "Extract structured data. If a field is not present in text, return null."
)

# تغییر ۳: آپدیت پرامپت برای توضیح منطق جدید
USER_TEMPLATE = """Analyze ONLY the text below.
Evidence must be an EXACT contiguous snippet. Confidence = integer 0..100.

Rules for 'customer_impact':
1. If no impact is mentioned: set identified=false, impact_level=null, impact_evidence=null.
2. If impact is mentioned BUT severity (Low/Medium/High) is NOT specified: set identified=true, impact_evidence="...", impact_level=null.
3. If impact AND severity are mentioned: set identified=true, impact_evidence="...", impact_level="Low"|"Medium"|"High".

TEXT START <<__PAYLOAD_START__>>
{t}
<<__PAYLOAD_END__>> TEXT END
"""

def _mean_logprob(resp: Any) -> Optional[float]:
    try:
        lp = resp.choices[0].logprobs
        toks = getattr(lp, "content", None)
        if not toks: return None
        vals = [float(t.logprob) for t in toks if t.logprob is not None]
        return (sum(vals) / len(vals)) if vals else None
    except Exception:
        return None

def _lp_score(mean_lp: float, lp_min: float, lp_max: float) -> int:
    if mean_lp is None or math.isnan(mean_lp): return 0
    lo, hi = (lp_min, lp_max) if lp_min < lp_max else (-5.0, 0.0)
    x = max(lo, min(hi, float(mean_lp)))
    return int(round((x - lo) / (hi - lo) * 100))

def analyze(text: str, client: Optional[OpenAI] = None, cfg: Cfg = Cfg()) -> Dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    
    client = client or OpenAI()
    prompt = USER_TEMPLATE.format(t=text.strip()[:8000])
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt}
    ]

    last_err: Optional[Exception] = None

    for i in range(cfg.retries + 1):
        try:
            # Attempt 1: Strict Structured Output WITH Logprobs
            resp = client.beta.chat.completions.parse(
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                timeout=cfg.timeout,
                messages=messages,
                response_format=JiraAnalysis,
                logprobs=True,
                top_logprobs=1,
            )
            
            out = resp.choices[0].message.parsed.model_dump()
            mlp = _mean_logprob(resp)
            out["generation_confidence"] = _lp_score(mlp, cfg.lp_min, cfg.lp_max) if mlp is not None else None
            return out

        except Exception as e:
            # Fallback: Retry WITHOUT Logprobs
            last_err = e
            try:
                resp = client.beta.chat.completions.parse(
                    model=cfg.model,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    timeout=cfg.timeout,
                    messages=messages,
                    response_format=JiraAnalysis,
                )
                out = resp.choices[0].message.parsed.model_dump()
                out["generation_confidence"] = None 
                return out
                
            except Exception as inner_e:
                last_err = inner_e
                if i < cfg.retries:
                    time.sleep(cfg.backoff * (i + 1))

    raise last_err or RuntimeError("Analysis failed after retries")

if __name__ == "__main__":
    # تست سناریوی "تاثیر هست ولی لول مشخص نیست"
    sample = """
    We need to refactor the payment gateway wrapper.
    NOTE: This change will definitely impact the checkout latency for end-users, so be careful.
    """
    try:
        result = analyze(sample)
        # انتظار داریم: identified=True, evidence="...", impact_level=None (چون نگفته Low/High)
        print(json.dumps(result["customer_impact"], indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Analysis Error: {e}")
