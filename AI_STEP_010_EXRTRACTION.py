# ultra_min_jira_analyzer.py
from __future__ import annotations

import json, time, math
from typing import Any, Dict, Optional, Literal, List
from dataclasses import dataclass

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# ---- Types ----
IntentType = Literal["Create", "Modify", "Remove", "Migrate", "Integrate", "Investigate", "Enforce"]
ValueCategory = Literal["Customer", "Cost", "Risk", "Compliance", "Internal Efficiency"]
ImpactLevel = Literal["No", "Low", "Medium", "High"]

# ---- Schema (minimal) ----
class Who(BaseModel):
    identified: bool
    confidence: int = Field(ge=0, le=100)
    actor: Optional[str] = None
    evidence: Optional[str] = None

class What(BaseModel):
    identified: bool
    confidence: int = Field(ge=0, le=100)
    intent_type: Optional[IntentType] = None
    intent_evidence: Optional[str] = None

class Why(BaseModel):
    identified: bool
    confidence: int = Field(ge=0, le=100)
    value_category: Optional[ValueCategory] = None
    value_evidence: Optional[str] = None

class CustomerImpact(BaseModel):
    identified: bool
    confidence: int = Field(ge=0, le=100)
    impact_level: ImpactLevel = "No"
    impact_evidence: Optional[str] = None

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
    max_tokens: int = 650
    timeout: float = 30.0
    retries: int = 2
    backoff: float = 1.2
    lp_min: float = -5.0
    lp_max: float = 0.0

SYSTEM = (
    "You are a strict enterprise Jira Story Analyzer. "
    "Use ONLY provided text. No guessing. Return ONLY valid JSON."
)

USER = """Analyze ONLY the text between delimiters. If missing/unclear: null or [] and lower confidence.
Evidence must be an EXACT contiguous snippet copied from text. Confidence = integer 0..100.
impact_level must be "No"|"Low"|"Medium"|"High" (default "No" if not explicit).

Allowed Intent Types: Create, Modify, Remove, Migrate, Integrate, Investigate, Enforce
Allowed Value Categories: Customer, Cost, Risk, Compliance, Internal Efficiency

TEXT START <<__PAYLOAD_START__>>
{t}
<<__PAYLOAD_END__>> TEXT END

Return ONLY JSON with this schema:
{{
  "who": {{"identified": bool, "confidence": int, "actor": str|null, "evidence": str|null}},
  "what": {{"identified": bool, "confidence": int, "intent_type": str|null, "intent_evidence": str|null}},
  "why": {{"identified": bool, "confidence": int, "value_category": str|null, "value_evidence": str|null}},
  "customer_impact": {{"identified": bool, "confidence": int, "impact_level": str, "impact_evidence": str|null}}
}}
"""

def _mean_logprob(resp: Any) -> Optional[float]:
    try:
        lp = resp.choices[0].logprobs
        toks = getattr(lp, "content", None)
        if not toks: return None
        vals = [float(getattr(t, "logprob")) for t in toks if getattr(t, "logprob", None) is not None]
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
    prompt = USER.format(t=text.strip()[:8000])

    last_err: Optional[Exception] = None
    for i in range(cfg.retries + 1):
        try:
            # 1) try with logprobs
            resp = client.chat.completions.create(
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                timeout=cfg.timeout,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                logprobs=True,
                top_logprobs=1,
            )
            raw = resp.choices[0].message.content
            data = json.loads(raw)
            out = JiraAnalysis.model_validate(data).model_dump()
            mlp = _mean_logprob(resp)
            out["generation_confidence"] = None if mlp is None else _lp_score(mlp, cfg.lp_min, cfg.lp_max)
            return out

        except (ValidationError, json.JSONDecodeError) as e:
            last_err = e
            if i >= cfg.retries: break
            time.sleep(cfg.backoff * (i + 1))
            # repair attempt (no extra complexity)
            prompt = prompt + "\n\nFix your previous output to match schema. Return ONLY JSON."
            continue

        except Exception as e:
            # 2) fallback without logprobs (model may not support logprobs+json_object)
            last_err = e
            if i >= cfg.retries: break
            time.sleep(cfg.backoff * (i + 1))
            try:
                resp = client.chat.completions.create(
                    model=cfg.model,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    timeout=cfg.timeout,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                )
                raw = resp.choices[0].message.content
                data = json.loads(raw)
                out = JiraAnalysis.model_validate(data).model_dump()
                out["generation_confidence"] = None
                return out
            except Exception as inner:
                last_err = inner

    raise last_err or RuntimeError("analysis failed")

if __name__ == "__main__":
    sample = """
As a backend team, we will migrate user authentication to OAuth2 provider.
This may affect customer login flow but will maintain backward compatibility.
"""
    print(json.dumps(analyze(sample), indent=2, ensure_ascii=False))
