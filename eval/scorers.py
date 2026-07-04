from __future__ import annotations
 
import json
import random
from dataclasses import dataclass
from typing import Any, Literal
 
import httpx
from jsonschema import Draft202012Validator
 
 
def normalise_text(x: str) -> str:
    return " ".join(x.lower().strip().split())
 
 
def exact_match(prediction: str, reference: str) -> float:
    return float(normalise_text(prediction) == normalise_text(reference))
 
 
def token_f1(prediction: str, reference: str) -> float:
    pred = normalise_text(prediction).split()
    ref = normalise_text(reference).split()
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    common = set(pred) & set(ref)
    overlap = sum(min(pred.count(t), ref.count(t)) for t in common)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)
 
 
def schema_validity(prediction: str, schema: dict[str, Any] | None) -> float:
    if schema is None:
        return 1.0
    try:
        obj = json.loads(prediction)
        Draft202012Validator(schema).validate(obj)
        return 1.0
    except Exception:
        return 0.0
 
 
@dataclass(slots=True)
class JudgeConfig:
    endpoint: str
    api_key: str | None = None
    model: str = "gpt-4o-mini"
    timeout: float = 30.0
 
 
JUDGE_PROMPT = """You are evaluating an LLM answer.
 
Question:
{question}
 
Reference/context:
{reference}
 
Answer A:
{answer_a}
 
Answer B:
{answer_b}
 
Rubric:
- Correctness is most important.
- Penalise unsupported claims.
- Penalise unnecessary verbosity.
- Ignore minor formatting differences.
 
Return JSON only:
{{"winner":"A"|"B"|"tie", "reason":"short reason"}}
"""
 
 
async def _judge_pair(config: JudgeConfig, question: str, reference: str, answer_a: str, answer_b: str) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, answer_a=answer_a, answer_b=answer_b)
    payload = {"model": config.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
    async with httpx.AsyncClient(timeout=config.timeout) as client:
        r = await client.post(config.endpoint, headers=headers, json=payload)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
    return json.loads(content)
 
 
async def debiased_win_rate(
    config: JudgeConfig,
    *,
    question: str,
    reference: str,
    candidate: str,
    baseline: str,
) -> dict[str, Any]:
    """Run pairwise judging in both orders to reduce position bias."""
    first = await _judge_pair(config, question, reference, candidate, baseline)
    second = await _judge_pair(config, question, reference, baseline, candidate)
 
    candidate_votes = 0
    baseline_votes = 0
    ties = 0
 
    if first["winner"] == "A":
        candidate_votes += 1
    elif first["winner"] == "B":
        baseline_votes += 1
    else:
        ties += 1
 
    if second["winner"] == "B":
        candidate_votes += 1
    elif second["winner"] == "A":
        baseline_votes += 1
    else:
        ties += 1
 
    total = candidate_votes + baseline_votes + ties
    return {
        "candidate_votes": candidate_votes,
        "baseline_votes": baseline_votes,
        "ties": ties,
        "candidate_win_rate": candidate_votes / total,
        "raw": [first, second],
    }
