"""LLM-as-a-judge evaluators for CentaurDrug LangSmith datasets."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any, Mapping

JUDGE_PROMPT_VERSION = "chat-judge-v1"
JUDGE_MODEL_ENV = "LANGSMITH_JUDGE_MODEL"
DEFAULT_JUDGE_MODEL = "gemini-2.5-flash"

CHAT_JUDGE_CRITERIA = {
    "usefulness": "Directly answers the user's question and gives actionable next steps.",
    "scientific_caution": "Distinguishes computational estimates from established evidence and avoids unsupported safety claims.",
    "clarity": "Is concise, coherent, and understandable to a scientifically literate user.",
    "applicability_domain": "Correctly discusses applicability-domain risk when it is present in the supplied context.",
}


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return str(content)


def _json_text(text: str) -> str:
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    return fenced.group(1).strip() if fenced else cleaned


def parse_judge_response(text: str) -> dict[str, Any]:
    """Parse and validate the judge's JSON response."""

    try:
        payload = json.loads(_json_text(text))
    except json.JSONDecodeError as exc:
        raise ValueError("LLM judge returned invalid JSON.") from exc

    criteria = payload.get("criteria")
    if not isinstance(criteria, Mapping):
        raise ValueError("LLM judge response is missing a criteria object.")

    scores: dict[str, float] = {}
    for key in CHAT_JUDGE_CRITERIA:
        try:
            score = float(criteria[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"LLM judge response is missing score: {key}.") from exc
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"LLM judge score must be between 0 and 1: {key}.")
        scores[key] = score

    overall_score = payload.get("overall_score")
    if overall_score is None:
        overall_score = sum(scores.values()) / len(scores)
    try:
        overall_score = float(overall_score)
    except (TypeError, ValueError) as exc:
        raise ValueError("LLM judge overall_score must be numeric.") from exc
    if not 0.0 <= overall_score <= 1.0:
        raise ValueError("LLM judge overall_score must be between 0 and 1.")

    rationale = payload.get("rationale", "")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("LLM judge response is missing rationale.")

    return {
        "criteria": scores,
        "overall_score": overall_score,
        "rationale": rationale.strip(),
        "needs_human_review": bool(payload.get("needs_human_review", False)),
    }


def build_chat_judge_prompt(
    inputs: Mapping[str, Any],
    outputs: Any,
    reference_outputs: Any | None = None,
) -> str:
    reference = reference_outputs if reference_outputs else "No reference answer supplied."
    return (
        "You are a strict scientific reviewer evaluating a medicinal-chemistry "
        "copilot response. Judge only the supplied evidence; do not invent facts.\n\n"
        f"Evaluation criteria:\n{json.dumps(CHAT_JUDGE_CRITERIA, indent=2)}\n\n"
        f"User inputs:\n{json.dumps(dict(inputs), indent=2, default=str)}\n\n"
        f"Agent response:\n{json.dumps(outputs, indent=2, default=str)}\n\n"
        f"Reference output, if any:\n{json.dumps(reference, indent=2, default=str)}\n\n"
        "Return JSON only, with this exact shape:\n"
        '{"criteria": {"usefulness": 0.0, "scientific_caution": 0.0, '
        '"clarity": 0.0, "applicability_domain": 0.0}, '
        '"overall_score": 0.0, "rationale": "...", '
        '"needs_human_review": false}\n'
        "Use scores from 0.0 to 1.0. Set needs_human_review=true when the "
        "answer makes a consequential scientific or safety claim that requires "
        "qualified human review."
    )


@lru_cache(maxsize=1)
def get_judge_llm():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required to run the LLM judge.")

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=os.getenv(JUDGE_MODEL_ENV, DEFAULT_JUDGE_MODEL).strip()
        or DEFAULT_JUDGE_MODEL,
        temperature=0.0,
        google_api_key=api_key,
        request_timeout=float(os.getenv("LANGSMITH_JUDGE_TIMEOUT", "30")),
        retries=1,
    )


def chat_llm_judge(
    inputs: Mapping[str, Any],
    outputs: Any,
    reference_outputs: Any | None = None,
) -> dict[str, Any]:
    """Score a chat response with a Gemini judge and return LangSmith feedback."""

    response = get_judge_llm().invoke(
        build_chat_judge_prompt(inputs, outputs, reference_outputs)
    )
    result = parse_judge_response(_message_text(response))
    return {
        "key": "chat_llm_judge",
        "score": result["overall_score"],
        "comment": result["rationale"],
        "metadata": {
            "judge_model": os.getenv(JUDGE_MODEL_ENV, DEFAULT_JUDGE_MODEL),
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "criterion_scores": result["criteria"],
            "needs_human_review": result["needs_human_review"],
        },
    }
