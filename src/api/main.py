from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.agent.graph import run_graph
from src.tools.evaluator import ADMETPanelEvaluator, evaluate_rules


DEFAULT_MODEL_ROOT = "models/admet_xgboost"
STATIC_DIR = Path(__file__).resolve().parents[1] / "ui" / "static"

app = FastAPI(
    title="CentaurDrug API",
    description="ADMET evaluation and agent-guided molecular optimization.",
    version="0.2.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class MoleculeRequest(BaseModel):
    smiles: str = Field(..., min_length=1, examples=["CCO"])


class EvaluateRequest(MoleculeRequest):
    model_root: str = DEFAULT_MODEL_ROOT


class OptimizeRequest(EvaluateRequest):
    max_depth: int = Field(default=2, ge=0, le=4)
    beam_width: int = Field(default=3, ge=1, le=20)
    max_candidates_per_node: int = Field(default=10, ge=1, le=100)
    include_full_tree: bool = False


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"] = "user"
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    smiles: str | None = Field(default=None, max_length=500)
    prompt: str = Field(..., min_length=1, max_length=4000)
    context: Dict[str, Any] = Field(default_factory=dict)
    messages: List[ChatMessage] = Field(default_factory=list, max_length=12)

    @field_validator("smiles")
    @classmethod
    def normalize_blank_smiles(cls, value: str | None) -> str | None:
        if value is None:
            return None

        stripped = value.strip()
        return stripped or None


@lru_cache(maxsize=4)
def get_evaluator(model_root: str = DEFAULT_MODEL_ROOT) -> ADMETPanelEvaluator:
    return ADMETPanelEvaluator(model_root=model_root)


@lru_cache(maxsize=1)
def get_chat_llm():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=api_key,
    )


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "centaurdrug-api",
        "default_model_root": DEFAULT_MODEL_ROOT,
    }


@app.post("/rules")
def rules(request: MoleculeRequest) -> Dict[str, Any]:
    return evaluate_rules(request.smiles)


@app.post("/evaluate")
def evaluate(request: EvaluateRequest) -> Dict[str, Any]:
    try:
        evaluator = get_evaluator(request.model_root)
        return evaluator.evaluate_molecule(request.smiles)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {exc}",
        ) from exc


@app.post("/optimize")
def optimize(request: OptimizeRequest) -> Dict[str, Any]:
    try:
        return run_graph(
            smiles=request.smiles,
            model_root=request.model_root,
            max_depth=request.max_depth,
            beam_width=request.beam_width,
            max_candidates_per_node=request.max_candidates_per_node,
            include_full_tree=request.include_full_tree,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {exc}",
        ) from exc


@app.post("/chat")
def chat(request: ChatRequest) -> Dict[str, Any]:
    prompt = build_chat_prompt(request)
    llm = get_chat_llm()

    if llm is None:
        return {
            "status": "ok",
            "mode": "fallback",
            "reply": build_fallback_chat_reply(request),
            "suggested_prompts": build_suggested_prompts(request.context),
        }

    try:
        result = llm.invoke(prompt)
        return {
            "status": "ok",
            "mode": "gemini",
            "reply": extract_llm_text(result),
            "suggested_prompts": build_suggested_prompts(request.context),
        }
    except Exception:
        return {
            "status": "ok",
            "mode": "fallback",
            "reply": build_fallback_chat_reply(request),
            "suggested_prompts": build_suggested_prompts(request.context),
        }


def build_chat_prompt(request: ChatRequest) -> str:
    history = "\n".join(
        f"{message.role}: {message.content}"
        for message in request.messages[-8:]
    )
    context = compact_chat_context(request.context)

    return f"""
You are the CentaurDrug medicinal chemistry copilot.

Your job is to explain the currently selected molecule or candidate in a careful,
scientific, and demo-friendly way.

Rules:
- Do not claim that a molecule is clinically safe or effective.
- Treat ADMET predictions as computational estimates.
- Mention applicability-domain risk when present.
- Prefer concise answers with concrete next questions or actions.
- If the user selected a generated candidate, compare it to the parent.

Current SMILES:
{request.smiles or "not provided"}

Current context:
{context}

Recent chat:
{history or "No previous chat."}

User prompt:
{request.prompt}
"""


def compact_chat_context(context: Dict[str, Any]) -> Dict[str, Any]:
    candidate = context.get("candidate") or {}
    evaluation = context.get("evaluation") or {}
    optimization = context.get("optimization") or {}

    return {
        "type": context.get("type"),
        "smiles": context.get("smiles"),
        "decision": (
            context.get("decision")
            or evaluation.get("final_decision", {}).get("decision")
            or candidate.get("decision")
        ),
        "main_risks": (
            context.get("main_risks")
            or evaluation.get("overall_assessment", {}).get("main_risks")
            or candidate.get("main_risks")
        ),
        "candidate": {
            "smiles": candidate.get("smiles"),
            "transformation": candidate.get("transformation"),
            "scalar_score": candidate.get("scalar_score"),
            "delta_vs_parent": candidate.get("delta_vs_parent"),
            "score_vector": candidate.get("score_vector"),
            "improvements": candidate.get("improvements"),
            "tradeoffs": candidate.get("tradeoffs"),
            "scaffold": candidate.get("scaffold"),
            "synthetic_accessibility": candidate.get("synthetic_accessibility"),
        },
        "optimization": {
            "n_nodes": optimization.get("n_nodes"),
            "n_candidate_nodes": optimization.get("n_candidate_nodes"),
            "last_strategy": optimization.get("last_strategy"),
        },
    }


def build_fallback_chat_reply(request: ChatRequest) -> str:
    context = compact_chat_context(request.context)
    candidate = context.get("candidate") or {}
    smiles = (
        candidate.get("smiles")
        or context.get("smiles")
        or request.smiles
        or "the current molecule"
    )
    decision = context.get("decision") or "not available"
    risks = context.get("main_risks") or []
    improvements = candidate.get("improvements") or []
    tradeoffs = candidate.get("tradeoffs") or []

    parts = [
        f"I am looking at `{smiles}`.",
        f"Current decision: `{decision}`.",
    ]

    if candidate.get("transformation"):
        parts.append(
            f"It was generated by `{candidate['transformation']}` "
            f"with score {candidate.get('scalar_score', 'n/a')}."
        )

    if candidate.get("delta_vs_parent") is not None:
        parts.append(
            f"Delta vs parent: {candidate['delta_vs_parent']}. "
            "Positive means better under the current heuristic score."
        )

    if improvements:
        parts.append("Main improvements: " + "; ".join(improvements[:3]) + ".")

    if tradeoffs:
        parts.append("Main tradeoffs: " + "; ".join(tradeoffs[:3]) + ".")

    if risks:
        parts.append("Risk flags: " + ", ".join(risks) + ".")

    parts.append(
        "Use this as computational triage, not a final medicinal chemistry decision."
    )

    return " ".join(parts)


def build_suggested_prompts(context: Dict[str, Any]) -> List[str]:
    if context.get("type") == "candidate":
        return [
            "Why did this candidate rank here?",
            "What improved compared with the parent?",
            "What is the biggest tradeoff for this candidate?",
        ]

    return [
        "What are the main ADMET risks?",
        "Which transformations should we try next?",
        "Explain this molecule for a project demo.",
    ]


def extract_llm_text(result: Any) -> str:
    content = getattr(result, "content", result)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)
