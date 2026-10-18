from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.observability import (
    build_trace_metadata,
    request_trace,
    submit_feedback,
)
from src.agent.graph import run_graph
from src.prompts import (
    CHAT_PROMPT_VERSION,
    STRATEGY_PROMPT_VERSION,
    build_chat_prompt as render_chat_prompt,
)
from src.tools.evaluator import (
    ADMETPanelEvaluator,
    ModelArtifactError,
    assert_model_artifacts_ready,
    evaluate_rules,
    validate_model_artifacts,
)


DEFAULT_MODEL_ROOT = os.getenv(
    "CENTAURDRUG_MODEL_ROOT",
    "models/admet_xgboost",
)
STATIC_DIR = Path(__file__).resolve().parents[1] / "ui" / "static"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)

HTTP_REQUESTS = Counter(
    "centaurdrug_http_requests_total",
    "Total HTTP requests handled by the API.",
    ("method", "route", "status"),
)
HTTP_ERRORS = Counter(
    "centaurdrug_http_errors_total",
    "Total HTTP requests that returned a server error.",
    ("method", "route", "status"),
)
HTTP_DURATION = Histogram(
    "centaurdrug_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ("method", "route"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
CHAT_REQUESTS = Counter(
    "centaurdrug_chat_requests_total",
    "Chat requests by response mode and outcome.",
    ("mode", "outcome"),
)
OPTIMIZATION_REQUESTS = Counter(
    "centaurdrug_optimization_requests_total",
    "Optimization requests by outcome.",
    ("outcome",),
)
EVALUATION_REQUESTS = Counter(
    "centaurdrug_evaluation_requests_total",
    "ADMET evaluation requests by outcome.",
    ("outcome",),
)
READINESS = Gauge(
    "centaurdrug_readiness",
    "Whether the API model artifacts are ready to serve requests.",
)
BUILD_INFO = Gauge(
    "centaurdrug_build_info",
    "Build information for the CentaurDrug API.",
    ("version", "environment"),
)
BUILD_INFO.labels(
    version=os.getenv("CENTAURDRUG_APP_VERSION", "unknown"),
    environment=os.getenv("CENTAURDRUG_ENVIRONMENT", "local"),
).set(1)

app = FastAPI(
    title="CentaurDrug API",
    description="ADMET evaluation and agent-guided molecular optimization.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CENTAURDRUG_CORS_ORIGINS",
            "http://localhost:8000,http://127.0.0.1:8000",
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class MoleculeRequest(BaseModel):
    smiles: str = Field(..., min_length=1, max_length=1000, examples=["CCO"])


class EvaluateRequest(MoleculeRequest):
    model_root: str = Field(default=DEFAULT_MODEL_ROOT, max_length=240)


class OptimizationConstraints(BaseModel):
    avoid_substructures: List[str] = Field(default_factory=list, max_length=12)
    max_mw: float | None = Field(default=None, gt=0, le=2000)
    max_logp: float | None = Field(default=None, ge=-10, le=15)
    max_tpsa: float | None = Field(default=None, ge=0, le=400)
    min_scaffold_preservation: float | None = Field(default=None, ge=0, le=1)

    @field_validator("avoid_substructures")
    @classmethod
    def normalize_avoid_substructures(cls, values: List[str]) -> List[str]:
        return [value.strip() for value in values if value.strip()]


class OptimizeRequest(EvaluateRequest):
    max_depth: int = Field(default=2, ge=0, le=4)
    beam_width: int = Field(default=3, ge=1, le=20)
    max_candidates_per_node: int = Field(default=10, ge=1, le=100)
    include_full_tree: bool = False
    constraints: OptimizationConstraints = Field(
        default_factory=OptimizationConstraints
    )


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


class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., min_length=1, max_length=100)
    key: Literal["user_feedback", "chat_usefulness", "optimization_usefulness"] = (
        "user_feedback"
    )
    score: float = Field(..., ge=0, le=1)
    comment: str | None = Field(default=None, max_length=2000)
    candidate_smiles: str | None = Field(default=None, max_length=1000)


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
        request_timeout=float(os.getenv("GEMINI_REQUEST_TIMEOUT", "12")),
        retries=1,
    )


def require_models_on_startup() -> bool:
    return os.getenv("CENTAURDRUG_REQUIRE_MODELS_ON_STARTUP", "0").lower() in {
        "1",
        "true",
        "yes",
    }


@app.on_event("startup")
def validate_startup_model_artifacts() -> None:
    if not require_models_on_startup():
        return

    model_root = resolve_model_root(DEFAULT_MODEL_ROOT)
    assert_model_artifacts_ready(model_root)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration = time.perf_counter() - start
        route = metric_route(request)
        HTTP_REQUESTS.labels(request.method, route, "500").inc()
        HTTP_ERRORS.labels(request.method, route, "500").inc()
        HTTP_DURATION.labels(request.method, route).observe(duration)
        LOGGER.exception("Unhandled request error: %s %s", request.method, request.url.path)
        raise

    duration = time.perf_counter() - start
    route = metric_route(request)
    status = str(response.status_code)
    HTTP_REQUESTS.labels(request.method, route, status).inc()
    HTTP_DURATION.labels(request.method, route).observe(duration)

    if response.status_code >= 500:
        HTTP_ERRORS.labels(request.method, route, status).inc()

    response.headers["X-Process-Time"] = f"{duration:.4f}"
    LOGGER.info(
        "%s %s -> %s in %.3fs",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


def metric_route(request: Request) -> str:
    """Return a bounded route label without exposing request data in metrics."""
    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    if isinstance(route_path, str) and route_path:
        return route_path

    return "unmatched"


def allowed_model_roots() -> tuple[Path, ...]:
    roots = [PROJECT_ROOT / DEFAULT_MODEL_ROOT]
    extra_roots = os.getenv("CENTAURDRUG_ALLOWED_MODEL_ROOTS", "")

    for value in extra_roots.split(","):
        value = value.strip()
        if not value:
            continue

        path = Path(value)
        roots.append(path if path.is_absolute() else PROJECT_ROOT / path)

    return tuple(path.resolve() for path in roots)


def resolve_model_root(model_root: str) -> str:
    requested = Path(model_root)
    resolved = (
        requested if requested.is_absolute() else PROJECT_ROOT / requested
    ).resolve()

    if resolved not in allowed_model_roots():
        raise ValueError("model_root is not allowed for this API.")

    return str(resolved)


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


@app.get("/ready")
def ready() -> Dict[str, Any]:
    try:
        model_root = resolve_model_root(DEFAULT_MODEL_ROOT)
    except ValueError as exc:
        READINESS.set(0)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    readiness = validate_model_artifacts(model_root)

    if readiness["status"] != "ok":
        READINESS.set(0)
        raise HTTPException(status_code=503, detail=readiness)

    READINESS.set(1)
    return {
        "status": "ok",
        "service": "centaurdrug-api",
        "models": readiness,
    }


@app.get("/version")
def version() -> Dict[str, Any]:
    return {
        "service": "centaurdrug-api",
        "api_version": app.version,
        "default_model_root": DEFAULT_MODEL_ROOT,
    }


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/rules")
def rules(request: MoleculeRequest) -> Dict[str, Any]:
    return evaluate_rules(request.smiles)


@app.post("/evaluate")
def evaluate(request: EvaluateRequest) -> Dict[str, Any]:
    try:
        model_root = resolve_model_root(request.model_root)
        evaluator = get_evaluator(model_root)
        result = evaluator.evaluate_molecule(request.smiles)
        EVALUATION_REQUESTS.labels("ok").inc()
        return result
    except ValueError as exc:
        EVALUATION_REQUESTS.labels("client_error").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelArtifactError as exc:
        EVALUATION_REQUESTS.labels("dependency_error").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        EVALUATION_REQUESTS.labels("error").inc()
        LOGGER.exception("Evaluation failed for submitted molecule.")
        raise HTTPException(
            status_code=500,
            detail="Evaluation failed. Check the molecule and model artifacts, then try again.",
        ) from exc


@app.post("/optimize")
def optimize(request: OptimizeRequest) -> Dict[str, Any]:
    constraints = request.constraints.model_dump(exclude_none=True)
    trace_metadata = build_trace_metadata(
        "optimize",
        prompt_version=STRATEGY_PROMPT_VERSION,
        gemini_model="gemini-2.5-flash",
        model_root=request.model_root,
        max_depth=request.max_depth,
        beam_width=request.beam_width,
        max_candidates_per_node=request.max_candidates_per_node,
        constraints=constraints,
    )
    trace_inputs = {
        "smiles": request.smiles,
        "model_root": request.model_root,
        "max_depth": request.max_depth,
        "beam_width": request.beam_width,
        "max_candidates_per_node": request.max_candidates_per_node,
        "constraints": constraints,
    }

    with request_trace(
        "centaurdrug.optimize",
        inputs=trace_inputs,
        metadata=trace_metadata,
        tags=[
            "endpoint:optimize",
            f"environment:{trace_metadata['environment']}",
        ],
    ) as trace_handle:
        try:
            model_root = resolve_model_root(request.model_root)
            result = run_graph(
                smiles=request.smiles,
                model_root=model_root,
                max_depth=request.max_depth,
                beam_width=request.beam_width,
                max_candidates_per_node=request.max_candidates_per_node,
                include_full_tree=request.include_full_tree,
                constraints=constraints,
            )
            result = dict(result)
            result["trace_id"] = trace_handle.trace_id
            OPTIMIZATION_REQUESTS.labels("ok").inc()
            return result
        except ValueError as exc:
            OPTIMIZATION_REQUESTS.labels("client_error").inc()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ModelArtifactError as exc:
            OPTIMIZATION_REQUESTS.labels("dependency_error").inc()
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            OPTIMIZATION_REQUESTS.labels("error").inc()
            trace_handle.add_metadata({"error": str(exc)[:500]})
            trace_handle.add_tags(["status:error"])
            LOGGER.exception("Optimization failed for submitted molecule.")
            raise HTTPException(
                status_code=500,
                detail="Optimization failed. Check the molecule and optimization settings, then try again.",
            ) from exc


@app.post("/chat")
def chat(request: ChatRequest) -> Dict[str, Any]:
    prompt = build_chat_prompt(request)
    compact_context = compact_chat_context(request.context)
    trace_metadata = build_trace_metadata(
        "chat",
        prompt_version=CHAT_PROMPT_VERSION,
        gemini_model="gemini-2.5-flash",
        context_type=request.context.get("type"),
    )

    with request_trace(
        "centaurdrug.chat",
        inputs={
            "smiles": request.smiles,
            "prompt": request.prompt,
            "context": compact_context,
            "messages": [message.model_dump() for message in request.messages],
        },
        metadata=trace_metadata,
        tags=[
            "endpoint:chat",
            f"environment:{trace_metadata['environment']}",
        ],
    ) as trace_handle:
        llm = get_chat_llm()

        if llm is None:
            trace_handle.add_metadata({"mode": "fallback"})
            trace_handle.add_tags(["mode:fallback"])
            CHAT_REQUESTS.labels("fallback", "ok").inc()
            return {
                "status": "ok",
                "mode": "fallback",
                "reply": build_fallback_chat_reply(request),
                "suggested_prompts": build_suggested_prompts(request.context),
                "trace_id": trace_handle.trace_id,
            }

        try:
            result = llm.invoke(prompt)
            trace_handle.add_metadata({"mode": "gemini"})
            trace_handle.add_tags(["mode:gemini"])
            CHAT_REQUESTS.labels("gemini", "ok").inc()
            return {
                "status": "ok",
                "mode": "gemini",
                "reply": extract_llm_text(result),
                "suggested_prompts": build_suggested_prompts(request.context),
                "trace_id": trace_handle.trace_id,
            }
        except Exception as exc:
            CHAT_REQUESTS.labels("fallback", "error").inc()
            trace_handle.add_metadata({"mode": "fallback", "error": str(exc)[:500]})
            trace_handle.add_tags(["mode:fallback", "status:error"])
            LOGGER.exception("Gemini chat call failed; using deterministic fallback.")
            return {
                "status": "ok",
                "mode": "fallback",
                "reply": build_fallback_chat_reply(request),
                "suggested_prompts": build_suggested_prompts(request.context),
                "trace_id": trace_handle.trace_id,
            }


def build_chat_prompt(request: ChatRequest) -> str:
    history = "\n".join(
        f"{message.role}: {message.content}"
        for message in request.messages[-8:]
    )
    context = compact_chat_context(request.context)

    return render_chat_prompt(
        smiles=request.smiles or "not provided",
        context=context,
        history=history,
        prompt=request.prompt,
    )


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> Dict[str, Any]:
    submitted = submit_feedback(
        request.trace_id,
        key=request.key,
        score=request.score,
        comment=request.comment,
        extra={
            "candidate_smiles": request.candidate_smiles,
        },
    )
    return {
        "status": "ok",
        "submitted": submitted,
    }


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
