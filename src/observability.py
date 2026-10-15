from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterator, Mapping

from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)
load_dotenv()
os.environ.setdefault(
    "LANGCHAIN_TRACING_V2",
    os.getenv("LANGSMITH_TRACING", "false"),
)


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def environment_name() -> str:
    value = os.getenv("CENTAURDRUG_ENVIRONMENT", "local").strip().lower()
    return value if value in {"local", "staging", "production"} else "local"


def project_name() -> str:
    configured = os.getenv(
        "LANGSMITH_PROJECT",
        f"centaurdrug-{environment_name()}",
    ).strip()
    return configured or f"centaurdrug-{environment_name()}"


def app_version() -> str:
    return os.getenv("CENTAURDRUG_APP_VERSION", "local").strip() or "local"


def workspace_id() -> str | None:
    value = os.getenv("LANGSMITH_WORKSPACE_ID", "").strip()
    return value or None


def tracing_enabled() -> bool:
    return _is_truthy(os.getenv("LANGSMITH_TRACING")) and bool(
        os.getenv("LANGSMITH_API_KEY", "").strip()
    )


@lru_cache(maxsize=1)
def get_langsmith_client():
    if not tracing_enabled():
        return None

    try:
        from langsmith import Client

        endpoint = os.getenv(
            "LANGSMITH_ENDPOINT",
            "https://api.smith.langchain.com",
        ).strip() or "https://api.smith.langchain.com"
        return Client(
            api_key=os.getenv("LANGSMITH_API_KEY"),
            api_url=endpoint,
            workspace_id=workspace_id(),
        )
    except Exception:
        LOGGER.exception("LangSmith client initialization failed.")
        return None


@dataclass
class TraceHandle:
    trace_id: str | None = None
    _run: Any = None

    def add_metadata(self, metadata: Mapping[str, Any]) -> None:
        if self._run is None:
            return
        try:
            self._run.add_metadata(dict(metadata))
        except Exception:
            LOGGER.exception("Failed to update LangSmith trace metadata.")

    def add_tags(self, tags: list[str]) -> None:
        if self._run is None:
            return
        try:
            self._run.add_tags(tags)
        except Exception:
            LOGGER.exception("Failed to update LangSmith trace tags.")


def build_trace_metadata(
    endpoint: str,
    *,
    prompt_version: str | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "endpoint": endpoint,
        "environment": environment_name(),
        "app_version": app_version(),
    }
    if prompt_version:
        result["prompt_version"] = prompt_version
    result.update(metadata)
    return result


@contextmanager
def request_trace(
    name: str,
    *,
    inputs: Mapping[str, Any],
    metadata: Mapping[str, Any],
    tags: list[str],
) -> Iterator[TraceHandle]:
    """Create a best-effort root trace without affecting application behavior."""

    if not tracing_enabled():
        yield TraceHandle()
        return

    try:
        from langsmith.run_helpers import trace

        trace_context = trace(
            name=name,
            run_type="chain",
            inputs=dict(inputs),
            project_name=project_name(),
            metadata=dict(metadata),
            tags=list(tags),
            client=get_langsmith_client(),
        )
        run = trace_context.__enter__()
    except Exception:
        LOGGER.exception("LangSmith root trace could not be started.")
        yield TraceHandle()
        return

    handle = TraceHandle(
        trace_id=str(getattr(run, "id", "")) or None,
        _run=run,
    )

    try:
        yield handle
    except BaseException as exc:
        try:
            trace_context.__exit__(type(exc), exc, exc.__traceback__)
        except Exception:
            LOGGER.exception("LangSmith trace finalization failed after an error.")
        raise
    else:
        try:
            trace_context.__exit__(None, None, None)
        except Exception:
            LOGGER.exception("LangSmith trace finalization failed.")


def submit_feedback(
    trace_id: str | None,
    *,
    key: str,
    score: float,
    comment: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> bool:
    if not trace_id or not tracing_enabled():
        return False

    client = get_langsmith_client()
    if client is None:
        return False

    try:
        client.create_feedback(
            trace_id=trace_id,
            key=key,
            score=score,
            comment=comment,
            extra=dict(extra or {}),
        )
        return True
    except Exception:
        LOGGER.exception("LangSmith feedback submission failed.")
        return False
