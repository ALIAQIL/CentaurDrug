from __future__ import annotations

from contextlib import contextmanager

from src import observability


def test_request_trace_is_disabled_by_default(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    with observability.request_trace(
        "test.trace",
        inputs={"smiles": "CCO"},
        metadata={"endpoint": "test"},
        tags=["test"],
    ) as handle:
        assert handle.trace_id is None


def test_request_trace_fails_open_when_langsmith_cannot_start(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    observability.get_langsmith_client.cache_clear()

    @contextmanager
    def broken_trace(*args, **kwargs):
        raise RuntimeError("LangSmith unavailable")
        yield

    monkeypatch.setattr("langsmith.run_helpers.trace", broken_trace)

    with observability.request_trace(
        "test.trace",
        inputs={},
        metadata={},
        tags=[],
    ) as handle:
        assert handle.trace_id is None


def test_request_trace_captures_root_metadata(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    captured = {}

    class FakeRun:
        id = "trace-456"

        def add_metadata(self, metadata):
            captured["updated_metadata"] = metadata

        def add_tags(self, tags):
            captured["updated_tags"] = tags

    @contextmanager
    def fake_trace(**kwargs):
        captured.update(kwargs)
        yield FakeRun()

    monkeypatch.setattr("langsmith.run_helpers.trace", fake_trace)
    monkeypatch.setattr(observability, "get_langsmith_client", lambda: object())

    with observability.request_trace(
        "test.trace",
        inputs={"smiles": "CCO"},
        metadata={"endpoint": "test"},
        tags=["endpoint:test"],
    ) as handle:
        assert handle.trace_id == "trace-456"
        handle.add_metadata({"mode": "test"})
        handle.add_tags(["status:ok"])

    assert captured["name"] == "test.trace"
    assert captured["inputs"] == {"smiles": "CCO"}
    assert captured["metadata"] == {"endpoint": "test"}
    assert captured["tags"] == ["endpoint:test"]
    assert captured["updated_metadata"] == {"mode": "test"}
    assert captured["updated_tags"] == ["status:ok"]


def test_feedback_submission_is_best_effort(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")

    class FakeClient:
        def create_feedback(self, **kwargs):
            assert kwargs["trace_id"] == "trace-123"
            assert kwargs["score"] == 1

    monkeypatch.setattr(observability, "get_langsmith_client", lambda: FakeClient())

    assert observability.submit_feedback(
        "trace-123",
        key="chat_usefulness",
        score=1,
    ) is True
