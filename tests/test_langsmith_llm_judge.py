import pytest

from evals.langsmith_llm_judge import (
    build_chat_judge_prompt,
    parse_judge_response,
)


def test_parse_judge_response_accepts_json_fence_and_scores():
    result = parse_judge_response(
        """```json
        {
          "criteria": {
            "usefulness": 1,
            "scientific_caution": 0.8,
            "clarity": 0.9,
            "applicability_domain": 0.7
          },
          "rationale": "The response is useful and appropriately cautious.",
          "needs_human_review": true
        }
        ```"""
    )

    assert result["overall_score"] == pytest.approx(0.85)
    assert result["needs_human_review"] is True


def test_parse_judge_response_rejects_missing_criterion():
    with pytest.raises(ValueError, match="missing score"):
        parse_judge_response(
            '{"criteria": {"usefulness": 1}, "rationale": "Too short."}'
        )


def test_build_chat_judge_prompt_contains_inputs_and_rubric():
    prompt = build_chat_judge_prompt(
        {"prompt": "What are the risks?"},
        {"reply": "This is a computational estimate."},
    )

    assert "scientific_caution" in prompt
    assert "What are the risks?" in prompt
    assert "Return JSON only" in prompt
