"""Run CentaurDrug's curated LangSmith evaluation suite."""

from __future__ import annotations

import argparse
import os
from typing import Any

from dotenv import load_dotenv
from langsmith import Client, evaluate

from evals.langsmith_llm_judge import chat_llm_judge
from src.api.main import ChatRequest, chat


def chat_target(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run the application chat path without creating a second trace."""

    return chat(ChatRequest.model_validate(inputs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CentaurDrug LangSmith evaluations.")
    parser.add_argument(
        "--dataset",
        default="centaurdrug-chat-v1",
        help="LangSmith dataset name to evaluate.",
    )
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("LANGSMITH_API_KEY", "").strip():
        raise SystemExit("LANGSMITH_API_KEY is required to upload evaluation results.")
    if not os.getenv("GEMINI_API_KEY", "").strip():
        raise SystemExit("GEMINI_API_KEY is required to run the LLM judge.")

    client = Client(
        api_url=os.getenv("LANGSMITH_ENDPOINT") or None,
        workspace_id=os.getenv("LANGSMITH_WORKSPACE_ID", "").strip() or None,
    )
    results = evaluate(
        chat_target,
        data=args.dataset,
        evaluators=[chat_llm_judge],
        client=client,
        experiment_prefix="centaurdrug-chat-llm-judge",
        description="CentaurDrug chat usefulness and scientific-caution evaluation.",
        max_concurrency=1,
        error_handling="log",
    )
    print(f"Completed evaluation: {results.experiment_name}")


if __name__ == "__main__":
    main()
