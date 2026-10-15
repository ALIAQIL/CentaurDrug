from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import Client


CASES_PATH = Path(__file__).with_name("cases.json")


def load_cases() -> list[dict[str, Any]]:
    return json.loads(CASES_PATH.read_text())


def seed_datasets(client: Client) -> dict[str, str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in load_cases():
        grouped[case["dataset"]].append(case)

    dataset_ids: dict[str, str] = {}
    for name, cases in grouped.items():
        if client.has_dataset(dataset_name=name):
            dataset = client.read_dataset(dataset_name=name)
            dataset_ids[name] = str(dataset.id)
            continue
        else:
            dataset = client.create_dataset(
                dataset_name=name,
                description=f"CentaurDrug curated evaluation cases for {name}.",
            )
        client.create_examples(
            examples=[
                {
                    "inputs": case["inputs"],
                    "outputs": {},
                    "metadata": case.get("metadata", {}),
                }
                for case in cases
            ],
            dataset_id=dataset.id,
        )
        dataset_ids[name] = str(dataset.id)

    return dataset_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed CentaurDrug LangSmith datasets.")
    parser.parse_args()
    load_dotenv()
    client = Client(
        workspace_id=os.getenv("LANGSMITH_WORKSPACE_ID", "").strip() or None,
    )
    try:
        dataset_ids = seed_datasets(client)
    except Exception as exc:
        raise SystemExit(
            "LangSmith dataset access failed. If this is a 403, set "
            "LANGSMITH_WORKSPACE_ID to the workspace ID from LangSmith Settings "
            "and verify that the API key can create datasets."
        ) from exc
    for name, dataset_id in dataset_ids.items():
        print(f"{name}: {dataset_id}")


if __name__ == "__main__":
    main()
