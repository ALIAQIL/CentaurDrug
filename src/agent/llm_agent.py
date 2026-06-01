from __future__ import annotations

import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.agent.knowledge_base import (
    get_available_transformations,
    summarize_transformations_for_prompt,
)


load_dotenv()


class TransformationStrategy(BaseModel):
    strategy_summary: str = Field(
        description="Short explanation of the medicinal chemistry strategy."
    )
    selected_transformations: List[str] = Field(
        description="Transformation names to try next."
    )
    avoid_transformations: List[str] = Field(
        default_factory=list,
        description="Transformation names to avoid."
    )
    rationale: str = Field(
        description="Why these transformations were selected."
    )


def fallback_strategy(evaluation_summary: Dict[str, Any]) -> TransformationStrategy:
    """
    Deterministic fallback if Gemini key is missing or LLM call fails.
    """

    risks = evaluation_summary.get("main_risks", [])
    available = get_available_transformations()

    selected = []

    if (
        "poor_solubility" in risks
        or "high_lipophilicity" in risks
        or "cyp3a4_inhibition_risk" in risks
    ):
        for name in ["aromatic_H_to_OH", "aromatic_H_to_OMe", "ester_to_acid"]:
            if name in available:
                selected.append(name)

    if "herg_cardiotoxicity_risk" in risks:
        for name in ["aromatic_H_to_OH", "aromatic_H_to_OMe"]:
            if name in available and name not in selected:
                selected.append(name)

    if not selected:
        selected = [
            name
            for name in [
                "aromatic_H_to_F",
                "aromatic_H_to_OH",
                "aromatic_H_to_OMe",
            ]
            if name in available
        ]

    if not selected:
        selected = available[:3]

    return TransformationStrategy(
        strategy_summary="Fallback deterministic strategy.",
        selected_transformations=selected,
        avoid_transformations=[],
        rationale="Selected transformations using simple rule-based risk mapping.",
    )


class GeminiStrategyAgent:
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
    ):
        self.model = model
        self.temperature = temperature
        self.api_key = os.getenv("GEMINI_API_KEY")

        self._llm = None

        if self.api_key:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._llm = ChatGoogleGenerativeAI(
                model=self.model,
                temperature=self.temperature,
                google_api_key=self.api_key,
            ).with_structured_output(TransformationStrategy)

    def choose_strategy(
        self,
        evaluation_summary: Dict[str, Any],
        score_vector: Dict[str, Any],
    ) -> TransformationStrategy:
        if self._llm is None:
            return fallback_strategy(evaluation_summary)

        prompt = self._build_prompt(
            evaluation_summary=evaluation_summary,
            score_vector=score_vector,
        )

        try:
            result = self._llm.invoke(prompt)
            return self._sanitize_strategy(result)

        except Exception:
            return fallback_strategy(evaluation_summary)

    def _sanitize_strategy(
        self,
        strategy: TransformationStrategy,
    ) -> TransformationStrategy:
        available = set(get_available_transformations())

        avoid = [
            name
            for name in strategy.avoid_transformations
            if name in available
        ]

        avoid_set = set(avoid)
        selected = [
            name
            for name in strategy.selected_transformations
            if name in available and name not in avoid_set
        ]

        if not selected:
            return fallback_strategy(
                {
                    "main_risks": [],
                }
            )

        return TransformationStrategy(
            strategy_summary=strategy.strategy_summary,
            selected_transformations=selected,
            avoid_transformations=avoid,
            rationale=strategy.rationale,
        )

    def _build_prompt(
        self,
        evaluation_summary: Dict[str, Any],
        score_vector: Dict[str, Any],
    ) -> str:
        transformations = summarize_transformations_for_prompt()

        return f"""
You are the strategy module of CentaurDrug, an AI medicinal chemistry copilot.

You do not directly generate final molecules.
Your job is to choose which allowed RDKit transformations should be tried next.

You must preserve the core scaffold as much as possible.
You must consider ADMET tradeoffs.
You must not over-trust out-of-domain predictions.

Parent molecule evaluation summary:
{evaluation_summary}

Parent score vector:
{score_vector}

Available transformations:
{transformations}

Return a structured strategy selecting only transformations from the available list.

Rules:
- If poor solubility or high lipophilicity is present, prefer polarity-increasing transformations.
- If hERG risk is present, avoid transformations that strongly increase lipophilicity.
- If predictions are out of applicability domain, be cautious.
- Do not choose transformations that are not in the available list.
"""
