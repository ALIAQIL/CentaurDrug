from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.agent.scoring import build_score_vector, compare_score_vectors
from src.agent.transformations import generate_candidates
from src.tools.evaluator import ADMETPanelEvaluator


def extract_candidate_score(evaluation: Dict[str, Any]) -> float:
    """
    Higher score = better candidate.

    Kept as a compatibility wrapper around the shared scoring module.
    """

    return float(build_score_vector(evaluation)["scalar_score"])


def summarize_improvement(
    parent_eval: Dict[str, Any],
    candidate_eval: Dict[str, Any],
) -> Dict[str, Any]:
    parent_smiles = parent_eval.get("canonical_smiles") or parent_eval.get(
        "original_smiles"
    )
    parent_vector = build_score_vector(
        parent_eval,
        reference_smiles=parent_smiles,
    )
    candidate_vector = build_score_vector(
        candidate_eval,
        reference_smiles=parent_smiles,
    )

    comparison = compare_score_vectors(parent_vector, candidate_vector)
    comparison.update(
        {
            "parent_score": comparison["parent_scalar_score"],
            "candidate_score": comparison["candidate_scalar_score"],
            "delta_score": comparison["delta_scalar_score"],
        }
    )
    return comparison


class MoleculeOptimizer:
    def __init__(
        self,
        model_root: str | Path = "models/admet_xgboost",
    ):
        self.evaluator = ADMETPanelEvaluator(model_root=model_root)

    def optimize(
        self,
        smiles: str,
        max_candidates: int = 50,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        parent_eval = self.evaluator.evaluate_molecule(smiles)
        parent_reference_smiles = (
            parent_eval.get("canonical_smiles")
            or parent_eval.get("original_smiles")
            or smiles
        )
        parent_score_vector = build_score_vector(
            parent_eval,
            reference_smiles=parent_reference_smiles,
        )
        parent_score = float(parent_score_vector["scalar_score"])

        candidates = generate_candidates(
            smiles,
            max_total_candidates=max_candidates,
        )

        evaluated_candidates: List[Dict[str, Any]] = []

        for candidate in candidates:
            evaluation = self.evaluator.evaluate_molecule(candidate.smiles)

            score_vector = build_score_vector(
                evaluation,
                reference_smiles=parent_reference_smiles,
            )
            score = float(score_vector["scalar_score"])
            improvement = summarize_improvement(parent_eval, evaluation)

            evaluated_candidates.append(
                {
                    "smiles": candidate.smiles,
                    "parent_smiles": candidate.parent_smiles,
                    "transformation": candidate.transformation,
                    "score": score,
                    "score_vector": score_vector,
                    "improvement": improvement,
                    "evaluation": evaluation,
                }
            )

        evaluated_candidates = sorted(
            evaluated_candidates,
            key=lambda x: x["score"],
            reverse=True,
        )

        return {
            "status": "ok",
            "parent_smiles": smiles,
            "parent_score": parent_score,
            "parent_score_vector": parent_score_vector,
            "parent_evaluation": parent_eval,
            "n_generated_candidates": len(candidates),
            "n_evaluated_candidates": len(evaluated_candidates),
            "top_candidates": evaluated_candidates[:top_k],
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize a molecule using deterministic transformations and ADMET evaluation."
    )

    parser.add_argument("--smiles", required=True)
    parser.add_argument("--model-root", default="models/admet_xgboost")
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)

    args = parser.parse_args()

    optimizer = MoleculeOptimizer(model_root=args.model_root)

    result = optimizer.optimize(
        smiles=args.smiles,
        max_candidates=args.max_candidates,
        top_k=args.top_k,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
