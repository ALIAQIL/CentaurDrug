from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, TypedDict

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from langgraph.graph import END, START, StateGraph

from src.agent.constraints import evaluate_candidate_constraints
from src.agent.llm_agent import GeminiStrategyAgent
from src.agent.scoring import (
    build_candidate_explanation,
    build_score_vector,
    select_diverse_items,
)
from src.agent.search_tree import OptimizationTree
from src.agent.transformations import generate_candidates
from src.tools.evaluator import ADMETPanelEvaluator


@lru_cache(maxsize=4)
def get_graph_evaluator(model_root: str) -> ADMETPanelEvaluator:
    return ADMETPanelEvaluator(model_root=model_root)


class OptimizationState(TypedDict, total=False):
    smiles: str
    model_root: str
    max_depth: int
    beam_width: int
    max_candidates_per_node: int
    include_full_tree: bool
    constraints: Dict[str, Any]

    current_depth: int
    tree: OptimizationTree
    frontier_node_ids: List[str]

    last_strategy: Dict[str, Any]
    messages: List[str]
    final_result: Dict[str, Any]


class CentaurDrugGraph:
    def __init__(self):
        self.evaluator: Optional[ADMETPanelEvaluator] = None
        self.strategy_agent = GeminiStrategyAgent()

    def build(self):
        graph = StateGraph(OptimizationState)

        graph.add_node("initialize", self.initialize)
        graph.add_node("choose_strategy", self.choose_strategy)
        graph.add_node("expand_frontier", self.expand_frontier)
        graph.add_node("select_next_frontier", self.select_next_frontier)
        graph.add_node("finalize", self.finalize)

        graph.add_edge(START, "initialize")
        graph.add_conditional_edges(
            "initialize",
            self.should_continue,
            {
                "continue": "choose_strategy",
                "stop": "finalize",
            },
        )
        graph.add_edge("choose_strategy", "expand_frontier")
        graph.add_edge("expand_frontier", "select_next_frontier")

        graph.add_conditional_edges(
            "select_next_frontier",
            self.should_continue,
            {
                "continue": "choose_strategy",
                "stop": "finalize",
            },
        )

        graph.add_edge("finalize", END)

        return graph.compile()

    def initialize(self, state: OptimizationState) -> Dict[str, Any]:
        model_root = state.get("model_root", "models/admet_xgboost")
        self.evaluator = get_graph_evaluator(model_root)

        smiles = state["smiles"]

        evaluation = self.evaluator.evaluate_molecule(smiles)
        node_smiles = evaluation.get("canonical_smiles", smiles)
        score_vector = build_score_vector(
            evaluation,
            reference_smiles=node_smiles,
        )

        tree = OptimizationTree()
        root_id = tree.add_root(
            smiles=node_smiles,
            evaluation=evaluation,
            score_vector=score_vector,
        )

        return {
            "tree": tree,
            "frontier_node_ids": [root_id],
            "current_depth": 0,
            "messages": [
                "Initialized optimization tree with parent molecule."
            ],
        }

    def choose_strategy(self, state: OptimizationState) -> Dict[str, Any]:
        tree = state["tree"]
        frontier_ids = state["frontier_node_ids"]

        if not frontier_ids:
            return {
                "last_strategy": {
                    "strategy_summary": "No frontier nodes available.",
                    "selected_transformations": [],
                    "avoid_transformations": [],
                    "rationale": "Search cannot continue.",
                }
            }

        # For now, use the best frontier node as the LLM context.
        best_node = sorted(
            [tree.get_node(node_id) for node_id in frontier_ids],
            key=lambda node: node.scalar_score,
            reverse=True,
        )[0]

        evaluation_summary = best_node.evaluation.get(
            "overall_assessment",
            {}
        )

        strategy = self.strategy_agent.choose_strategy(
            evaluation_summary=evaluation_summary,
            score_vector=best_node.score_vector,
        )

        return {
            "last_strategy": strategy.model_dump(),
            "messages": state.get("messages", [])
            + [
                f"Strategy selected: {strategy.strategy_summary}"
            ],
        }

    def expand_frontier(self, state: OptimizationState) -> Dict[str, Any]:
        tree = state["tree"]
        frontier_ids = state["frontier_node_ids"]
        strategy = state["last_strategy"]

        selected_transformations = strategy.get(
            "selected_transformations",
            [],
        )
        avoid_transformations = set(strategy.get("avoid_transformations", []))
        selected_transformations = [
            name
            for name in selected_transformations
            if name not in avoid_transformations
        ]

        max_candidates_per_node = int(
            state.get("max_candidates_per_node", 20)
        )
        constraints = state.get("constraints") or {}

        new_node_ids = []
        skipped_by_constraints = 0
        seen_smiles = tree.get_smiles_set()

        if self.evaluator is None:
            raise RuntimeError("Graph evaluator has not been initialized.")

        root_smiles = self._get_root_smiles(tree)

        for node_id in frontier_ids:
            parent_node = tree.get_node(node_id)

            candidates = generate_candidates(
                parent_node.smiles,
                transformations=selected_transformations,
                max_total_candidates=max_candidates_per_node,
                excluded_smiles=seen_smiles,
            )

            for candidate in candidates:
                if candidate.smiles in seen_smiles:
                    continue

                constraint_result = evaluate_candidate_constraints(
                    candidate.smiles,
                    constraints=constraints,
                    reference_smiles=root_smiles,
                )
                if not constraint_result["passed"]:
                    skipped_by_constraints += 1
                    continue

                evaluation = self.evaluator.evaluate_molecule(
                    candidate.smiles
                )
                evaluation["constraints"] = constraint_result
                node_smiles = evaluation.get(
                    "canonical_smiles",
                    candidate.smiles,
                )

                if node_smiles in seen_smiles:
                    continue

                score_vector = build_score_vector(
                    evaluation,
                    reference_smiles=root_smiles,
                )

                child_id = tree.add_child(
                    parent_id=node_id,
                    smiles=node_smiles,
                    transformation=candidate.transformation,
                    evaluation=evaluation,
                    score_vector=score_vector,
                    llm_rationale=strategy.get("rationale"),
                )

                seen_smiles.add(node_smiles)
                new_node_ids.append(child_id)

        return {
            "frontier_node_ids": new_node_ids,
            "messages": state.get("messages", [])
            + [
                f"Expanded frontier and generated {len(new_node_ids)} new nodes."
                f" Skipped {skipped_by_constraints} candidates by constraints."
            ],
        }

    def select_next_frontier(
        self,
        state: OptimizationState,
    ) -> Dict[str, Any]:
        tree = state["tree"]
        current_depth = int(state.get("current_depth", 0)) + 1
        beam_width = int(state.get("beam_width", 3))

        candidates_at_depth = tree.get_frontier(
            depth=current_depth,
            top_k=beam_width,
        )

        frontier_node_ids = [node.node_id for node in candidates_at_depth]

        return {
            "current_depth": current_depth,
            "frontier_node_ids": frontier_node_ids,
            "messages": state.get("messages", [])
            + [
                f"Selected {len(frontier_node_ids)} frontier nodes at depth {current_depth}."
            ],
        }

    def should_continue(self, state: OptimizationState) -> str:
        current_depth = int(state.get("current_depth", 0))
        max_depth = int(state.get("max_depth", 2))
        frontier = state.get("frontier_node_ids", [])

        if current_depth >= max_depth:
            return "stop"

        if not frontier:
            return "stop"

        return "continue"

    def finalize(self, state: OptimizationState) -> Dict[str, Any]:
        tree = state["tree"]
        root_node = tree.get_node(tree.root_id) if tree.root_id else None
        best_nodes = tree.get_best_nodes(top_k=10)
        best_candidate_pool = tree.get_best_nodes(top_k=50, min_depth=1)
        best_candidate_nodes = select_diverse_items(
            best_candidate_pool,
            top_k=10,
            smiles_getter=lambda node: node.smiles,
            score_getter=lambda node: node.scalar_score,
        )

        final_result = {
            "status": "ok",
            "input_smiles": state["smiles"],
            "max_depth": state.get("max_depth", 2),
            "beam_width": state.get("beam_width", 3),
            "n_nodes": len(tree.nodes),
            "n_candidate_nodes": max(0, len(tree.nodes) - 1),
            "parent_evaluation": root_node.evaluation if root_node else None,
            "parent_score_vector": root_node.score_vector if root_node else None,
            "constraints": state.get("constraints") or {},
            "last_strategy": state.get("last_strategy"),
            "messages": state.get("messages", []),
            "best_nodes": [
                self._build_node_summary(tree, node)
                for node in best_nodes
            ],
            "best_candidate_nodes": [
                self._build_node_summary(tree, node)
                for node in best_candidate_nodes
            ],
            "tree_summary": {
                "root_id": tree.root_id,
                "nodes": {
                    node_id: self._build_node_summary(tree, node)
                    for node_id, node in tree.nodes.items()
                },
            },
        }

        if state.get("include_full_tree", False):
            final_result["tree"] = tree.to_dict()

        return {
            "final_result": final_result,
        }

    def _build_node_summary(
        self,
        tree: OptimizationTree,
        node,
    ) -> Dict[str, Any]:
        summary = node.to_summary_dict()
        raw = node.score_vector.get("raw", {})
        normalized = node.score_vector.get("normalized", {})

        summary.update(
            {
                "score_vector": normalized,
                "scaffold": raw.get("scaffold"),
                "synthetic_accessibility": raw.get(
                    "synthetic_accessibility"
                ),
                "constraints": node.evaluation.get("constraints"),
            }
        )

        if node.parent_id is None:
            summary.update(
                {
                    "delta_vs_parent": 0.0,
                    "parent_comparison": None,
                    "improvements": [],
                    "tradeoffs": [],
                }
            )
            return summary

        parent_node = tree.get_node(node.parent_id)
        explanation = build_candidate_explanation(
            parent_vector=parent_node.score_vector,
            candidate_vector=node.score_vector,
            candidate_evaluation=node.evaluation,
        )

        summary.update(
            {
                "delta_vs_parent": explanation["delta_vs_parent"],
                "parent_comparison": explanation["parent_comparison"],
                "improvements": explanation["improvements"],
                "tradeoffs": explanation["tradeoffs"],
            }
        )

        if tree.root_id and node.parent_id != tree.root_id:
            root_node = tree.get_node(tree.root_id)
            root_explanation = build_candidate_explanation(
                parent_vector=root_node.score_vector,
                candidate_vector=node.score_vector,
                candidate_evaluation=node.evaluation,
            )
            summary["root_comparison"] = root_explanation[
                "parent_comparison"
            ]
            summary["delta_vs_root"] = root_explanation["delta_vs_parent"]

        return summary

    @staticmethod
    def _get_root_smiles(tree: OptimizationTree) -> str | None:
        if tree.root_id is None:
            return None

        return tree.get_node(tree.root_id).smiles


def run_graph(
    smiles: str,
    model_root: str = "models/admet_xgboost",
    max_depth: int = 2,
    beam_width: int = 3,
    max_candidates_per_node: int = 20,
    include_full_tree: bool = False,
    constraints: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    app = CentaurDrugGraph().build()

    result = app.invoke(
        {
            "smiles": smiles,
            "model_root": model_root,
            "max_depth": max_depth,
            "beam_width": beam_width,
            "max_candidates_per_node": max_candidates_per_node,
            "include_full_tree": include_full_tree,
            "constraints": constraints or {},
        }
    )

    return result["final_result"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CentaurDrug LangGraph optimization loop."
    )

    parser.add_argument("--smiles", required=True)
    parser.add_argument("--model-root", default="models/admet_xgboost")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-candidates-per-node", type=int, default=20)
    parser.add_argument("--include-full-tree", action="store_true")

    args = parser.parse_args()

    result = run_graph(
        smiles=args.smiles,
        model_root=args.model_root,
        max_depth=args.max_depth,
        beam_width=args.beam_width,
        max_candidates_per_node=args.max_candidates_per_node,
        include_full_tree=args.include_full_tree,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
