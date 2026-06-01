from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class OptimizationNode:
    node_id: str
    smiles: str
    parent_id: Optional[str]
    depth: int
    transformation: Optional[str]
    evaluation: Dict[str, Any]
    score_vector: Dict[str, Any]
    children: List[str] = field(default_factory=list)
    llm_rationale: Optional[str] = None
    human_status: str = "pending"

    @property
    def scalar_score(self) -> float:
        return float(self.score_vector.get("scalar_score", 0.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "smiles": self.smiles,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "transformation": self.transformation,
            "evaluation": self.evaluation,
            "score_vector": self.score_vector,
            "scalar_score": self.scalar_score,
            "children": self.children,
            "llm_rationale": self.llm_rationale,
            "human_status": self.human_status,
        }

    def to_summary_dict(self) -> Dict[str, Any]:
        decision = self.evaluation.get("final_decision", {}).get("decision")
        risks = self.evaluation.get("overall_assessment", {}).get("main_risks", [])

        return {
            "node_id": self.node_id,
            "smiles": self.smiles,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "transformation": self.transformation,
            "scalar_score": self.scalar_score,
            "decision": decision,
            "main_risks": risks,
            "children": self.children,
            "human_status": self.human_status,
        }


class OptimizationTree:
    def __init__(self):
        self.nodes: Dict[str, OptimizationNode] = {}
        self.root_id: Optional[str] = None

    def add_root(
        self,
        smiles: str,
        evaluation: Dict[str, Any],
        score_vector: Dict[str, Any],
    ) -> str:
        node_id = self._new_node_id()

        node = OptimizationNode(
            node_id=node_id,
            smiles=smiles,
            parent_id=None,
            depth=0,
            transformation=None,
            evaluation=evaluation,
            score_vector=score_vector,
        )

        self.nodes[node_id] = node
        self.root_id = node_id

        return node_id

    def add_child(
        self,
        parent_id: str,
        smiles: str,
        transformation: str,
        evaluation: Dict[str, Any],
        score_vector: Dict[str, Any],
        llm_rationale: Optional[str] = None,
    ) -> str:
        if parent_id not in self.nodes:
            raise ValueError(f"Unknown parent_id: {parent_id}")

        parent = self.nodes[parent_id]
        node_id = self._new_node_id()

        node = OptimizationNode(
            node_id=node_id,
            smiles=smiles,
            parent_id=parent_id,
            depth=parent.depth + 1,
            transformation=transformation,
            evaluation=evaluation,
            score_vector=score_vector,
            llm_rationale=llm_rationale,
        )

        self.nodes[node_id] = node
        parent.children.append(node_id)

        return node_id

    def get_node(self, node_id: str) -> OptimizationNode:
        return self.nodes[node_id]

    def has_smiles(self, smiles: str) -> bool:
        return any(node.smiles == smiles for node in self.nodes.values())

    def get_smiles_set(self) -> Set[str]:
        return {node.smiles for node in self.nodes.values()}

    def get_best_nodes(
        self,
        top_k: int = 5,
        max_depth: Optional[int] = None,
        min_depth: Optional[int] = None,
    ) -> List[OptimizationNode]:
        nodes = list(self.nodes.values())

        if max_depth is not None:
            nodes = [node for node in nodes if node.depth <= max_depth]

        if min_depth is not None:
            nodes = [node for node in nodes if node.depth >= min_depth]

        nodes = sorted(nodes, key=lambda node: node.scalar_score, reverse=True)

        return nodes[:top_k]

    def get_frontier(
        self,
        depth: int,
        top_k: int = 3,
    ) -> List[OptimizationNode]:
        nodes = [node for node in self.nodes.values() if node.depth == depth]
        nodes = sorted(nodes, key=lambda node: node.scalar_score, reverse=True)
        return nodes[:top_k]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_id": self.root_id,
            "nodes": {
                node_id: node.to_dict()
                for node_id, node in self.nodes.items()
            },
        }

    @staticmethod
    def _new_node_id() -> str:
        return str(uuid.uuid4())
