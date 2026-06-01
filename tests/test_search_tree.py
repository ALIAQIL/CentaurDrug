from src.agent.search_tree import OptimizationTree


def _score(value):
    return {
        "raw": {},
        "normalized": {},
        "scalar_score": value,
    }


def test_get_best_nodes_can_exclude_root():
    tree = OptimizationTree()
    root_id = tree.add_root(
        smiles="CCO",
        evaluation={"status": "ok"},
        score_vector=_score(10.0),
    )
    low_child_id = tree.add_child(
        parent_id=root_id,
        smiles="CCCO",
        transformation="aromatic_H_to_OH",
        evaluation={"status": "ok"},
        score_vector=_score(4.0),
    )
    high_child_id = tree.add_child(
        parent_id=root_id,
        smiles="CCCCO",
        transformation="aromatic_H_to_Me",
        evaluation={"status": "ok"},
        score_vector=_score(6.0),
    )

    best_candidates = tree.get_best_nodes(top_k=2, min_depth=1)

    assert [node.node_id for node in best_candidates] == [
        high_child_id,
        low_child_id,
    ]
