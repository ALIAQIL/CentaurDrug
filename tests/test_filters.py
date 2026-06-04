from src.tools.filters import (
    brenk_filter,
    is_valid_smiles,
    lipinski_filter,
    pains_filter,
    qed_score,
    veber_filter,
)


def test_valid_smiles():
    assert is_valid_smiles("CCO") is True


def test_invalid_smiles():
    assert is_valid_smiles("INVALID") is False


def test_lipinski_output():
    result = lipinski_filter("CCO")
    assert result["valid"] is True
    assert "passed" in result


def test_qed_score():
    score = qed_score("CCO")
    assert score is not None
    assert 0 <= score <= 1


def test_pains_filter():
    result = pains_filter("CCO")
    assert result["valid"] is True
    assert "passed" in result


def test_veber_filter():
    result = veber_filter("CCO")
    assert result["valid"] is True
    assert result["passed"] is True
    assert result["rotatable_bonds"] == 0
    assert result["tpsa"] >= 0


def test_brenk_filter():
    result = brenk_filter("CCO")
    assert result["valid"] is True
    assert "passed" in result
