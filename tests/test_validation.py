from src.models.validation import validate_smiles


def test_validate_valid_smiles():
    result = validate_smiles("CCO")

    assert result.is_valid
    assert result.canonical_smiles is not None


def test_validate_empty_smiles():
    result = validate_smiles("")

    assert not result.is_valid
    assert result.rejection_reason == "empty_smiles"


def test_validate_none_smiles():
    result = validate_smiles(None)

    assert not result.is_valid
    assert result.rejection_reason == "missing_smiles"


def test_validate_invalid_smiles():
    result = validate_smiles("not_a_smiles")

    assert not result.is_valid
    assert result.rejection_reason == "invalid_smiles"
