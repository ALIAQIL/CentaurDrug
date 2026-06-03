.PHONY: test verify-mlflow train-sol train-lipo train-ames train-herg train-cyp3a4 train-phase1 predict-sol

test:
	uv run pytest -q

verify-mlflow:
	uv run python -m src.mlops.verify_mlflow --config configs/training.yaml

train-sol:
	uv run python -m src.models.train_admet_xgboost --config configs/training.yaml --dataset Solubility_AqSolDB

train-lipo:
	uv run python -m src.models.train_admet_xgboost --config configs/training.yaml --dataset Lipophilicity_AstraZeneca

train-ames:
	uv run python -m src.models.train_admet_xgboost --config configs/training.yaml --dataset AMES

train-herg:
	uv run python -m src.models.train_admet_xgboost --config configs/training.yaml --dataset hERG

train-cyp3a4:
	uv run python -m src.models.train_admet_xgboost --config configs/training.yaml --dataset CYP3A4_Veith

train-phase1: train-sol train-lipo train-ames train-herg train-cyp3a4

predict-sol:
	uv run python -m src.models.predict --artifact-dir models/admet_xgboost/Solubility_AqSolDB --smiles "CCO"

evaluate:
	uv run python -m src.tools.evaluator --smiles "CCO"

agent-search:
	uv run python -m src.agent.graph --smiles "CC(=O)Oc1ccccc1C(=O)O" --max-depth 2 --beam-width 3 --max-candidates-per-node 10
