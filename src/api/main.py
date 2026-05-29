from fastapi import FastAPI
from pydantic import BaseModel

from src.tools.evaluator import evaluate_rules

app = FastAPI(title="CentaurDrug API")


class MoleculeRequest(BaseModel):
    smiles: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/evaluate")
def evaluate(request: MoleculeRequest):
    return evaluate_rules(request.smiles)