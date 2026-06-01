from __future__ import annotations

from typing import Any, Dict, List


TRANSFORMATION_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "aromatic_H_to_F": {
        "description": "Replace an aromatic hydrogen with fluorine.",
        "expected_effects": {
            "solubility": "usually neutral or slightly worse",
            "lipophilicity": "may increase slightly",
            "metabolic_stability": "may improve",
            "binding": "may improve or worsen depending on pocket geometry",
        },
        "use_when": [
            "need subtle electronic tuning",
            "need to block metabolic soft spots",
            "need small steric change",
        ],
        "avoid_when": [
            "high_lipophilicity",
            "hERG risk",
            "already too hydrophobic",
        ],
        "risk": [
            "may increase lipophilicity",
            "may worsen hERG risk",
        ],
    },
    "aromatic_H_to_Cl": {
        "description": "Replace an aromatic hydrogen with chlorine.",
        "expected_effects": {
            "solubility": "often decreases",
            "lipophilicity": "usually increases",
            "potency": "may improve hydrophobic pocket binding",
        },
        "use_when": [
            "need stronger hydrophobic interaction",
            "need potency exploration",
        ],
        "avoid_when": [
            "poor_solubility",
            "high_lipophilicity",
            "hERG risk",
        ],
        "risk": [
            "can worsen solubility",
            "can worsen hERG or CYP liability",
        ],
    },
    "aromatic_H_to_OH": {
        "description": "Replace an aromatic hydrogen with hydroxyl.",
        "expected_effects": {
            "solubility": "usually increases",
            "lipophilicity": "usually decreases",
            "permeability": "may decrease",
            "metabolism": "may introduce phase-II metabolism",
        },
        "use_when": [
            "poor_solubility",
            "high_lipophilicity",
        ],
        "avoid_when": [
            "already too polar",
            "too many hydrogen bond donors",
            "low permeability",
        ],
        "risk": [
            "may reduce membrane permeability",
            "may disrupt hydrophobic binding",
        ],
    },
    "aromatic_H_to_OMe": {
        "description": "Replace an aromatic hydrogen with methoxy.",
        "expected_effects": {
            "solubility": "may improve slightly",
            "lipophilicity": "may increase or remain moderate",
            "binding": "can add H-bond acceptor and steric bulk",
        },
        "use_when": [
            "need mild polarity increase",
            "need H-bond acceptor without adding donor",
        ],
        "avoid_when": [
            "high_lipophilicity",
            "steric sensitivity",
        ],
        "risk": [
            "may increase molecular weight and lipophilicity",
            "may create metabolic liability",
        ],
    },
    "aromatic_H_to_Me": {
        "description": "Replace an aromatic hydrogen with methyl.",
        "expected_effects": {
            "solubility": "often decreases",
            "lipophilicity": "usually increases",
            "binding": "may improve hydrophobic interactions",
        },
        "use_when": [
            "need hydrophobic pocket filling",
            "need potency exploration",
        ],
        "avoid_when": [
            "poor_solubility",
            "high_lipophilicity",
            "hERG risk",
        ],
        "risk": [
            "may worsen ADMET",
            "may increase hERG risk",
        ],
    },
    "ester_to_acid": {
        "description": "Convert an ester-like motif into a carboxylic acid-like motif.",
        "expected_effects": {
            "solubility": "usually increases",
            "lipophilicity": "usually decreases",
            "permeability": "may decrease",
            "metabolic_stability": "may avoid ester hydrolysis liability",
        },
        "use_when": [
            "poor_solubility",
            "ester liability",
            "need polarity increase",
        ],
        "avoid_when": [
            "low permeability",
            "need neutral molecule",
        ],
        "risk": [
            "may reduce permeability",
            "may change binding mode strongly",
        ],
    },
}


def get_transformation_kb() -> Dict[str, Dict[str, Any]]:
    return TRANSFORMATION_KNOWLEDGE_BASE


def get_available_transformations() -> List[str]:
    return list(TRANSFORMATION_KNOWLEDGE_BASE.keys())


def summarize_transformations_for_prompt() -> List[Dict[str, Any]]:
    summary = []

    for name, info in TRANSFORMATION_KNOWLEDGE_BASE.items():
        summary.append(
            {
                "name": name,
                "description": info["description"],
                "expected_effects": info["expected_effects"],
                "use_when": info["use_when"],
                "avoid_when": info["avoid_when"],
                "risk": info["risk"],
            }
        )

    return summary