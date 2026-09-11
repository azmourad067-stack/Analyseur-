"""
HORSEPRONO NEURAL V1-B
Expérience SANS COTES

Ce script réutilise exactement le moteur Neural V1,
mais retire toutes les variables liées au marché.
"""

from pathlib import Path

import neural.train as base_train


# ============================================================
# VARIABLES INTERDITES POUR V1-B
# ============================================================

ODDS_FEATURES = {
    "odds",
    "odds_inv",
    "log_odds",
}


# ============================================================
# FEATURES V1-B
# ============================================================

base_train.NUMERIC_FEATURES = [
    feature
    for feature in base_train.NUMERIC_FEATURES
    if feature not in ODDS_FEATURES
]


# ============================================================
# ARTEFACTS SEPARES
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "neural"
    / "v1b_no_odds"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

base_train.MODEL_PATH = (
    MODEL_DIR
    / "horseprono_neural_v1b_no_odds.pt"
)

base_train.SCALER_PATH = (
    MODEL_DIR
    / "horseprono_neural_v1b_no_odds_scaler.joblib"
)

base_train.FEATURES_PATH = (
    MODEL_DIR
    / "horseprono_neural_v1b_no_odds_features.json"
)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("HORSEPRONO NEURAL V1-B — SANS COTES")
    print("=" * 70)

    print("\nVariables supprimées :")

    for feature in sorted(
        ODDS_FEATURES
    ):
        print(
            f"  - {feature}"
        )

    print(
        "\nNombre de variables numériques restantes : "
        f"{len(base_train.NUMERIC_FEATURES)}"
    )

    print(
        "\nFeatures numériques :"
    )

    for feature in (
        base_train.NUMERIC_FEATURES
    ):
        print(
            f"  - {feature}"
        )

    print()

    base_train.train()
