from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from neural.model import HorsePronoNeuralV1
from neural.prepare_dataset import parse_recent_form


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models" / "neural"

MODEL_PATH = MODEL_DIR / "horseprono_neural_v1.pt"
SCALER_PATH = MODEL_DIR / "horseprono_neural_v1_scaler.joblib"
FEATURES_PATH = MODEL_DIR / "horseprono_neural_v1_features.json"


# ============================================================
# HASH DU MODELE
# ============================================================

def sha256_file(path: Path) -> str:

    sha = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            sha.update(chunk)

    return sha.hexdigest()


# ============================================================
# CHARGEMENT
# ============================================================

def load_neural_v1(
    device: torch.device | None = None,
):

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        FEATURES_PATH,
    ]

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Fichier Neural manquant : {path}"
            )

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    preprocessing = joblib.load(
        SCALER_PATH
    )

    scaler = preprocessing["scaler"]
    medians = preprocessing["medians"]

    with open(
        FEATURES_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        feature_config = json.load(f)

    numeric_features = (
        feature_config[
            "numeric_features"
        ]
    )

    categorical_features = (
        feature_config[
            "categorical_features"
        ]
    )

    vocabs = (
        feature_config[
            "categorical_vocabs"
        ]
    )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    cardinalities = (
        checkpoint[
            "categorical_cardinalities"
        ]
    )

    hidden_layers = checkpoint.get(
        "hidden_layers",
        [128, 64, 32],
    )

    model = HorsePronoNeuralV1(
        num_numeric_features=len(
            numeric_features
        ),
        categorical_cardinalities=(
            cardinalities
        ),
        hidden_1=hidden_layers[0],
        hidden_2=hidden_layers[1],
        hidden_3=hidden_layers[2],
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model.to(device)
    model.eval()

    return {
        "model": model,
        "device": device,
        "checkpoint": checkpoint,
        "scaler": scaler,
        "medians": medians,
        "vocabs": vocabs,
        "numeric_features":
            numeric_features,
        "categorical_features":
            categorical_features,
    }


# ============================================================
# FEATURE ENGINEERING INFERENCE
# ============================================================

def prepare_inference_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    # ========================================================
    # ALIASES PMU LIVE -> FORMAT NEURAL TRAINING
    # ========================================================

    aliases = {
        "jockey": "jockey_name",
        "trainer": "trainer_name",
        "weight": "weight_kg",
        "reunion": "meeting_number",
        "course_number": "race_number",
        "distance": "distance_m",
    }

    for source, target in aliases.items():

        if (
            source in df.columns
            and target not in df.columns
        ):
            df[target] = df[source]

    # ========================================================
    # COTE
    # ========================================================

    if "odds" not in df.columns:
        df["odds"] = np.nan

    df["odds"] = pd.to_numeric(
        df["odds"],
        errors="coerce",
    )

    df.loc[
        df["odds"] <= 0,
        "odds",
    ] = np.nan

    df["odds_inv"] = np.where(
        df["odds"].notna(),
        1.0 / df["odds"],
        np.nan,
    )

    df["log_odds"] = np.where(
        df["odds"].notna(),
        np.log1p(df["odds"]),
        np.nan,
    )

    # ========================================================
    # MUSIQUE
    # ========================================================

    if "recent_form" not in df.columns:
        df["recent_form"] = None

    form_df = (
        df["recent_form"]
        .apply(parse_recent_form)
        .apply(pd.Series)
    )

    for column in form_df.columns:
        df[column] = form_df[column]

    return df


# ============================================================
# ENCODAGE
# ============================================================

def build_model_inputs(
    df: pd.DataFrame,
    bundle: dict,
):

    df = prepare_inference_dataframe(
        df
    )

    numeric_features = (
        bundle["numeric_features"]
    )

    categorical_features = (
        bundle["categorical_features"]
    )

    scaler = bundle["scaler"]
    medians = bundle["medians"]
    vocabs = bundle["vocabs"]

    # --------------------------------------------------------
    # NUMERIQUES
    # --------------------------------------------------------

    numeric = pd.DataFrame(
        index=df.index
    )

    for column in numeric_features:

        if column in df.columns:

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        else:

            values = pd.Series(
                np.nan,
                index=df.index,
            )

        values = values.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        numeric[column] = (
            values.fillna(
                medians[column]
            )
        )

    numeric_x = scaler.transform(
        numeric[numeric_features]
    )

    numeric_x = np.asarray(
        numeric_x,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # CATEGORIELS
    # --------------------------------------------------------

    categorical_arrays = []

    for column in categorical_features:

        if column in df.columns:

            values = (
                df[column]
                .fillna("UNKNOWN")
                .astype(str)
                .str.strip()
            )

        else:

            values = pd.Series(
                "UNKNOWN",
                index=df.index,
            )

        vocab = vocabs[column]

        encoded = (
            values
            .map(vocab)
            .fillna(0)
            .astype(np.int64)
            .to_numpy()
        )

        categorical_arrays.append(
            encoded
        )

    categorical_x = np.column_stack(
        categorical_arrays
    ).astype(np.int64)

    return numeric_x, categorical_x


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def predict_dataframe(
    df: pd.DataFrame,
    bundle: dict | None = None,
) -> pd.DataFrame:

    if bundle is None:
        bundle = load_neural_v1()

    model = bundle["model"]
    device = bundle["device"]

    numeric_x, categorical_x = (
        build_model_inputs(
            df,
            bundle,
        )
    )

    numeric_tensor = torch.tensor(
        numeric_x,
        dtype=torch.float32,
        device=device,
    )

    categorical_tensor = torch.tensor(
        categorical_x,
        dtype=torch.long,
        device=device,
    )

    logits = model(
        numeric_tensor,
        categorical_tensor,
    )

    probabilities = torch.sigmoid(
        logits
    ).cpu().numpy()

    result = df.copy()

    result[
        "neural_top3_probability"
    ] = probabilities

    # --------------------------------------------------------
    # CLASSEMENT PAR COURSE
    # --------------------------------------------------------

    if "race_id" in result.columns:

        result[
            "neural_rank"
        ] = (
            result
            .groupby("race_id")[
                "neural_top3_probability"
            ]
            .rank(
                method="first",
                ascending=False,
            )
            .astype(int)
        )

    return result


# ============================================================
# SELF TEST
# ============================================================

def self_test():

    print()
    print("=" * 70)
    print("HORSEPRONO NEURAL V1 - SELF TEST")
    print("=" * 70)

    bundle = load_neural_v1()

    checkpoint = bundle[
        "checkpoint"
    ]

    print(
        f"\nDevice : "
        f"{bundle['device']}"
    )

    print(
        f"Modèle : "
        f"{checkpoint.get('model_name')}"
    )

    print(
        f"Best epoch : "
        f"{checkpoint.get('best_epoch')}"
    )

    print(
        f"Train : "
        f"{checkpoint.get('train_start')} "
        f"→ {checkpoint.get('train_end')}"
    )

    print(
        f"Validation : "
        f"{checkpoint.get('validation_start')} "
        f"→ {checkpoint.get('validation_end')}"
    )

    print(
        f"Test : "
        f"{checkpoint.get('test_start')} "
        f"→ {checkpoint.get('test_end')}"
    )

    print(
        "\nFeatures numériques : "
        f"{len(bundle['numeric_features'])}"
    )

    print(
        "Features catégorielles : "
        f"{len(bundle['categorical_features'])}"
    )

    print(
        "\nSHA256 :"
    )

    print(
        sha256_file(
            MODEL_PATH
        )
    )

    print()
    print(
        "✅ Neural V1 chargé correctement."
    )


if __name__ == "__main__":
    self_test()
