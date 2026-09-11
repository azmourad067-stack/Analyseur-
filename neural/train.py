from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from neural.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    FEATURES_PATH,
    HIDDEN_LAYERS,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_PATH,
    RANDOM_SEED,
    SCALER_PATH,
    TARGET_COLUMN,
    WEIGHT_DECAY,
)
from neural.model import (
    HorsePronoNeuralV1,
    count_parameters,
    print_model_summary,
)
from neural.prepare_dataset import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    prepare_dataset,
)


# ============================================================
# HORSEPRONO NEURAL V1
# TRAINING
# ============================================================


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET PYTORCH
# ============================================================

class HorseDataset(Dataset):

    def __init__(
        self,
        numeric_x: np.ndarray,
        categorical_x: np.ndarray,
        targets: np.ndarray,
    ):

        self.numeric_x = torch.tensor(
            numeric_x,
            dtype=torch.float32,
        )

        self.categorical_x = torch.tensor(
            categorical_x,
            dtype=torch.long,
        )

        self.targets = torch.tensor(
            targets,
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):

        return (
            self.numeric_x[index],
            self.categorical_x[index],
            self.targets[index],
        )


# ============================================================
# PREPROCESSING NUMERIQUE
# ============================================================

def fit_numeric_preprocessor(
    train_df: pd.DataFrame,
) -> Tuple[StandardScaler, Dict[str, float]]:

    medians = {}

    train_numeric = train_df[
        NUMERIC_FEATURES
    ].copy()

    for column in NUMERIC_FEATURES:

        values = pd.to_numeric(
            train_numeric[column],
            errors="coerce",
        )

        median = values.median()

        if pd.isna(median):
            median = 0.0

        medians[column] = float(median)

        train_numeric[column] = (
            values.fillna(median)
        )

    scaler = StandardScaler()

    scaler.fit(
        train_numeric[NUMERIC_FEATURES]
    )

    return scaler, medians


def transform_numeric(
    df: pd.DataFrame,
    scaler: StandardScaler,
    medians: Dict[str, float],
) -> np.ndarray:

    numeric = df[
        NUMERIC_FEATURES
    ].copy()

    for column in NUMERIC_FEATURES:

        numeric[column] = pd.to_numeric(
            numeric[column],
            errors="coerce",
        )

        numeric[column] = (
            numeric[column]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(
                medians[column]
            )
        )

    transformed = scaler.transform(
        numeric[NUMERIC_FEATURES]
    )

    return transformed.astype(
        np.float32
    )


# ============================================================
# VOCABULAIRES CATEGORIELS
# ============================================================

def build_category_vocabs(
    train_df: pd.DataFrame,
) -> Dict[str, Dict[str, int]]:

    vocabs = {}

    for column in CATEGORICAL_FEATURES:

        values = (
            train_df[column]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
        )

        unique_values = sorted(
            value
            for value in values.unique()
            if value
            and value != "UNKNOWN"
        )

        # 0 = catégorie inconnue
        vocab = {
            value: index + 1
            for index, value
            in enumerate(unique_values)
        }

        vocabs[column] = vocab

    return vocabs


def transform_categories(
    df: pd.DataFrame,
    vocabs: Dict[str, Dict[str, int]],
) -> np.ndarray:

    encoded_columns = []

    for column in CATEGORICAL_FEATURES:

        values = (
            df[column]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
        )

        vocab = vocabs[column]

        encoded = (
            values
            .map(vocab)
            .fillna(0)
            .astype(np.int64)
            .to_numpy()
        )

        encoded_columns.append(
            encoded
        )

    return np.column_stack(
        encoded_columns
    ).astype(np.int64)


# ============================================================
# PREPARATION
# ============================================================

def build_arrays(
    df: pd.DataFrame,
    scaler: StandardScaler,
    medians: Dict[str, float],
    vocabs: Dict[str, Dict[str, int]],
):

    numeric_x = transform_numeric(
        df,
        scaler,
        medians,
    )

    categorical_x = transform_categories(
        df,
        vocabs,
    )

    targets = (
        df[TARGET_COLUMN]
        .astype(np.float32)
        .to_numpy()
    )

    return (
        numeric_x,
        categorical_x,
        targets,
    )


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):

    model.eval()

    all_targets = []
    all_probs = []

    for (
        numeric_x,
        categorical_x,
        targets,
    ) in loader:

        numeric_x = numeric_x.to(
            device
        )

        categorical_x = categorical_x.to(
            device
        )

        logits = model(
            numeric_x,
            categorical_x,
        )

        probs = torch.sigmoid(
            logits
        )

        all_targets.append(
            targets.numpy()
        )

        all_probs.append(
            probs.cpu().numpy()
        )

    y_true = np.concatenate(
        all_targets
    )

    y_prob = np.concatenate(
        all_probs
    )

    return y_true, y_prob


# ============================================================
# METRIQUES
# ============================================================

def binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
):

    y_prob = np.clip(
        y_prob,
        1e-7,
        1 - 1e-7,
    )

    return {
        "auc": float(
            roc_auc_score(
                y_true,
                y_prob,
            )
        ),

        "logloss": float(
            log_loss(
                y_true,
                y_prob,
                labels=[0, 1],
            )
        ),

        "brier": float(
            brier_score_loss(
                y_true,
                y_prob,
            )
        ),
    }


def race_metrics(
    df: pd.DataFrame,
    probabilities: np.ndarray,
):

    temp = df[
        [
            "race_id",
            "finish_position",
        ]
    ].copy()

    temp["probability"] = (
        probabilities
    )

    race_count = 0

    top1_winner = 0
    top1_placed = 0
    top3_contains_winner = 0

    top3_overlap_total = 0.0

    for _, race in temp.groupby(
        "race_id"
    ):

        race = race.sort_values(
            "probability",
            ascending=False,
        )

        if race.empty:
            continue

        race_count += 1

        # -----------------------------------------------
        # Cheval n°1 du modèle
        # -----------------------------------------------

        model_top1 = race.iloc[0]

        if (
            model_top1[
                "finish_position"
            ]
            == 1
        ):
            top1_winner += 1

        if (
            model_top1[
                "finish_position"
            ]
            <= 3
        ):
            top1_placed += 1

        # -----------------------------------------------
        # Top 3 modèle
        # -----------------------------------------------

        model_top3 = race.head(3)

        if (
            model_top3[
                "finish_position"
            ]
            .eq(1)
            .any()
        ):
            top3_contains_winner += 1

        actual_top3_indices = set(
            race[
                race[
                    "finish_position"
                ].between(1, 3)
            ].index
        )

        predicted_top3_indices = set(
            model_top3.index
        )

        overlap = len(
            actual_top3_indices
            & predicted_top3_indices
        )

        top3_overlap_total += (
            overlap / 3.0
        )

    if race_count == 0:

        return {
            "race_count": 0,
            "top1_winner_rate": 0.0,
            "top1_place_rate": 0.0,
            "top3_contains_winner_rate": 0.0,
            "top3_overlap": 0.0,
        }

    return {
        "race_count": race_count,

        "top1_winner_rate":
            top1_winner / race_count,

        "top1_place_rate":
            top1_placed / race_count,

        "top3_contains_winner_rate":
            top3_contains_winner
            / race_count,

        "top3_overlap":
            top3_overlap_total
            / race_count,
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    source_df,
    device,
):

    y_true, y_prob = predict_loader(
        model,
        loader,
        device,
    )

    metrics = binary_metrics(
        y_true,
        y_prob,
    )

    metrics.update(
        race_metrics(
            source_df,
            y_prob,
        )
    )

    return metrics


# ============================================================
# AFFICHAGE
# ============================================================

def print_metrics(
    name: str,
    metrics: Dict,
):

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print(
        f"AUC                 : "
        f"{metrics['auc']:.4f}"
    )

    print(
        f"LogLoss             : "
        f"{metrics['logloss']:.4f}"
    )

    print(
        f"Brier               : "
        f"{metrics['brier']:.4f}"
    )

    print(
        f"Top1 gagnant        : "
        f"{metrics['top1_winner_rate']:.2%}"
    )

    print(
        f"Top1 placé          : "
        f"{metrics['top1_place_rate']:.2%}"
    )

    print(
        f"Top3 contient gagnant : "
        f"{metrics['top3_contains_winner_rate']:.2%}"
    )

    print(
        f"Recouvrement Top3   : "
        f"{metrics['top3_overlap']:.2%}"
    )


# ============================================================
# SAUVEGARDE PREPROCESSING
# ============================================================

def save_preprocessing(
    scaler,
    medians,
    vocabs,
):

    SCALER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "scaler": scaler,
            "medians": medians,
        },
        SCALER_PATH,
    )

    payload = {
        "numeric_features":
            NUMERIC_FEATURES,

        "categorical_features":
            CATEGORICAL_FEATURES,

        "target":
            TARGET_COLUMN,

        "categorical_vocabs":
            vocabs,
    }

    with open(
        FEATURES_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# TRAINING
# ============================================================

def train():

    print("\n")
    print("=" * 70)
    print("HORSEPRONO NEURAL V1 - TRAINING")
    print("=" * 70)

    set_seed()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice : {device}"
    )

    # ========================================================
    # DONNEES
    # ========================================================

    print(
        "\nChargement et préparation "
        "des données..."
    )

    train_df, valid_df, test_df = (
        prepare_dataset()
    )

    print(
        f"\nTRAIN : {len(train_df):,}"
    )

    print(
        f"VALID : {len(valid_df):,}"
    )

    print(
        f"TEST  : {len(test_df):,}"
    )

    # ========================================================
    # PREPROCESSING
    # ========================================================

    scaler, medians = (
        fit_numeric_preprocessor(
            train_df
        )
    )

    vocabs = build_category_vocabs(
        train_df
    )

    print(
        "\nCardinalités catégorielles :"
    )

    for column in CATEGORICAL_FEATURES:

        print(
            f"  {column:<20} "
            f"{len(vocabs[column]):,}"
        )

    save_preprocessing(
        scaler,
        medians,
        vocabs,
    )

    # ========================================================
    # ARRAYS
    # ========================================================

    train_arrays = build_arrays(
        train_df,
        scaler,
        medians,
        vocabs,
    )

    valid_arrays = build_arrays(
        valid_df,
        scaler,
        medians,
        vocabs,
    )

    # IMPORTANT :
    # le test est transformé ici,
    # mais jamais utilisé pour choisir le modèle.
    test_arrays = build_arrays(
        test_df,
        scaler,
        medians,
        vocabs,
    )

    train_dataset = HorseDataset(
        *train_arrays
    )

    valid_dataset = HorseDataset(
        *valid_arrays
    )

    test_dataset = HorseDataset(
        *test_arrays
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ========================================================
    # MODELE
    # ========================================================

    cardinalities = {
        column: len(vocabs[column])
        for column
        in CATEGORICAL_FEATURES
    }

    model = HorsePronoNeuralV1(
        num_numeric_features=len(
            NUMERIC_FEATURES
        ),
        categorical_cardinalities=(
            cardinalities
        ),
        hidden_1=HIDDEN_LAYERS[0],
        hidden_2=HIDDEN_LAYERS[1],
        hidden_3=HIDDEN_LAYERS[2],
    )

    model = model.to(device)

    print_model_summary(
        model
    )

    # ========================================================
    # BASELINE
    # ========================================================

    train_rate = float(
        train_df[
            TARGET_COLUMN
        ].mean()
    )

    baseline_probs = np.full(
        len(valid_df),
        train_rate,
    )

    baseline_logloss = log_loss(
        valid_df[
            TARGET_COLUMN
        ],
        baseline_probs,
        labels=[0, 1],
    )

    print(
        "\nBaseline naïve "
        f"P(Top3)={train_rate:.2%}"
    )

    print(
        "Baseline validation LogLoss : "
        f"{baseline_logloss:.4f}"
    )

    # ========================================================
    # OPTIMISATION
    # ========================================================

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4,
        )
    )

    # ========================================================
    # BOUCLE
    # ========================================================

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    best_metrics = None

    patience_counter = 0

    history = []

    print(
        "\nDébut entraînement...\n"
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0
        seen = 0

        for (
            numeric_x,
            categorical_x,
            targets,
        ) in train_loader:

            numeric_x = numeric_x.to(
                device
            )

            categorical_x = (
                categorical_x.to(
                    device
                )
            )

            targets = targets.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                numeric_x,
                categorical_x,
            )

            loss = criterion(
                logits,
                targets,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            batch_size = len(
                targets
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            seen += batch_size

        train_loss = (
            running_loss / seen
        )

        # -----------------------------------------------
        # VALIDATION
        # -----------------------------------------------

        valid_metrics = evaluate(
            model,
            valid_loader,
            valid_df,
            device,
        )

        val_loss = (
            valid_metrics[
                "logloss"
            ]
        )

        scheduler.step(
            val_loss
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss":
                    train_loss,
                "val_logloss":
                    val_loss,
                "val_auc":
                    valid_metrics[
                        "auc"
                    ],
                "val_brier":
                    valid_metrics[
                        "brier"
                    ],
                "lr":
                    current_lr,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.4f} | "
            f"val={val_loss:.4f} | "
            f"AUC={valid_metrics['auc']:.4f} | "
            f"Brier={valid_metrics['brier']:.4f} | "
            f"lr={current_lr:.2e}"
        )

        # -----------------------------------------------
        # EARLY STOPPING
        # -----------------------------------------------

        improvement = (
            val_loss
            < best_val_loss - 1e-5
        )

        if improvement:

            best_val_loss = val_loss
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            best_metrics = copy.deepcopy(
                valid_metrics
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if (
            patience_counter
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping déclenché."
            )

            break

    # ========================================================
    # RESTAURATION MEILLEUR MODELE
    # ========================================================

    if best_state is None:
        raise RuntimeError(
            "Aucun modèle valide entraîné."
        )

    model.load_state_dict(
        best_state
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"MEILLEUR EPOCH : {best_epoch}"
    )

    print(
        "=" * 70
    )

    print_metrics(
        "VALIDATION",
        best_metrics,
    )

    # ========================================================
    # TEST AVEUGLE
    # ========================================================

    print(
        "\nOuverture du TEST AVEUGLE..."
    )

    test_metrics = evaluate(
        model,
        test_loader,
        test_df,
        device,
    )

    print_metrics(
        "TEST AVEUGLE",
        test_metrics,
    )

    # ========================================================
    # SAUVEGARDE MODELE
    # ========================================================

    checkpoint = {
        "model_name":
            "horseprono_neural_v1",

        "state_dict":
            model.state_dict(),

        "num_numeric_features":
            len(NUMERIC_FEATURES),

        "categorical_cardinalities":
            cardinalities,

        "numeric_features":
            NUMERIC_FEATURES,

        "categorical_features":
            CATEGORICAL_FEATURES,

        "target":
            TARGET_COLUMN,

        "hidden_layers":
            HIDDEN_LAYERS,

        "best_epoch":
            best_epoch,

        "validation_metrics":
            best_metrics,

        "test_metrics":
            test_metrics,

        "train_start":
            str(
                train_df[
                    "race_date"
                ].min().date()
            ),

        "train_end":
            str(
                train_df[
                    "race_date"
                ].max().date()
            ),

        "validation_start":
            str(
                valid_df[
                    "race_date"
                ].min().date()
            ),

        "validation_end":
            str(
                valid_df[
                    "race_date"
                ].max().date()
            ),

        "test_start":
            str(
                test_df[
                    "race_date"
                ].min().date()
            ),

        "test_end":
            str(
                test_df[
                    "race_date"
                ].max().date()
            ),
    }

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    # ========================================================
    # HISTORIQUE
    # ========================================================

    history_path = (
        MODEL_PATH.parent
        / "horseprono_neural_v1_history.csv"
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "NEURAL V1 TERMINE"
    )

    print(
        "=" * 70
    )

    print(
        f"Paramètres : "
        f"{count_parameters(model):,}"
    )

    print(
        f"Modèle : {MODEL_PATH}"
    )

    print(
        f"Scaler : {SCALER_PATH}"
    )

    print(
        f"Features : {FEATURES_PATH}"
    )

    print(
        f"Historique : {history_path}"
    )


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    train()
