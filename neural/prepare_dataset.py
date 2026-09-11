from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from supabase import Client, create_client

from neural.config import (
    RANDOM_SEED,
    TARGET_COLUMN,
    TRAIN_RATIO,
    VALID_RATIO,
    TEST_RATIO,
)


# ============================================================
# HORSEPRONO NEURAL V1
# Préparation du dataset
# ============================================================

PAGE_SIZE = 1000


# ------------------------------------------------------------
# Colonnes Supabase
# ------------------------------------------------------------

PARTICIPANT_COLUMNS = [
    "id",
    "race_id",
    "horse_name",
    "horse_number",
    "jockey_name",
    "trainer_name",
    "odds",
    "weight_kg",
    "draw",
    "age",
    "sex",
    "recent_form",
    "finish_position",
    "is_non_runner",
]

RACE_COLUMNS = [
    "id",
    "race_date",
    "meeting_number",
    "race_number",
    "hippodrome",
    "discipline",
    "distance_m",
    "terrain",
    "field_size",
]


# ------------------------------------------------------------
# Variables prévues pour Neural V1
# ------------------------------------------------------------

NUMERIC_FEATURES = [
    "horse_number",
    "odds",
    "odds_inv",
    "log_odds",
    "weight_kg",
    "draw",
    "age",
    "meeting_number",
    "race_number",
    "distance_m",
    "field_size",

    # musique
    "form_runs",
    "form_mean_position",
    "form_best_position",
    "form_last_position",
    "form_win_rate",
    "form_top3_rate",
    "form_bad_rate",
]

CATEGORICAL_FEATURES = [
    "sex",
    "hippodrome",
    "discipline",
    "terrain",
    "jockey_name",
    "trainer_name",
]

METADATA_COLUMNS = [
    "participant_id",
    "race_id",
    "race_date",
    "horse_name",
    "finish_position",
]


# ============================================================
# SUPABASE
# ============================================================

def _get_secret(name: str) -> str | None:
    """
    Cherche d'abord dans les variables d'environnement,
    puis éventuellement dans st.secrets.
    """

    value = os.getenv(name)

    if value:
        return value

    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])

    except Exception:
        pass

    return None


def get_supabase_client() -> Client:
    url = _get_secret("SUPABASE_URL")

    key = (
        _get_secret("SUPABASE_KEY")
        or _get_secret("SUPABASE_ANON_KEY")
    )

    if not url:
        raise RuntimeError(
            "SUPABASE_URL introuvable."
        )

    if not key:
        raise RuntimeError(
            "SUPABASE_KEY ou SUPABASE_ANON_KEY introuvable."
        )

    return create_client(url, key)


def fetch_table_paginated(
    client: Client,
    table_name: str,
    columns: List[str],
    page_size: int = PAGE_SIZE,
) -> pd.DataFrame:
    """
    Charge une table Supabase par blocs pour ne pas être limité
    par la pagination PostgREST.
    """

    all_rows = []

    start = 0

    print(f"\nChargement Supabase : {table_name}")

    while True:

        end = start + page_size - 1

        response = (
            client
            .table(table_name)
            .select(",".join(columns))
            .order("id")
            .range(start, end)
            .execute()
        )

        batch = response.data or []

        if not batch:
            break

        all_rows.extend(batch)

        print(
            f"  {len(all_rows):,} lignes chargées..."
        )

        if len(batch) < page_size:
            break

        start += page_size

    df = pd.DataFrame(all_rows)

    print(
        f"{table_name}: {len(df):,} lignes au total."
    )

    return df


# ============================================================
# MUSIQUE / RECENT FORM
# ============================================================

def parse_recent_form(value) -> Dict[str, float]:
    """
    Convertit la musique du cheval en variables numériques.

    Exemples possibles :
        1a2a4a
        3p5p1p
        Da2a1a
        0p7p3p

    Les D/A/T sont considérés comme mauvaises performances.
    """

    empty = {
        "form_runs": 0.0,
        "form_mean_position": np.nan,
        "form_best_position": np.nan,
        "form_last_position": np.nan,
        "form_win_rate": 0.0,
        "form_top3_rate": 0.0,
        "form_bad_rate": 0.0,
    }

    if value is None:
        return empty

    form = str(value).strip()

    if not form:
        return empty

    # Supprime les années éventuelles :
    # (25), (2025), etc.
    form = re.sub(
        r"\([^)]*\)",
        "",
        form,
    )

    form = form.upper()

    # Cherche essentiellement :
    # 1A, 2P, 5A, DA, etc.
    raw_tokens = re.findall(
        r"(?:\d{1,2}|D|A|T)(?=[A-Z])",
        form,
    )

    # Fallback pour certaines musiques sans lettre
    if not raw_tokens:
        raw_tokens = re.findall(
            r"\d{1,2}|D|T",
            form,
        )

    if not raw_tokens:
        return empty

    positions = []

    bad_count = 0

    for token in raw_tokens[:10]:

        token = token.upper()

        if token.isdigit():

            position = int(token)

            # Dans certaines musiques :
            # 0 = non placé / >9
            if position == 0:
                position = 10

            positions.append(
                float(position)
            )

            if position >= 8:
                bad_count += 1

        else:

            # D = disqualifié
            # A = arrêté
            # T = tombé
            positions.append(15.0)
            bad_count += 1

    if not positions:
        return empty

    arr = np.asarray(
        positions,
        dtype=float,
    )

    return {
        "form_runs": float(len(arr)),
        "form_mean_position": float(np.mean(arr)),
        "form_best_position": float(np.min(arr)),
        "form_last_position": float(arr[0]),
        "form_win_rate": float(np.mean(arr == 1)),
        "form_top3_rate": float(np.mean(arr <= 3)),
        "form_bad_rate": float(
            bad_count / len(arr)
        ),
    }


def add_form_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print(
        "\nCréation des variables de forme récente..."
    )

    form_features = (
        df["recent_form"]
        .apply(parse_recent_form)
        .apply(pd.Series)
    )

    return pd.concat(
        [
            df.reset_index(drop=True),
            form_features.reset_index(drop=True),
        ],
        axis=1,
    )


# ============================================================
# NETTOYAGE
# ============================================================

def clean_and_merge(
    participants: pd.DataFrame,
    races: pd.DataFrame,
) -> pd.DataFrame:

    print("\nNettoyage du dataset...")

    participants = participants.copy()
    races = races.copy()

    # --------------------------------------------------------
    # Déduplication
    # --------------------------------------------------------

    participants = (
        participants
        .drop_duplicates(subset=["id"])
    )

    races = (
        races
        .drop_duplicates(subset=["id"])
    )

    # --------------------------------------------------------
    # Non-partants
    # --------------------------------------------------------

    participants["is_non_runner"] = (
        participants["is_non_runner"]
        .fillna(False)
        .astype(bool)
    )

    participants = participants[
        ~participants["is_non_runner"]
    ]

    # --------------------------------------------------------
    # Résultat obligatoire pour l'entraînement
    # --------------------------------------------------------

    participants["finish_position"] = (
        pd.to_numeric(
            participants["finish_position"],
            errors="coerce",
        )
    )

    participants = participants[
        participants["finish_position"].notna()
    ]

    # --------------------------------------------------------
    # Jointure courses / participants
    # --------------------------------------------------------

    races["race_date"] = pd.to_datetime(
        races["race_date"],
        errors="coerce",
    )

    participants = participants.rename(
        columns={"id": "participant_id"}
    )

    races = races.rename(
        columns={"id": "race_id"}
    )

    df = participants.merge(
        races,
        on="race_id",
        how="inner",
        validate="many_to_one",
    )

    df = df[
        df["race_date"].notna()
    ].copy()

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    df["target_win"] = (
        df["finish_position"] == 1
    ).astype(int)

    df["target_top3"] = (
        df["finish_position"]
        .between(1, 3)
    ).astype(int)

    df["target_top5"] = (
        df["finish_position"]
        .between(1, 5)
    ).astype(int)

    # --------------------------------------------------------
    # Numériques
    # --------------------------------------------------------

    numeric_columns = [
        "horse_number",
        "odds",
        "weight_kg",
        "draw",
        "age",
        "meeting_number",
        "race_number",
        "distance_m",
        "field_size",
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Cotes
    # --------------------------------------------------------

    df.loc[
        df["odds"] <= 0,
        "odds"
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

    # --------------------------------------------------------
    # Catégorielles
    # --------------------------------------------------------

    for col in CATEGORICAL_FEATURES:

        df[col] = (
            df[col]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
        )

        df.loc[
            df[col] == "",
            col
        ] = "UNKNOWN"

    # --------------------------------------------------------
    # Musique
    # --------------------------------------------------------

    df = add_form_features(df)

    # --------------------------------------------------------
    # Tri temporel
    # --------------------------------------------------------

    df = (
        df
        .sort_values(
            [
                "race_date",
                "race_id",
                "participant_id",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"Dataset exploitable : {len(df):,} chevaux"
    )

    return df


# ============================================================
# SPLIT TEMPOREL
# ============================================================

def temporal_split(
    df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    if not np.isclose(
        TRAIN_RATIO
        + VALID_RATIO
        + TEST_RATIO,
        1.0,
    ):
        raise ValueError(
            "TRAIN_RATIO + VALID_RATIO + "
            "TEST_RATIO doit être égal à 1."
        )

    dates = np.array(
        sorted(
            df["race_date"]
            .dt.normalize()
            .unique()
        )
    )

    n_dates = len(dates)

    if n_dates < 10:
        raise ValueError(
            "Pas assez de dates pour réaliser "
            "un split temporel fiable."
        )

    train_end = int(
        n_dates * TRAIN_RATIO
    )

    valid_end = train_end + int(
        n_dates * VALID_RATIO
    )

    train_dates = dates[:train_end]

    valid_dates = dates[
        train_end:valid_end
    ]

    test_dates = dates[
        valid_end:
    ]

    train_df = df[
        df["race_date"]
        .dt.normalize()
        .isin(train_dates)
    ].copy()

    valid_df = df[
        df["race_date"]
        .dt.normalize()
        .isin(valid_dates)
    ].copy()

    test_df = df[
        df["race_date"]
        .dt.normalize()
        .isin(test_dates)
    ].copy()

    return (
        train_df,
        valid_df,
        test_df,
    )


# ============================================================
# DATASET FINAL
# ============================================================

def select_model_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    columns = (
        METADATA_COLUMNS
        + NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [
            "target_win",
            "target_top3",
            "target_top5",
        ]
    )

    columns = [
        col
        for col in columns
        if col in df.columns
    ]

    return df[columns].copy()


def prepare_dataset():
    """
    Pipeline complet :
        Supabase
            ↓
        nettoyage
            ↓
        feature engineering
            ↓
        split temporel
            ↓
        train / validation / test
    """

    np.random.seed(
        RANDOM_SEED
    )

    client = get_supabase_client()

    participants = fetch_table_paginated(
        client,
        "participants",
        PARTICIPANT_COLUMNS,
    )

    races = fetch_table_paginated(
        client,
        "races",
        RACE_COLUMNS,
    )

    df = clean_and_merge(
        participants,
        races,
    )

    df = select_model_columns(df)

    train_df, valid_df, test_df = (
        temporal_split(df)
    )

    return (
        train_df,
        valid_df,
        test_df,
    )


# ============================================================
# DIAGNOSTIC
# ============================================================

def print_split_summary(
    name: str,
    df: pd.DataFrame,
):

    print("\n" + "=" * 60)

    print(name)

    print("=" * 60)

    print(
        f"Chevaux : {len(df):,}"
    )

    print(
        f"Courses : {df['race_id'].nunique():,}"
    )

    print(
        "Dates :",
        df["race_date"].min().date(),
        "→",
        df["race_date"].max().date(),
    )

    print(
        f"Top3 : "
        f"{df[TARGET_COLUMN].mean():.2%}"
    )

    print(
        f"Cotes disponibles : "
        f"{df['odds'].notna().mean():.2%}"
    )


# ============================================================
# EXECUTION DIRECTE
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("HORSEPRONO NEURAL V1 - PREPARATION DATASET")
    print("=" * 70)

    train_df, valid_df, test_df = (
        prepare_dataset()
    )

    print_split_summary(
        "TRAIN",
        train_df,
    )

    print_split_summary(
        "VALIDATION",
        valid_df,
    )

    print_split_summary(
        "TEST AVEUGLE",
        test_df,
    )

    print("\n")
    print("Dataset Neural V1 prêt.")
