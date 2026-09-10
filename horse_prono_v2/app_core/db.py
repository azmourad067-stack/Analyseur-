from __future__ import annotations

import math
import os
from datetime import date
from typing import Any

import pandas as pd
import streamlit as st
from supabase import Client, create_client


# ============================================================
# SUPABASE CONNECTION
# ============================================================

@st.cache_resource
def get_supabase_client() -> Client | None:
    """
    Retourne le client Supabase.

    Streamlit :
        SUPABASE_URL
        SUPABASE_KEY

    GitHub Actions / CLI :
        SUPABASE_URL
        SUPABASE_SERVICE_KEY
    """

    try:
        url = st.secrets["SUPABASE_URL"]

        key = (
            st.secrets.get("SUPABASE_KEY")
            or os.getenv("SUPABASE_KEY")
            or os.getenv("SUPABASE_SERVICE_KEY")
        )

    except Exception:
        url = os.getenv("SUPABASE_URL")

        key = (
            os.getenv("SUPABASE_KEY")
            or os.getenv("SUPABASE_SERVICE_KEY")
        )

    if not url or not key:
        return None

    return create_client(
        str(url),
        str(key),
    )


def is_configured() -> bool:
    return get_supabase_client() is not None


def _client() -> Client:
    client = get_supabase_client()

    if client is None:
        raise RuntimeError("Supabase non configuré")

    return client


# ============================================================
# SUPABASE RESPONSE HELPERS
# ============================================================

def _rows(resp: Any) -> list[dict]:
    return list(
        getattr(resp, "data", None)
        or []
    )


def healthcheck() -> tuple[bool, str]:
    client = get_supabase_client()

    if client is None:
        return (
            False,
            "Supabase non configuré (secrets absents).",
        )

    try:
        client.table(
            "races"
        ).select(
            "id"
        ).limit(
            1
        ).execute()

        return (
            True,
            "Connexion Supabase opérationnelle.",
        )

    except Exception as exc:
        return (
            False,
            f"Supabase inaccessible : {exc}",
        )


# ============================================================
# JSON / PANDAS NORMALIZATION
# ============================================================

def _json_safe(value):
    """
    Convertit les types Pandas / NumPy vers des types JSON.

    NaN / NaT / +/-inf deviennent None.
    """

    if value is None:
        return None

    # Timestamp Pandas
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None

        return value.isoformat()

    # Valeurs manquantes :
    # NaN, pd.NA, NaT...
    try:
        missing = pd.isna(value)

        if isinstance(missing, bool) and missing:
            return None

    except (TypeError, ValueError):
        pass

    # Types NumPy -> types Python
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    # Nouvelle vérification après conversion
    if value is None:
        return None

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value


def _records(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Transforme un DataFrame en liste de dictionnaires
    compatibles JSON / Supabase.
    """

    rows = df.to_dict(
        orient="records"
    )

    clean_rows = []

    for row in rows:

        clean_row = {}

        for key, value in row.items():
            clean_row[key] = _json_safe(
                value
            )

        clean_rows.append(
            clean_row
        )

    return clean_rows


def _to_nullable_int(
    series: pd.Series,
) -> pd.Series:
    """
    Convertit une colonne vers un entier nullable Pandas.

    Exemples :
        1      -> 1
        1.0    -> 1
        "8"    -> 8
        "8.0"  -> 8
        NaN    -> <NA>

    Les valeurs non entières sont remplacées par <NA>.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values.where(
        values.isna()
        | ((values % 1) == 0)
    )

    return values.astype(
        "Int64"
    )


def _to_numeric(
    series: pd.Series,
) -> pd.Series:
    """
    Conversion numérique sécurisée.
    """

    return pd.to_numeric(
        series,
        errors="coerce",
    )


# ============================================================
# GENERIC UPSERT
# ============================================================

def bulk_upsert(
    table: str,
    df: pd.DataFrame,
    on_conflict: str,
    chunk_size: int = 500,
) -> int:
    """
    Upsert d'un DataFrame vers Supabase
    par blocs.
    """

    if df.empty:
        return 0

    rows = _records(df)

    total = 0

    for i in range(
        0,
        len(rows),
        chunk_size,
    ):

        chunk = rows[
            i:i + chunk_size
        ]

        resp = (
            _client()
            .table(table)
            .upsert(
                chunk,
                on_conflict=on_conflict,
            )
            .execute()
        )

        total += (
            len(_rows(resp))
            or len(chunk)
        )

    return total


# ============================================================
# RACES
# ============================================================

def upsert_races(
    df: pd.DataFrame,
) -> int:
    """
    Insère / met à jour les courses.

    DataFrame ETL :
        race_id
        distance
        reunion
        course_number

    Supabase :
        external_id
        distance_m
        meeting_number
        race_number
    """

    if df.empty:
        return 0

    out = df.copy()

    rename = {
        "race_id": "external_id",
        "distance": "distance_m",
        "reunion": "meeting_number",
        "course_number": "race_number",
    }

    for src, dst in rename.items():

        if (
            src in out.columns
            and dst not in out.columns
        ):
            out[dst] = out[src]

    required = [
        "external_id",
        "race_date",
        "race_number",
        "discipline",
        "hippodrome",
        "distance_m",
        "terrain",
        "field_size",
        "meeting_number",
        "status",
    ]

    for column in required:

        if column not in out.columns:
            out[column] = None

    out = out[
        required
    ].copy()

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    parsed_dates = pd.to_datetime(
        out["race_date"],
        errors="coerce",
    )

    out["race_date"] = parsed_dates.map(
        lambda value: (
            value.date().isoformat()
            if pd.notna(value)
            else None
        )
    )

    # --------------------------------------------------------
    # Identifiant externe
    # --------------------------------------------------------

    out["external_id"] = (
        out["external_id"]
        .astype(str)
    )

    # --------------------------------------------------------
    # Colonnes INTEGER PostgreSQL
    # --------------------------------------------------------

    for column in [
        "race_number",
        "meeting_number",
        "distance_m",
        "field_size",
    ]:

        out[column] = _to_nullable_int(
            out[column]
        )

    return bulk_upsert(
        "races",
        out,
        "external_id",
    )


# ============================================================
# PARTICIPANTS
# ============================================================

def upsert_participants(
    df: pd.DataFrame,
) -> int:
    """
    Insère / met à jour les participants.

    Le race_id venant de l'ETL est un identifiant PMU
    du type :

        R1C4_2026-09-02

    Il est traduit vers races.id, le bigint interne
    de PostgreSQL.
    """

    if df.empty:
        return 0

    out = df.copy()

    if "race_id" not in out.columns:
        raise ValueError(
            "participants: race_id absent"
        )

    # Garder l'identifiant externe original
    out["_external_race_id"] = (
        out["race_id"]
        .astype(str)
    )

    external_ids = (
        out["_external_race_id"]
        .dropna()
        .unique()
        .tolist()
    )

    if not external_ids:
        return 0

    # --------------------------------------------------------
    # Recherche races.id à partir de races.external_id
    # --------------------------------------------------------

    race_response = (
        _client()
        .table("races")
        .select(
            "id,external_id"
        )
        .in_(
            "external_id",
            external_ids,
        )
        .execute()
    )

    race_rows = _rows(
        race_response
    )

    mapping = {
        str(row["external_id"]): row["id"]
        for row in race_rows
    }

    out["race_id"] = (
        out["_external_race_id"]
        .map(mapping)
    )

    # Courses qui n'ont pas trouvé d'ID Supabase
    # sont ignorées.
    out = out[
        out["race_id"].notna()
    ].copy()

    if out.empty:
        return 0

    # --------------------------------------------------------
    # Renommage vers schéma Supabase
    # --------------------------------------------------------

    rename = {
        "jockey": "jockey_name",
        "trainer": "trainer_name",
        "weight": "weight_kg",
    }

    for src, dst in rename.items():

        if (
            src in out.columns
            and dst not in out.columns
        ):
            out[dst] = out[src]

    # --------------------------------------------------------
    # Colonnes manquantes
    # --------------------------------------------------------

    optional_columns = [
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

    for column in optional_columns:

        if column not in out.columns:
            out[column] = None

    # --------------------------------------------------------
    # Types INTEGER PostgreSQL
    # --------------------------------------------------------

    out["race_id"] = _to_nullable_int(
        out["race_id"]
    )

    for column in [
        "horse_number",
        "draw",
        "age",
        "finish_position",
    ]:

        out[column] = _to_nullable_int(
            out[column]
        )

    # --------------------------------------------------------
    # Types numériques décimaux
    # --------------------------------------------------------

    for column in [
        "odds",
        "weight_kg",
    ]:

        out[column] = _to_numeric(
            out[column]
        )

    # --------------------------------------------------------
    # Boolean non-partant
    # --------------------------------------------------------

    out["is_non_runner"] = (
        out["is_non_runner"]
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    # --------------------------------------------------------
    # Identifiant externe participant
    # --------------------------------------------------------

    if "external_id" not in out.columns:

        out["external_id"] = out.apply(
            lambda row: (
                f"{row['_external_race_id']}"
                f"-{_json_safe(row['horse_number'])}"
            ),
            axis=1,
        )

    else:

        missing_external = (
            out["external_id"].isna()
            | out["external_id"]
            .astype(str)
            .isin(
                ["", "nan", "None"]
            )
        )

        out.loc[
            missing_external,
            "external_id",
        ] = out.loc[
            missing_external
        ].apply(
            lambda row: (
                f"{row['_external_race_id']}"
                f"-{_json_safe(row['horse_number'])}"
            ),
            axis=1,
        )

    # --------------------------------------------------------
    # Colonnes finales Supabase
    # --------------------------------------------------------

    required = [
        "race_id",
        "external_id",
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

    out = out[
        required
    ].copy()

    # Numéro du cheval nécessaire pour l'upsert
    out = out[
        out["horse_number"].notna()
    ].copy()

    if out.empty:
        return 0

    return bulk_upsert(
        "participants",
        out,
        "race_id,horse_number",
    )


# ============================================================
# MARKET SNAPSHOTS
# ============================================================

def insert_market_snapshots(
    df: pd.DataFrame,
) -> int:

    if df.empty:
        return 0

    rows = _records(df)

    response = (
        _client()
        .table(
            "market_snapshots"
        )
        .insert(
            rows
        )
        .execute()
    )

    return (
        len(_rows(response))
        or len(rows)
    )


# ============================================================
# RACE QUERIES
# ============================================================

def list_races(
    race_date: date | None = None,
    limit: int = 200,
) -> pd.DataFrame:

    query = (
        _client()
        .table("races")
        .select("*")
        .order(
            "race_date",
            desc=True,
        )
        .limit(limit)
    )

    if race_date:

        query = query.eq(
            "race_date",
            race_date.isoformat(),
        )

    return pd.DataFrame(
        _rows(
            query.execute()
        )
    )


def get_participants(
    race_id: str | int,
) -> pd.DataFrame:

    query = (
        _client()
        .table("participants")
        .select("*")
        .eq(
            "race_id",
            race_id,
        )
        .order(
            "horse_number"
        )
    )

    return pd.DataFrame(
        _rows(
            query.execute()
        )
    )


# ============================================================
# TRAINING HISTORY
# ============================================================

def _get_training_history_paginated(
    race_date: date,
    limit_rows: int = 200_000,
    page_size: int = 1000,
) -> pd.DataFrame:
    """
    Récupère tout l'historique via le RPC Supabase
    en contournant la limite API de 1000 lignes
    grâce à la pagination.
    """

    all_rows: list[dict] = []

    start = 0

    while start < limit_rows:

        end = min(
            start + page_size - 1,
            limit_rows - 1,
        )

        response = (
            _client()
            .rpc(
                "get_training_history_until",
                {
                    "p_race_date": race_date.isoformat(),
                    "p_limit": limit_rows,
                },
            )
            .range(start, end)
            .execute()
        )

        batch = _rows(response)

        if not batch:
            break

        all_rows.extend(batch)

        print(
            f"Historique Supabase : "
            f"{len(all_rows)} lignes chargées..."
        )

        if len(batch) < page_size:
            break

        start += page_size

    if not all_rows:
        return pd.DataFrame(
            columns=[
                "race_id",
                "race_date",
                "discipline",
                "hippodrome",
                "distance",
                "terrain",
                "field_size",
                "horse_number",
                "horse_name",
                "jockey",
                "trainer",
                "odds",
                "draw",
                "weight",
                "recent_form",
                "finish_position",
            ]
        )

    return pd.DataFrame(all_rows)


def get_training_history(
    limit_rows: int = 200_000,
) -> pd.DataFrame:
    """
    Historique strictement antérieur à aujourd'hui.
    """

    return _get_training_history_paginated(
        date.today(),
        limit_rows=limit_rows,
    )


def get_history_as_of(
    race_date: date,
    limit_rows: int = 200_000,
) -> pd.DataFrame:
    """
    Historique strictement antérieur à une date donnée.
    """

    return _get_training_history_paginated(
        race_date,
        limit_rows=limit_rows,
    )
# ============================================================
# INGESTION RUNS
# ============================================================

def save_ingestion_run(
    source: str,
    status: str,
    rows_read: int,
    rows_written: int,
    message: str = "",
    window_start: Any = None,
    window_end: Any = None,
) -> None:
    """
    Sauvegarde le résultat d'une exécution ETL.
    """

    _client().table(
        "ingestion_runs"
    ).insert(
        {
            "source": source,
            "status": status,
            "rows_read": int(rows_read),
            "rows_written": int(rows_written),

            "error_message": (
                str(message)[:4000]
                if status != "success"
                else None
            ),
        }
    ).execute()


# ============================================================
# MODEL VERSIONS
# ============================================================

def save_model_version(
    record: dict[str, Any],
) -> dict[str, Any]:

    rows = _rows(
        _client()
        .table(
            "model_versions"
        )
        .insert(
            record
        )
        .execute()
    )

    if not rows:
        raise RuntimeError(
            "Version modèle non créée"
        )

    return rows[0]


def activate_model(
    model_id: int,
) -> None:
    """
    Désactive l'ancien modèle actif
    et active le nouveau.
    """

    (
        _client()
        .table(
            "model_versions"
        )
        .update(
            {
                "is_active": False
            }
        )
        .eq(
            "is_active",
            True,
        )
        .execute()
    )

    (
        _client()
        .table(
            "model_versions"
        )
        .update(
            {
                "is_active": True
            }
        )
        .eq(
            "id",
            model_id,
        )
        .execute()
    )


def get_latest_active_model(
) -> dict[str, Any] | None:

    rows = _rows(
        _client()
        .table(
            "model_versions"
        )
        .select("*")
        .eq(
            "is_active",
            True,
        )
        .order(
            "trained_at",
            desc=True,
        )
        .limit(1)
        .execute()
    )

    return (
        rows[0]
        if rows
        else None
    )


def list_model_versions(
    limit: int = 20,
) -> pd.DataFrame:

    response = (
        _client()
        .table(
            "model_versions"
        )
        .select(
            "id,"
            "trained_at,"
            "model_name,"
            "model_type,"
            "metrics,"
            "artifact_hash,"
            "is_active,"
            "training_rows,"
            "features"
        )
        .order(
            "trained_at",
            desc=True,
        )
        .limit(
            limit
        )
        .execute()
    )

    return pd.DataFrame(
        _rows(response)
    )


# ============================================================
# BACKTEST
# ============================================================

def save_backtest_run(
    record: dict[str, Any],
) -> dict[str, Any] | None:

    rows = _rows(
        _client()
        .table(
            "backtest_runs"
        )
        .insert(
            record
        )
        .execute()
    )

    return (
        rows[0]
        if rows
        else None
    )


# ============================================================
# PREDICTIONS
# ============================================================

def save_prediction_batch(
    race_id: int,
    model_version_id: int | None,
    predictions: pd.DataFrame,
) -> int:

    rows = []

    for _, row in predictions.iterrows():

        win_probability = _json_safe(
            row["win_probability"]
        )

        place_probability = _json_safe(
            row["place_probability"]
        )

        expected_value = _json_safe(
            row.get(
                "expected_value",
                0,
            )
        )

        rows.append(
            {
                "race_id":
                    int(race_id),

                "participant_id":
                    int(
                        row[
                            "participant_id"
                        ]
                    ),

                "model_version_id":
                    (
                        int(model_version_id)
                        if model_version_id
                        is not None
                        else None
                    ),

                "win_probability":
                    (
                        float(win_probability)
                        if win_probability
                        is not None
                        else None
                    ),

                "place_probability":
                    (
                        float(place_probability)
                        if place_probability
                        is not None
                        else None
                    ),

                "expected_value":
                    (
                        float(expected_value)
                        if expected_value
                        is not None
                        else None
                    ),

                "rank":
                    int(
                        row["rank"]
                    ),
            }
        )

    if not rows:
        return 0

    response = (
        _client()
        .table(
            "predictions"
        )
        .insert(
            rows
        )
        .execute()
    )

    return (
        len(_rows(response))
        or len(rows)
    )
