from __future__ import annotations

from datetime import date
from typing import Any

import os

import pandas as pd
import streamlit as st
from supabase import Client, create_client


@st.cache_resource
def get_supabase_client() -> Client | None:
    """Create one cached Supabase client.

    The Streamlit app reads credentials from st.secrets. Never commit these secrets.
    """
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
    except Exception:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return None
    return create_client(str(url), str(key))


def is_configured() -> bool:
    return get_supabase_client() is not None


def _rows(response: Any) -> list[dict]:
    return list(getattr(response, "data", None) or [])


def healthcheck() -> tuple[bool, str]:
    client = get_supabase_client()
    if client is None:
        return False, "Supabase non configuré (secrets absents)."
    try:
        client.table("races").select("race_id").limit(1).execute()
        return True, "Connexion Supabase opérationnelle."
    except Exception as exc:
        return False, f"Supabase inaccessible : {exc}"


def upsert_races(df: pd.DataFrame) -> int:
    client = _required_client()
    rows = _records(df)
    if not rows:
        return 0
    response = client.table("races").upsert(rows, on_conflict="race_id").execute()
    return len(_rows(response)) or len(rows)


def upsert_participants(df: pd.DataFrame) -> int:
    client = _required_client()
    rows = _records(df)
    if not rows:
        return 0
    # One race can contain a horse with the same number only once.
    response = client.table("participants").upsert(
        rows, on_conflict="race_id,horse_number"
    ).execute()
    return len(_rows(response)) or len(rows)


def list_races(race_date: date | None = None, limit: int = 100) -> pd.DataFrame:
    client = _required_client()
    query = client.table("races").select("*").order("race_date", desc=True).limit(limit)
    if race_date is not None:
        query = query.eq("race_date", race_date.isoformat())
    return pd.DataFrame(_rows(query.execute()))


def get_participants(race_id: str) -> pd.DataFrame:
    client = _required_client()
    response = (
        client.table("participants")
        .select("*")
        .eq("race_id", race_id)
        .order("horse_number")
        .execute()
    )
    return pd.DataFrame(_rows(response))


def get_training_history(limit_rows: int = 100_000) -> pd.DataFrame:
    """Return finished runners suitable for model training."""
    client = _required_client()
    response = (
        client.table("participants")
        .select("*, races(race_date,discipline,hippodrome,distance,terrain,field_size)")
        .not_.is_("finish_position", "null")
        .order("race_date", desc=False)
        .limit(limit_rows)
        .execute()
    )
    rows = _rows(response)
    if not rows:
        return pd.DataFrame()

    flat = []
    for row in rows:
        race = row.pop("races", {}) or {}
        row.update({k: race.get(k) for k in ["race_date", "discipline", "hippodrome", "distance", "terrain", "field_size"]})
        flat.append(row)
    return pd.DataFrame(flat)


def save_ingestion_run(source: str, status: str, rows_read: int, rows_written: int, message: str = "") -> None:
    client = _required_client()
    client.table("ingestion_runs").insert({
        "source": source,
        "status": status,
        "rows_read": int(rows_read),
        "rows_written": int(rows_written),
        "message": message[:2000],
    }).execute()


def save_model_version(model_record: dict[str, Any]) -> dict[str, Any]:
    client = _required_client()
    response = client.table("model_versions").insert(model_record).execute()
    rows = _rows(response)
    if not rows:
        raise RuntimeError("Supabase n'a pas retourné la version du modèle créée.")
    return rows[0]


def get_latest_active_model() -> dict[str, Any] | None:
    client = _required_client()
    response = (
        client.table("model_versions")
        .select("*")
        .eq("is_active", True)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    rows = _rows(response)
    return rows[0] if rows else None


def save_prediction_batch(race_id: str, model_version_id: int | None, predictions: pd.DataFrame) -> int:
    client = _required_client()
    rows: list[dict[str, Any]] = []
    for _, r in predictions.iterrows():
        rows.append({
            "race_id": race_id,
            "model_version_id": model_version_id,
            "horse_number": _safe_int(r.get("horse_number")),
            "win_probability": _safe_float(r.get("win_probability")),
            "place_probability": _safe_float(r.get("place_probability")),
            "rank": _safe_int(r.get("rank")),
            "explanation": str(r.get("explanation", ""))[:2000],
        })
    if not rows:
        return 0
    response = client.table("predictions").insert(rows).execute()
    return len(_rows(response)) or len(rows)


def _required_client() -> Client:
    client = get_supabase_client()
    if client is None:
        raise RuntimeError("Supabase n'est pas configuré. Ajoute SUPABASE_URL et SUPABASE_KEY dans secrets.")
    return client


def _safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    clean = df.copy()
    clean = clean.where(pd.notna(clean), None)
    records = clean.to_dict(orient="records")
    # Ensure pandas timestamps become ISO strings accepted by the API.
    for row in records:
        for key, value in list(row.items()):
            if isinstance(value, pd.Timestamp):
                row[key] = value.isoformat()
    return records
