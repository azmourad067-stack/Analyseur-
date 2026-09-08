from __future__ import annotations

import os
from datetime import date
from typing import Any

import pandas as pd
import streamlit as st
from supabase import Client, create_client


@st.cache_resource
def get_supabase_client() -> Client | None:
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


def _client() -> Client:
    c = get_supabase_client()
    if c is None:
        raise RuntimeError("Supabase non configuré")
    return c


def _rows(resp: Any) -> list[dict]:
    return list(getattr(resp, "data", None) or [])


def healthcheck() -> tuple[bool, str]:
    c = get_supabase_client()
    if c is None:
        return False, "Supabase non configuré (secrets absents)."
    try:
        c.table("races").select("race_id").limit(1).execute()
        return True, "Connexion Supabase opérationnelle."
    except Exception as exc:
        return False, f"Supabase inaccessible : {exc}"


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    clean = df.copy().where(pd.notna(df), None)
    rows = clean.to_dict(orient="records")
    for row in rows:
        for k, v in list(row.items()):
            if isinstance(v, pd.Timestamp):
                row[k] = v.isoformat()
    return rows


def bulk_upsert(table: str, df: pd.DataFrame, on_conflict: str, chunk_size: int = 500) -> int:
    rows = _records(df)
    total = 0
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        resp = _client().table(table).upsert(chunk, on_conflict=on_conflict).execute()
        total += len(_rows(resp)) or len(chunk)
    return total


def upsert_races(df: pd.DataFrame) -> int:
    return bulk_upsert("races", df, "race_id")


def upsert_participants(df: pd.DataFrame) -> int:
    return bulk_upsert("participants", df, "race_id,horse_number")


def insert_market_snapshots(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = _records(df)
    return len(_rows(_client().table("market_snapshots").insert(rows).execute())) or len(rows)


def list_races(race_date: date | None = None, limit: int = 200) -> pd.DataFrame:
    q = _client().table("races").select("*").order("race_date", desc=True).limit(limit)
    if race_date:
        q = q.eq("race_date", race_date.isoformat())
    return pd.DataFrame(_rows(q.execute()))


def get_participants(race_id: str) -> pd.DataFrame:
    q = _client().table("participants").select("*").eq("race_id", race_id).order("horse_number")
    return pd.DataFrame(_rows(q.execute()))


def get_training_history(limit_rows: int = 200_000) -> pd.DataFrame:
    """Use the SQL function so temporal filtering happens server-side."""
    resp = _client().rpc("get_training_history_until", {"p_race_date": date.today().isoformat(), "p_limit": limit_rows}).execute()
    return pd.DataFrame(_rows(resp))


def get_history_as_of(race_date: date, limit_rows: int = 200_000) -> pd.DataFrame:
    resp = _client().rpc("get_training_history_until", {"p_race_date": race_date.isoformat(), "p_limit": limit_rows}).execute()
    return pd.DataFrame(_rows(resp))


def save_ingestion_run(source: str, status: str, rows_read: int, rows_written: int, message: str = "", window_start: Any = None, window_end: Any = None) -> None:
    _client().table("ingestion_runs").insert({
        "source": source, "status": status, "rows_read": int(rows_read), "rows_written": int(rows_written),
        "message": str(message)[:4000], "window_start": window_start.isoformat() if hasattr(window_start, "isoformat") else window_start,
        "window_end": window_end.isoformat() if hasattr(window_end, "isoformat") else window_end,
    }).execute()


def save_model_version(record: dict[str, Any]) -> dict[str, Any]:
    rows = _rows(_client().table("model_versions").insert(record).execute())
    if not rows:
        raise RuntimeError("Version modèle non créée")
    return rows[0]


def activate_model(model_id: int) -> None:
    # PostgREST updates are used instead of requiring a privileged SQL function.
    _client().table("model_versions").update({"is_active": False}).eq("is_active", True).execute()
    _client().table("model_versions").update({"is_active": True}).eq("id", model_id).execute()


def get_latest_active_model() -> dict[str, Any] | None:
    rows = _rows(_client().table("model_versions").select("*").eq("is_active", True).order("created_at", desc=True).limit(1).execute())
    return rows[0] if rows else None


def list_model_versions(limit: int = 20) -> pd.DataFrame:
    return pd.DataFrame(_rows(_client().table("model_versions").select("id,created_at,model_name,model_type,metrics,artifact_hash,is_active").order("created_at", desc=True).limit(limit).execute()))


def save_backtest_run(record: dict[str, Any]) -> dict[str, Any] | None:
    rows = _rows(_client().table("backtest_runs").insert(record).execute())
    return rows[0] if rows else None


def save_prediction_batch(race_id: str, model_version_id: int | None, predictions: pd.DataFrame) -> int:
    rows = []
    for _, r in predictions.iterrows():
        rows.append({
            "race_id": race_id, "model_version_id": model_version_id,
            "horse_number": int(r["horse_number"]), "win_probability": float(r["win_probability"]),
            "place_probability": float(r["place_probability"]), "market_probability": float(r.get("market_probability", 0)),
            "expected_value": float(r.get("expected_value", 0)), "rank": int(r["rank"]),
            "recommendation": str(r.get("recommendation", "")), "explanation": str(r.get("explanation", ""))[:2000],
        })
    if not rows:
        return 0
    return len(_rows(_client().table("predictions").insert(rows).execute())) or len(rows)
