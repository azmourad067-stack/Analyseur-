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
        c.table("races").select("id").limit(1).execute()
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
    """Upsert races using the production schema.

    The ETL dataframe historically uses race_id/distance/reunion/course_number;
    production Supabase uses external_id/distance_m/meeting_number/race_number.
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
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
    required = ["external_id", "race_date", "race_number", "discipline", "hippodrome",
                "distance_m", "terrain", "field_size", "meeting_number", "status"]
    for c in required:
        if c not in out.columns:
            out[c] = None
    out = out[required].copy()
    out["race_date"] = pd.to_datetime(out["race_date"], errors="coerce").dt.date.astype(str)
    out["external_id"] = out["external_id"].astype(str)
    return bulk_upsert("races", out, "external_id")


def upsert_participants(df: pd.DataFrame) -> int:
    """Upsert participants, translating external race IDs to races.id."""
    if df.empty:
        return 0
    out = df.copy()
    if "race_id" not in out.columns:
        raise ValueError("participants: race_id absent")
    external_ids = out["race_id"].astype(str).dropna().unique().tolist()
    if not external_ids:
        return 0
    race_rows = _rows(_client().table("races").select("id,external_id").in_("external_id", external_ids).execute())
    mapping = {str(r["external_id"]): r["id"] for r in race_rows}
    out["race_id"] = out["race_id"].astype(str).map(mapping)
    out = out[out["race_id"].notna()].copy()
    if out.empty:
        return 0
    rename = {
        "jockey": "jockey_name",
        "trainer": "trainer_name",
        "weight": "weight_kg",
    }
    for src, dst in rename.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
    if "external_id" not in out.columns:
        out["external_id"] = out.apply(lambda r: f"{r['race_id']}-{r.get('horse_number')}", axis=1)
    required = ["race_id", "external_id", "horse_name", "horse_number", "jockey_name",
                "trainer_name", "odds", "weight_kg", "draw", "age", "sex",
                "recent_form", "finish_position", "is_non_runner"]
    for c in required:
        if c not in out.columns:
            out[c] = None
    out = out[required].copy()
    out["race_id"] = out["race_id"].astype(int)
    out["horse_number"] = pd.to_numeric(out["horse_number"], errors="coerce").astype("Int64")
    out["is_non_runner"] = out["is_non_runner"].fillna(False).astype(bool)
    return bulk_upsert("participants", out, "race_id,horse_number")

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
    rows = _rows(resp)
    if not rows:
        return pd.DataFrame(columns=[
            "race_id", "race_date", "discipline", "hippodrome", "distance", "terrain",
            "field_size", "horse_number", "horse_name", "jockey", "trainer", "odds",
            "draw", "weight", "recent_form", "finish_position"
        ])
    return pd.DataFrame(rows)


def get_history_as_of(race_date: date, limit_rows: int = 200_000) -> pd.DataFrame:
    resp = _client().rpc("get_training_history_until", {"p_race_date": race_date.isoformat(), "p_limit": limit_rows}).execute()
    return pd.DataFrame(_rows(resp))


def save_ingestion_run(source: str, status: str, rows_read: int, rows_written: int, message: str = "", window_start: Any = None, window_end: Any = None) -> None:
    _client().table("ingestion_runs").insert({
        "source": source, "status": status, "rows_read": int(rows_read), "rows_written": int(rows_written),
        "error_message": str(message)[:4000] if status != "success" else None,
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
    rows = _rows(_client().table("model_versions").select("*").eq("is_active", True).order("trained_at", desc=True).limit(1).execute())
    return rows[0] if rows else None


def list_model_versions(limit: int = 20) -> pd.DataFrame:
    return pd.DataFrame(_rows(_client().table("model_versions").select("id,trained_at,model_name,model_type,metrics,artifact_hash,is_active,training_rows,features").order("trained_at", desc=True).limit(limit).execute()))


def save_backtest_run(record: dict[str, Any]) -> dict[str, Any] | None:
    rows = _rows(_client().table("backtest_runs").insert(record).execute())
    return rows[0] if rows else None


def save_prediction_batch(race_id: int, model_version_id: int | None, predictions: pd.DataFrame) -> int:
    rows = []
    for _, r in predictions.iterrows():
        rows.append({
            "race_id": int(race_id),
            "participant_id": int(r["participant_id"]),
            "model_version_id": model_version_id,
            "win_probability": float(r["win_probability"]),
            "place_probability": float(r["place_probability"]),
            "expected_value": float(r.get("expected_value", 0)),
            "rank": int(r["rank"]),
        })
    if not rows:
        return 0
    return len(_rows(_client().table("predictions").insert(rows).execute())) or len(rows)
