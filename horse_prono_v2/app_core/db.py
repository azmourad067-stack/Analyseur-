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
        url = st.secrets.get("SUPABASE_URL")
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
        raise RuntimeError("Supabase non configuré : renseigne SUPABASE_URL et SUPABASE_KEY dans les secrets Streamlit.")
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
    if df.empty:
        return 0
    rows = _records(df)
    total = 0
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        resp = _client().table(table).upsert(chunk, on_conflict=on_conflict).execute()
        total += len(_rows(resp)) or len(chunk)
    return total


def upsert_races(df: pd.DataFrame) -> int:
    x = df.copy()
    rename = {"race_id": "external_id", "distance": "distance_m", "reunion": "meeting_number", "course_number": "race_number"}
    x = x.rename(columns={k: v for k, v in rename.items() if k in x.columns})
    keep = ["external_id", "race_date", "meeting_number", "race_number", "hippodrome", "discipline", "distance_m", "terrain", "field_size", "status"]
    x = x[[c for c in keep if c in x.columns]]
    return bulk_upsert("races", x, "external_id")


def upsert_participants(df: pd.DataFrame) -> int:
    x = df.copy()
    rename = {"race_id": "external_race_id", "jockey": "jockey_name", "trainer": "trainer_name", "weight": "weight_kg", "non_runner": "is_non_runner"}
    x = x.rename(columns={k: v for k, v in rename.items() if k in x.columns})
    # Resolve the bigint race FK from races.external_id.
    race_ids = x["external_race_id"].dropna().astype(str).unique().tolist()
    mapping: dict[str, int] = {}
    for i in range(0, len(race_ids), 500):
        vals = race_ids[i:i + 500]
        resp = _client().table("races").select("id,external_id").in_("external_id", vals).execute()
        for row in _rows(resp):
            mapping[str(row["external_id"])] = int(row["id"])
    x["race_id"] = x["external_race_id"].astype(str).map(mapping)
    x = x.dropna(subset=["race_id", "horse_number"]).copy()
    x["race_id"] = x["race_id"].astype(int)
    keep = ["race_id", "external_id", "horse_name", "horse_number", "jockey_name", "trainer_name", "odds", "weight_kg", "draw", "age", "sex", "recent_form", "finish_position", "is_non_runner"]
    for c in keep:
        if c not in x.columns:
            x[c] = None
    x = x[keep]
    return bulk_upsert("participants", x, "race_id,horse_number")


def insert_market_snapshots(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return len(_rows(_client().table("market_snapshots").insert(_records(df)).execute()))


def list_races(race_date: date | None = None, limit: int = 200) -> pd.DataFrame:
    q = _client().table("races").select("*").order("race_date", desc=True).limit(limit)
    if race_date:
        q = q.eq("race_date", race_date.isoformat())
    rows = _rows(q.execute())
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.rename(columns={"id": "race_pk", "external_id": "race_id", "meeting_number": "reunion", "race_number": "course_number", "distance_m": "distance"})
    return out


def get_participants(race_id: str | int) -> pd.DataFrame:
    race = _client().table("races").select("id").eq("external_id", str(race_id)).limit(1).execute()
    race_rows = _rows(race)
    if not race_rows:
        try:
            race_rows = _rows(_client().table("races").select("id").eq("id", int(race_id)).limit(1).execute())
        except Exception:
            pass
    if not race_rows:
        return pd.DataFrame()
    rid = race_rows[0]["id"]
    rows = _rows(_client().table("participants").select("*").eq("race_id", rid).order("horse_number").execute())
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.rename(columns={"jockey_name": "jockey", "trainer_name": "trainer", "weight_kg": "weight", "is_non_runner": "non_runner"})
        out["race_id"] = str(race_id)
    return out


def get_training_history(limit_rows: int = 200_000) -> pd.DataFrame:
    resp = _client().rpc("get_training_history_until", {"p_race_date": date.today().isoformat(), "p_limit": limit_rows}).execute()
    return pd.DataFrame(_rows(resp))


def get_history_as_of(race_date: date, limit_rows: int = 200_000) -> pd.DataFrame:
    resp = _client().rpc("get_training_history_until", {"p_race_date": race_date.isoformat(), "p_limit": limit_rows}).execute()
    return pd.DataFrame(_rows(resp))


def save_ingestion_run(source: str, status: str, rows_read: int, rows_written: int, message: str = "", window_start: Any = None, window_end: Any = None) -> None:
    payload = {"source": source, "status": status, "rows_read": int(rows_read), "rows_written": int(rows_written), "error_message": str(message)[:4000] if status == "error" else None}
    _client().table("ingestion_runs").insert(payload).execute()


def save_model_version(record: dict[str, Any]) -> dict[str, Any]:
    x = dict(record)
    if "feature_columns" in x:
        x["features"] = x.pop("feature_columns")
    rows = _rows(_client().table("model_versions").insert(x).execute())
    if not rows:
        raise RuntimeError("Version modèle non créée")
    return rows[0]


def activate_model(model_id: int) -> None:
    _client().table("model_versions").update({"is_active": False}).eq("is_active", True).execute()
    _client().table("model_versions").update({"is_active": True}).eq("id", model_id).execute()


def get_latest_active_model() -> dict[str, Any] | None:
    rows = _rows(_client().table("model_versions").select("*").eq("is_active", True).order("created_at", desc=True).limit(1).execute())
    if not rows:
        return None
    row = rows[0]
    if "features" in row and "feature_columns" not in row:
        row["feature_columns"] = row["features"]
    return row


def list_model_versions(limit: int = 20) -> pd.DataFrame:
    return pd.DataFrame(_rows(_client().table("model_versions").select("id,created_at,model_name,model_type,metrics,artifact_hash,is_active").order("created_at", desc=True).limit(limit).execute()))


def save_backtest_run(record: dict[str, Any]) -> dict[str, Any] | None:
    return (_rows(_client().table("backtest_runs").insert(record).execute()) or [None])[0]


def save_prediction_batch(race_id: str | int, model_version_id: int | None, predictions: pd.DataFrame) -> int:
    race_rows = _rows(_client().table("races").select("id").eq("external_id", str(race_id)).limit(1).execute())
    if not race_rows:
        try:
            race_rows = _rows(_client().table("races").select("id").eq("id", int(race_id)).limit(1).execute())
        except Exception:
            pass
    if not race_rows:
        raise RuntimeError(f"Course introuvable: {race_id}")
    rid = int(race_rows[0]["id"])
    nums = [int(x) for x in predictions["horse_number"].dropna().tolist()]
    p_rows = _rows(_client().table("participants").select("id,horse_number").eq("race_id", rid).in_("horse_number", nums).execute())
    pmap = {int(r["horse_number"]): int(r["id"]) for r in p_rows}
    rows = []
    for _, r in predictions.iterrows():
        num = int(r["horse_number"])
        if num not in pmap:
            continue
        rows.append({"race_id": rid, "participant_id": pmap[num], "model_version_id": model_version_id, "win_probability": float(r["win_probability"]), "place_probability": float(r["place_probability"]), "expected_value": float(r.get("expected_value", 0)), "rank": int(r["rank"])})
    if not rows:
        return 0
    return len(_rows(_client().table("predictions").insert(rows).execute())) or len(rows)
