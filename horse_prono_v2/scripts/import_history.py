from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from app_core.data import normalize_history
from app_core.db import save_ingestion_run, upsert_participants, upsert_races

API_URL = "https://open-pmu-api.vercel.app/api/arrivees"


def _pick(d: dict[str, Any], *keys, default=None):
    for k in keys:
        cur = d
        try:
            for part in k.split("."):
                cur = cur[part]
            if cur is not None:
                return cur
        except Exception:
            pass
    return default


def _as_int(v):
    try:
        return int(float(v)) if v not in (None, "", "nan") else None
    except Exception:
        return None


def _as_float(v):
    try:
        if isinstance(v, str):
            v = v.replace(",", ".").replace("€", "").strip()
        return float(v) if v not in (None, "", "nan") else None
    except Exception:
        return None


def extract_races(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("message", "races", "data", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
        if isinstance(value, dict):
            nested = extract_races(value)
            if nested:
                return nested
    return [payload]


def normalize_api_payload(payload: Any, target_date: date) -> pd.DataFrame:
    rows = []
    for race in extract_races(payload):
        race_id = _pick(race, "race_id", "id", "id_course", "external_id")
        meeting = _as_int(_pick(race, "reunion", "num_reunion", "numero_reunion", "r"))
        course = _as_int(_pick(race, "course", "num_course", "numero_course", "c"))
        if race_id is None and meeting is not None and course is not None:
            race_id = f"{target_date.isoformat()}_R{meeting}C{course}"
        details = _pick(race, "arrivee_details", "arriveeDetails", default={}) or {}
        finish = _pick(race, "arrivee", "arrival", "resultat", default=None)
        finish_map = {}
        if isinstance(finish, list):
            for pos, num in enumerate(finish, 1):
                n = _as_int(num)
                if n is not None:
                    finish_map[n] = pos
        if isinstance(details, list):
            details = {str(i): x for i, x in enumerate(details)}
        if not isinstance(details, dict):
            details = {}
        participants = _pick(race, "participants", "chevaux", "horses", default=None)
        if not isinstance(participants, list):
            participants = list(details.values()) if details else []
        if not participants and isinstance(race.get("arrivee_details_raw"), dict):
            participants = list(race["arrivee_details_raw"].values())
        for idx, horse in enumerate(participants, 1):
            if not isinstance(horse, dict):
                continue
            number = _as_int(_pick(horse, "numPmu", "numero", "num", "numero_partant", "number", default=idx))
            name = _pick(horse, "nom_cheval", "nomCheval", "cheval.nom", "nom", "horse_name", default="Inconnu")
            row = {
                "race_id": str(race_id), "race_date": pd.Timestamp(target_date),
                "discipline": _pick(horse, "discipline", default=_pick(race, "discipline", "type", default="INCONNU")),
                "hippodrome": _pick(race, "hippodrome", "lieu", "track", default="INCONNU"),
                "distance": _as_int(_pick(horse, "distance", default=_pick(race, "distance", default=None))),
                "terrain": _pick(race, "terrain", "going", default=None),
                "field_size": _as_int(_pick(race, "field_size", "partants", "nombre_partants", default=None)),
                "horse_number": number, "horse_name": str(name),
                "jockey": _pick(horse, "nom_jockey", "jockey.nom", "jockey", "driver", default=""),
                "trainer": _pick(horse, "nom_entraineur", "entraineur.nom", "entraineur", "trainer", default=""),
                "odds": _as_float(_pick(horse, "cote", "odds", "cotes", "cote_finale", default=None)),
                "draw": _as_int(_pick(horse, "corde", "draw", default=None)),
                "weight": _as_float(_pick(horse, "poids", "weight", default=None)),
                "recent_form": _pick(horse, "musique", "performance", "recent_form", default=""),
                "finish_position": _as_int(_pick(horse, "finish_position", "placeFinal", "classement", default=finish_map.get(number))),
            }
            row["non_runner"] = bool(_pick(horse, "non_partant", "non_runner", default=False))
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["race_id", "horse_number"])
    return normalize_history(df)


def import_day(target: date, session: requests.Session, timeout: int = 30) -> tuple[int, int]:
    r = session.get(API_URL, params={"date": target.strftime("%d/%m/%Y")}, timeout=timeout)
    r.raise_for_status()
    df = normalize_api_payload(r.json(), target)
    if df.empty:
        return 0, 0
    races = df[["race_id", "race_date", "discipline", "hippodrome", "distance", "terrain", "field_size"]].drop_duplicates("race_id").copy()
    races["source"] = "open-pmu-api"
    races["status"] = races["race_id"].isin(df.loc[df["finish_position"].notna(), "race_id"]).map({True: "finished", False: "unknown"})
    pcols = ["race_id", "horse_number", "horse_name", "jockey", "trainer", "odds", "draw", "weight", "recent_form", "finish_position", "non_runner"]
    n1 = upsert_races(races)
    n2 = upsert_participants(df[pcols])
    return len(df), n1 + n2


def import_api(start: date, end: date, delay: float) -> None:
    session = requests.Session()
    read = written = errors = 0
    d = start
    while d <= end:
        try:
            r, w = import_day(d, session)
            read += r; written += w
            print(f"{d}: {r} partants lus, {w} lignes écrites")
        except Exception as exc:
            errors += 1
            print(f"{d}: ERREUR — {exc}")
        if d < end:
            time.sleep(max(0.0, delay))
        d += timedelta(days=1)
    status = "success" if errors == 0 else ("partial" if written else "error")
    save_ingestion_run("open-pmu-api", status, read, written, f"{errors} jour(s) en erreur", start, end)
    if errors:
        print(f"Terminé avec {errors} jour(s) en erreur.")


def import_csv(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Fichier introuvable: {p}")
    df = normalize_history(pd.read_csv(p))
    if df.empty:
        raise SystemExit("CSV vide")
    races = df[["race_id", "race_date", "discipline", "hippodrome", "distance", "terrain", "field_size"]].drop_duplicates("race_id").copy()
    races["source"] = f"csv:{p.name}"
    finished = set(df.loc[df["finish_position"].notna(), "race_id"].astype(str))
    races["status"] = races["race_id"].astype(str).map(lambda x: "finished" if x in finished else "unknown")
    pcols = ["race_id", "horse_number", "horse_name", "jockey", "trainer", "odds", "draw", "weight", "recent_form", "finish_position"]
    n1 = upsert_races(races)
    n2 = upsert_participants(df[pcols])
    save_ingestion_run("csv", "success", len(df), n1 + n2, f"{n1} courses, {n2} partants")
    print(f"Import CSV terminé: {n1} courses / {n2} partants")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import historique HorseProno V3")
    sub = parser.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("api")
    a.add_argument("--start", required=True)
    a.add_argument("--end", required=True)
    a.add_argument("--delay", type=float, default=1.0)
    c = sub.add_parser("csv")
    c.add_argument("path")
    args = parser.parse_args()
    if args.mode == "api":
        import_api(date.fromisoformat(args.start), date.fromisoformat(args.end), args.delay)
    else:
        import_csv(args.path)
