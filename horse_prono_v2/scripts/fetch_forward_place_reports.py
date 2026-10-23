#!/usr/bin/env python3
"""Collecte les rapports définitifs PMU des courses ayant des prédictions M8/Neural.

Les payloads complets sont stockés dans quinte_payouts_raw.
Usage:
  python scripts/fetch_forward_place_reports.py --start 2026-09-09 --end 2026-10-23
  python scripts/fetch_forward_place_reports.py --start 2026-09-09 --end 2026-10-23 --force
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import date

import requests
from supabase import create_client

PMU_BASE_URL = "https://online.turfinfo.api.pmu.fr/rest/client/1"
HEADERS = {"User-Agent": "HorseProno/1.0", "Accept": "application/json"}
TIMEOUT = 20
PAGE_SIZE = 1000
PAUSE_SECONDS = 0.4
MODEL_IDS = (8, 9)


def paged_query(query):
    rows = []
    start = 0
    while True:
        batch = query.range(start, start + PAGE_SIZE - 1).execute().data or []
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            return rows
        start += PAGE_SIZE


def fetch_final_reports(race_date: date, meeting: int, race: int):
    url = (
        f"{PMU_BASE_URL}/programme/{race_date.strftime('%d%m%Y')}"
        f"/R{meeting}/C{race}/rapports-definitifs"
    )
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                params={"combinaisonEnTableau": "true", "specialisation": "INTERNET"},
                timeout=TIMEOUT,
            )
            if response.status_code in (204, 404):
                return None
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"Échec PMU pour {race_date} R{meeting}C{race}: {last_error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Date de début YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Date de fin YYYY-MM-DD")
    parser.add_argument("--force", action="store_true", help="Retélécharger aussi les payloads déjà stockés")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    if end_date < start_date:
        parser.error("--end doit être >= --start")

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise SystemExit("Variables SUPABASE_URL et SUPABASE_SERVICE_KEY requises.")
    client = create_client(url, key)

    races = paged_query(
        client.table("races")
        .select("id,race_date,meeting_number,race_number")
        .gte("race_date", start_date.isoformat())
        .lte("race_date", end_date.isoformat())
        .order("race_date")
    )
    if not races:
        print("Aucune course dans cette période.")
        return

    predicted_ids = set()
    race_ids = [int(r["id"]) for r in races]
    for offset in range(0, len(race_ids), 200):
        batch_ids = race_ids[offset:offset + 200]
        for model_id in MODEL_IDS:
            preds = paged_query(
                client.table("forward_model_predictions")
                .select("race_id")
                .in_("race_id", batch_ids)
                .eq("model_version_id", model_id)
            )
            predicted_ids.update(int(p["race_id"]) for p in preds if p.get("race_id") is not None)

    targets = [r for r in races if int(r["id"]) in predicted_ids]
    existing = set()
    if not args.force:
        raw_rows = paged_query(
            client.table("quinte_payouts_raw")
            .select("race_date,meeting_number,race_number")
            .gte("race_date", start_date.isoformat())
            .lte("race_date", end_date.isoformat())
            .order("race_date")
        )
        existing = {
            (str(r["race_date"]), int(r["meeting_number"]), int(r["race_number"]))
            for r in raw_rows
        }

    todo = [
        r for r in targets
        if args.force or (
            str(r["race_date"]), int(r["meeting_number"]), int(r["race_number"])
        ) not in existing
    ]
    print(
        f"Période {start_date} → {end_date} | courses en base: {len(races)} | "
        f"avec prédictions M8/Neural: {len(targets)} | déjà stockées: {len(existing)} | "
        f"à interroger: {len(todo)}",
        flush=True,
    )

    saved = unavailable = errors = 0
    for index, race_row in enumerate(todo, start=1):
        race_date = date.fromisoformat(str(race_row["race_date"])[:10])
        meeting = int(race_row["meeting_number"])
        course = int(race_row["race_number"])
        key_row = {
            "race_date": race_date.isoformat(),
            "meeting_number": meeting,
            "race_number": course,
        }
        try:
            payload = fetch_final_reports(race_date, meeting, course)
            if payload is None:
                unavailable += 1
                print(f"[{index}/{len(todo)}] {race_date} R{meeting}C{course}: rapports non publiés/indisponibles", flush=True)
            else:
                client.table("quinte_payouts_raw").upsert(
                    {**key_row, "payload": payload},
                    on_conflict="race_date,meeting_number,race_number",
                ).execute()
                saved += 1
                print(f"[{index}/{len(todo)}] {race_date} R{meeting}C{course}: payload enregistré", flush=True)
        except Exception as exc:
            errors += 1
            print(f"[{index}/{len(todo)}] {race_date} R{meeting}C{course}: ERREUR {exc}", flush=True)
        time.sleep(PAUSE_SECONDS)

    print(
        f"Terminé — payloads enregistrés: {saved}, sans rapport: {unavailable}, erreurs: {errors}.",
        flush=True,
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
