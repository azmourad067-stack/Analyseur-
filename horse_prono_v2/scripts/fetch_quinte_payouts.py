"""
Récupère les rapports définitifs PMU des Quinté+ listés dans `quinte_supports`
et les stocke dans Supabase (`quinte_payouts_raw` + `quinte_payouts`).

Usage :
    # Test sans base de données (affiche les rapports parsés d'une course)
    python horse_prono_v2/scripts/fetch_quinte_payouts.py --probe 2026-07-18 1 4

    # Récupération complète (SUPABASE_URL + SUPABASE_SERVICE_KEY requis)
    python horse_prono_v2/scripts/fetch_quinte_payouts.py
    python horse_prono_v2/scripts/fetch_quinte_payouts.py --since 2026-06-01 --until 2026-09-08
    python horse_prono_v2/scripts/fetch_quinte_payouts.py --force      # re-télécharge tout
    python horse_prono_v2/scripts/fetch_quinte_payouts.py --dry-run    # pas d'écriture en base

Le JSON brut est toujours conservé : si le parsing doit évoluer, on le refait
depuis `quinte_payouts_raw` sans rappeler l'API PMU.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata
from datetime import date, datetime
from pathlib import Path

import requests

PMU_BASE_URL = "https://online.turfinfo.api.pmu.fr/rest/client/1"
HEADERS = {"User-Agent": "HorseProno/1.0", "Accept": "application/json"}
REQUEST_TIMEOUT = 20
PAGE_SIZE = 1000
PAUSE_SECONDS = 0.4

HORSE_PRONO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = HORSE_PRONO_ROOT / "artifacts" / "quinte_payouts"

UPSERT_CONFLICT = "race_date,meeting_number,race_number,bet_type,category,combination_key"


# ============================================================
# PARSING
# ============================================================

def _norm(text) -> str:
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip().lower()


def _safe_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_category(libelle, type_rapport=None) -> str:
    """ORDRE / ORDRE_TIRELIRE / DESORDRE / BONUS_4SUR5 / BONUS_4 / BONUS_3 / TIRELIRE / OTHER."""
    text = _norm(f"{libelle or ''} {type_rapport or ''}")
    if "tirelire" in text:          # « Ordre + e-Tirelire » est un rapport distinct de « Ordre »
        return "ORDRE_TIRELIRE" if re.search(r"\bordre\b", text) else "TIRELIRE"
    if "desordre" in text:          # à tester AVANT "ordre" (désordre contient « ordre »)
        return "DESORDRE"
    if "bonus" in text:
        if re.search(r"bonus\s*4\s*(sur|/)\s*5", text) or "4sur5" in text:
            return "BONUS_4SUR5"
        if re.search(r"bonus\s*4", text):
            return "BONUS_4"
        if re.search(r"bonus\s*3", text):
            return "BONUS_3"
        return "OTHER"
    if re.search(r"\bordre\b", text):
        return "ORDRE"
    return "OTHER"


def iter_bet_blocks(obj):
    """Parcourt le JSON et renvoie chaque bloc {typePari, rapports}."""
    if isinstance(obj, dict):
        if "typePari" in obj and "rapports" in obj:
            yield obj
        for value in obj.values():
            yield from iter_bet_blocks(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_bet_blocks(value)


def combination_numbers(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, float):
        return [int(value)] if value.is_integer() else []
    if isinstance(value, str):
        return [int(x) for x in re.findall(r"\d+", value)]
    if isinstance(value, list):
        out: list[int] = []
        for item in value:
            out.extend(combination_numbers(item))
        return out
    if isinstance(value, dict):
        for key in ("numPmu", "numero", "numParticipant", "cheval"):
            if key in value:
                nums = combination_numbers(value[key])
                if nums:
                    return nums
        out = []
        for item in value.values():
            out.extend(combination_numbers(item))
        return out
    return []


def per_euro_dividend(block: dict, report: dict):
    """
    Rapport pour 1 € en euros.

    L'API PMU exprime les montants en centimes :
      - dividendePourUneMiseDeBase : gain pour la mise de base (ex. 2 € -> 200 centimes)
      - miseBase                   : mise de base en centimes
      - dividendePourUnEuro        : gain pour 1 € en centimes

    On privilégie dividendePourUneMiseDeBase / miseBase (pas de plafond, contrairement
    au backtest placé : un Quinté+ ordre dépasse 1000 €/€), puis dividendePourUnEuro / 100.
    Renvoie (valeur, warning|None).
    """
    base_stake = _safe_float(block.get("miseBase"))
    div_base = _safe_float(report.get("dividendePourUneMiseDeBase"))
    div_euro = _safe_float(report.get("dividendePourUnEuro"))

    via_base = None
    if base_stake and base_stake > 0 and div_base is not None and div_base >= 0:
        via_base = div_base / base_stake
    via_euro = div_euro / 100.0 if div_euro is not None and div_euro >= 0 else None

    if via_base is not None and via_euro is not None:
        denom = max(via_base, via_euro, 1e-9)
        if abs(via_base - via_euro) / denom > 0.01:
            return via_base, f"ecart_base_vs_euro:{via_base:.2f}/{via_euro:.2f}"
        return via_base, None
    if via_base is not None:
        return via_base, None
    if via_euro is not None:
        return via_euro, None
    return None, "dividende_introuvable"


def parse_payload(payload) -> list[dict]:
    rows: list[dict] = []
    seen: dict[tuple, int] = {}
    for block in iter_bet_blocks(payload):
        bet_type = str(block.get("typePari") or "").strip().upper()
        if "QUINTE" not in bet_type:
            continue
        base_stake_c = _safe_float(block.get("miseBase"))
        reports = block.get("rapports") or []
        if not isinstance(reports, list):
            continue
        for report in reports:
            if not isinstance(report, dict):
                continue
            libelle = report.get("libelle") or report.get("libelleRapport") or ""
            category = classify_category(libelle, report.get("typeRapport"))
            combo = combination_numbers(report.get("combinaison"))
            per_euro, warning = per_euro_dividend(block, report)
            div_base_c = _safe_float(report.get("dividendePourUneMiseDeBase"))
            winners = report.get("nombreGagnants")
            key = "-".join(map(str, combo))
            count = seen.get((bet_type, category, key), 0)
            seen[(bet_type, category, key)] = count + 1
            if count:                       # même clé déjà vue dans cette course : on garde la ligne
                key = f"{key}#{count + 1}"
            rows.append(
                {
                    "bet_type": bet_type,
                    "category": category,
                    "libelle": str(libelle) or None,
                    "combination": combo or None,
                    "combination_key": key,
                    "dividend_per_euro": round(per_euro, 2) if per_euro is not None else None,
                    "dividend_per_base": round(div_base_c / 100.0, 2) if div_base_c is not None else None,
                    "base_stake": round(base_stake_c / 100.0, 2) if base_stake_c is not None else None,
                    "winners": round(float(winners), 2) if isinstance(winners, (int, float)) and not isinstance(winners, bool) else None,
                    "parse_warning": warning,
                }
            )
    return rows


# ============================================================
# API PMU
# ============================================================

def fetch_final_reports(race_date: date, reunion: int, course: int):
    """Renvoie le JSON, ou None si aucun rapport n'est publié (204/404)."""
    url = f"{PMU_BASE_URL}/programme/{race_date.strftime('%d%m%Y')}/R{reunion}/C{course}/rapports-definitifs"
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                params={"combinaisonEnTableau": "true", "specialisation": "INTERNET"},
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code in (204, 404):
                return None
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"Rapports PMU indisponibles ({url}) : {last_error}")


# ============================================================
# SUPABASE
# ============================================================

def get_client():
    from supabase import create_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise SystemExit("SUPABASE_URL et SUPABASE_SERVICE_KEY sont requis.")
    return create_client(url, key)


def load_all(client, table: str, columns: str, **filters):
    out, start = [], 0
    while True:
        query = client.table(table).select(columns)
        if filters.get("since"):
            query = query.gte("race_date", filters["since"])
        if filters.get("until"):
            query = query.lte("race_date", filters["until"])
        batch = query.order("race_date").range(start, start + PAGE_SIZE - 1).execute().data or []
        out.extend(batch)
        if len(batch) < PAGE_SIZE:
            return out
        start += PAGE_SIZE


# ============================================================
# MAIN
# ============================================================

def run_probe(args) -> int:
    race_date = datetime.strptime(args.probe[0], "%Y-%m-%d").date()
    payload = fetch_final_reports(race_date, int(args.probe[1]), int(args.probe[2]))
    if payload is None:
        print("Aucun rapport publié pour cette course.")
        return 1
    rows = parse_payload(payload)
    if not rows:
        print("Aucun bloc Quinté+ trouvé. Types de paris présents :")
        print(sorted({str(b.get('typePari')) for b in iter_bet_blocks(payload)}))
        return 1
    for r in rows:
        print(
            f"{r['category']:<12} comb={r['combination_key']:<16} "
            f"pour 1€={r['dividend_per_euro']!s:>12}  pour {r['base_stake']}€={r['dividend_per_base']!s:>12}  "
            f"gagnants={r['winners']}  {r['parse_warning'] or ''}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since")
    parser.add_argument("--until")
    parser.add_argument("--force", action="store_true", help="re-télécharge même si déjà en base")
    parser.add_argument("--dry-run", action="store_true", help="pas d'écriture en base")
    parser.add_argument("--probe", nargs=3, metavar=("DATE", "REUNION", "COURSE"),
                        help="teste une course sans base de données")
    args = parser.parse_args()

    if args.probe:
        return run_probe(args)

    client = get_client()
    supports = load_all(client, "quinte_supports", "race_date,meeting_number,race_number", since=args.since, until=args.until)
    done = set()
    if not args.force:
        for r in load_all(client, "quinte_payouts", "race_date,meeting_number,race_number", since=args.since, until=args.until):
            done.add((r["race_date"], r["meeting_number"], r["race_number"]))

    todo = [s for s in supports if (s["race_date"], s["meeting_number"], s["race_number"]) not in done]
    print(f"Filtres : since={args.since or '-'} until={args.until or '-'} force={args.force}", flush=True)
    print(f"Quinté+ : {len(supports)} | déjà en base (rapports parsés) : {len(done)} | à récupérer : {len(todo)}", flush=True)

    all_rows, failures, empty = [], 0, 0
    for i, s in enumerate(todo, 1):
        d = datetime.strptime(s["race_date"], "%Y-%m-%d").date()
        key = {"race_date": s["race_date"], "meeting_number": s["meeting_number"], "race_number": s["race_number"]}
        try:
            payload = fetch_final_reports(d, s["meeting_number"], s["race_number"])
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[{i}/{len(todo)}] {s['race_date']} R{s['meeting_number']}C{s['race_number']} ERREUR : {exc}", flush=True)
            continue
        if payload is None:
            empty += 1
            print(f"[{i}/{len(todo)}] {s['race_date']} R{s['meeting_number']}C{s['race_number']} : aucun rapport publié", flush=True)
            continue

        rows = [{**key, **r} for r in parse_payload(payload)]
        cats = sorted({r["category"] for r in rows})
        print(f"[{i}/{len(todo)}] {s['race_date']} R{s['meeting_number']}C{s['race_number']} : {len(rows)} rapports {cats}", flush=True)
        if not rows:
            empty += 1
        all_rows.extend(rows)

        if not args.dry_run:
            client.table("quinte_payouts_raw").upsert({**key, "payload": payload}, on_conflict="race_date,meeting_number,race_number").execute()
            if rows:
                client.table("quinte_payouts").upsert(rows, on_conflict=UPSERT_CONFLICT).execute()
        time.sleep(PAUSE_SECONDS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / "quinte_payouts.csv"
    fields = ["race_date", "meeting_number", "race_number", "bet_type", "category", "libelle",
              "combination_key", "dividend_per_euro", "dividend_per_base", "base_stake", "winners", "parse_warning"]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    warned = sum(1 for r in all_rows if r.get("parse_warning"))
    print(f"\nTerminé : {len(all_rows)} rapports | courses sans rapport : {empty} | erreurs : {failures} | avertissements parsing : {warned}")
    print(f"CSV : {out_csv}")
    return 1 if todo and failures > len(todo) / 2 else 0


if __name__ == "__main__":
    sys.exit(main())
