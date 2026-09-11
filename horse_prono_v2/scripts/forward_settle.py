from __future__ import annotations

import os
import sys

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app_core.db import get_supabase_client, upsert_participants
from app_core.forward_utils import non_runner_numbers
from app_core.pmu import get_participants, participants_to_df


RESULT_GRACE_MINUTES = 15


def _rows(response) -> list[dict]:
    return list(getattr(response, "data", None) or [])


def _current_experiment(client) -> dict | None:
    rows = _rows(
        client.table("forward_experiments")
        .select("*")
        .in_("status", ["active", "paused"])
        .order("started_at")
        .limit(1)
        .execute()
    )
    return rows[0] if rows else None


def _race_rows(client, race_ids: list[int]) -> dict[int, dict]:
    if not race_ids:
        return {}

    rows = _rows(
        client.table("races")
        .select(
            "id,external_id,race_date,"
            "meeting_number,race_number,"
            "discipline,hippodrome,distance_m,"
            "terrain,field_size,status"
        )
        .in_("id", race_ids)
        .execute()
    )
    return {int(row["id"]): row for row in rows}


def _result_map(result_df: pd.DataFrame) -> dict[int, int | None]:
    mapping: dict[int, int | None] = {}

    if result_df.empty:
        return mapping

    numbers = pd.to_numeric(
        result_df["horse_number"],
        errors="coerce",
    )
    positions = pd.to_numeric(
        result_df["finish_position"],
        errors="coerce",
    )

    for number, position in zip(numbers, positions):
        if pd.isna(number):
            continue
        mapping[int(number)] = (
            int(position)
            if pd.notna(position)
            else None
        )

    return mapping


def _summary(client, experiment: dict) -> dict[str, Any]:
    experiment_id = int(experiment["id"])

    rows = _rows(
        client.table("forward_bets")
        .select("id,status,stake,gross_return,profit")
        .eq("experiment_id", experiment_id)
        .execute()
    )

    counts = {
        "pending": 0,
        "won": 0,
        "lost": 0,
        "void": 0,
    }

    stake = 0.0
    gross = 0.0
    profit = 0.0

    for row in rows:
        status = str(row.get("status") or "")
        if status in counts:
            counts[status] += 1

        if status in ("won", "lost"):
            stake += float(row.get("stake") or 0)
            gross += float(row.get("gross_return") or 0)
            profit += float(row.get("profit") or 0)

    effective_selected = (
        counts["pending"]
        + counts["won"]
        + counts["lost"]
    )
    settled_effective = counts["won"] + counts["lost"]

    roi = profit / stake if stake > 0 else None

    return {
        "total_rows": len(rows),
        "effective_selected": effective_selected,
        "settled_effective": settled_effective,
        "pending": counts["pending"],
        "won": counts["won"],
        "lost": counts["lost"],
        "void": counts["void"],
        "stake": stake,
        "gross_return": gross,
        "profit": profit,
        "roi_proxy": roi,
    }


def main() -> None:
    client = get_supabase_client()
    if client is None:
        raise SystemExit("Supabase non configuré.")

    experiment = _current_experiment(client)
    if not experiment:
        print("Aucune expérience forward active ou en pause.")
        return

    experiment_id = int(experiment["id"])
    target_bets = int(experiment["target_bets"])

    print("=" * 72)
    print("HORSEPRONO - FORWARD SETTLEMENT")
    print("=" * 72)
    print(f"Expérience : #{experiment_id} {experiment['name']}")

    all_pending = _rows(
        client.table("forward_bets")
        .select("*")
        .eq("experiment_id", experiment_id)
        .eq("status", "pending")
        .order("scheduled_start")
        .execute()
    )

    if not all_pending:
        summary = _summary(client, experiment)
        print("Aucun pari pending.")
        print(summary)

        if (
            summary["settled_effective"] >= target_bets
            and summary["pending"] == 0
        ):
            client.table("forward_experiments").update(
                {
                    "status": "completed",
                    "completed_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                }
            ).eq("id", experiment_id).execute()

            print("Expérience terminée.")
        return

    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(
        minutes=RESULT_GRACE_MINUTES
    )

    due = []
    for bet in all_pending:
        start = pd.to_datetime(
            bet["scheduled_start"],
            utc=True,
            errors="coerce",
        )
        if pd.isna(start):
            continue
        if start.to_pydatetime() <= cutoff:
            due.append(bet)

    if not due:
        print("Aucun pari assez ancien pour être réglé.")
        return

    print(f"Paris à vérifier : {len(due)}")

    race_ids = sorted(
        {int(bet["race_id"]) for bet in due}
    )
    races = _race_rows(client, race_ids)

    settled = 0
    voided = 0

    bets_by_race: dict[int, list[dict]] = {}
    for bet in due:
        bets_by_race.setdefault(
            int(bet["race_id"]),
            [],
        ).append(bet)

    for race_id, bets in bets_by_race.items():
        race = races.get(race_id)
        if not race:
            print(f"Course DB #{race_id} introuvable.")
            continue

        race_date = pd.to_datetime(
            race["race_date"],
            errors="coerce",
        )
        if pd.isna(race_date):
            continue

        target_date = race_date.date()
        reunion = race.get("meeting_number")
        course = race.get("race_number")

        if reunion is None or course is None:
            print(f"Course #{race_id} sans R/C.")
            continue

        reunion = int(reunion)
        course = int(course)

        print()
        print(f"{target_date} R{reunion}C{course}")

        try:
            payload = get_participants(
                target_date,
                reunion,
                course,
            )
        except Exception as exc:
            print(f"  PMU participants indisponible : {exc}")
            continue

        result_df = participants_to_df(
            payload,
            target_date,
            reunion,
            course,
        )

        non_runners = non_runner_numbers(payload)

        if not result_df.empty:
            result_df["is_non_runner"] = (
                pd.to_numeric(
                    result_df["horse_number"],
                    errors="coerce",
                )
                .map(
                    lambda value: (
                        int(value) in non_runners
                        if pd.notna(value)
                        else False
                    )
                )
            )

            try:
                upsert_participants(result_df)
            except Exception as exc:
                print(
                    "  Mise à jour participants ignorée : "
                    f"{exc}"
                )

        positions = _result_map(result_df)
        winner_known = any(
            position == 1
            for position in positions.values()
        )

        print(f"  Winner connu : {winner_known}")

        for bet in bets:
            number = bet.get("horse_number")
            if number is None:
                continue

            number = int(number)
            update = None

            if number in non_runners:
                update = {
                    "status": "void",
                    "finish_position": None,
                    "stake": 0.0,
                    "gross_return": 0.0,
                    "profit": 0.0,
                    "settled_at": now_utc.isoformat(),
                    "settlement_note": "Non-partant explicite PMU.",
                }
                voided += 1

            else:
                position = positions.get(number)

                if position == 1:
                    odds = float(bet["odds_snapshot"])
                    update = {
                        "status": "won",
                        "finish_position": 1,
                        "stake": 1.0,
                        "gross_return": odds,
                        "profit": odds - 1.0,
                        "settled_at": now_utc.isoformat(),
                        "settlement_note": (
                            "Victoire réglée avec la cote "
                            "snapshot pré-course."
                        ),
                    }
                    settled += 1

                elif winner_known:
                    update = {
                        "status": "lost",
                        "finish_position": position,
                        "stake": 1.0,
                        "gross_return": 0.0,
                        "profit": -1.0,
                        "settled_at": now_utc.isoformat(),
                        "settlement_note": (
                            "Perdu : un gagnant PMU "
                            "est connu pour la course."
                        ),
                    }
                    settled += 1

            if update is None:
                continue

            client.table("forward_bets").update(
                update
            ).eq("id", int(bet["id"])).execute()

    summary = _summary(client, experiment)

    print()
    print("=" * 72)
    print(f"Réglés ce run : {settled}")
    print(f"Voids ce run : {voided}")
    print(
        f"Effectifs : "
        f"{summary['effective_selected']}/{target_bets}"
    )
    print(
        f"Settled : "
        f"{summary['settled_effective']}/{target_bets}"
    )
    print(
        "W/L/V/P : "
        f"{summary['won']}/"
        f"{summary['lost']}/"
        f"{summary['void']}/"
        f"{summary['pending']}"
    )

    if summary["roi_proxy"] is not None:
        print(
            "ROI forward proxy : "
            f"{summary['roi_proxy']:+.2%}"
        )

    if (
        summary["settled_effective"] >= target_bets
        and summary["pending"] == 0
    ):
        client.table("forward_experiments").update(
            {
                "status": "completed",
                "completed_at": now_utc.isoformat(),
            }
        ).eq("id", experiment_id).execute()

        print(
            "✅ Objectif atteint : "
            "expérience forward terminée."
        )
    else:
        print("Expérience toujours active.")

    print("=" * 72)


if __name__ == "__main__":
    main()
