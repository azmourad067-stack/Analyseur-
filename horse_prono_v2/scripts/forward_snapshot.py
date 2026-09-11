from __future__ import annotations

import os
import re
import sys

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd


# ============================================================
# PATHS
# ============================================================

HORSE_PRONO_ROOT = os.path.dirname(
    os.path.dirname(__file__)
)

PROJECT_ROOT = os.path.dirname(
    HORSE_PRONO_ROOT
)

sys.path.insert(
    0,
    HORSE_PRONO_ROOT,
)

sys.path.insert(
    0,
    PROJECT_ROOT,
)


# ============================================================
# IMPORTS HORSEPRONO #8
# ============================================================

from app_core.data import normalize_race_input

from app_core.db import (
    get_history_as_of,
    get_supabase_client,
    upsert_participants,
    upsert_races,
)

from app_core.features import (
    entity_snapshot_from_history,
)

from app_core.forward_utils import (
    course_objects,
    course_start_time,
    non_runner_numbers,
)

from app_core.model import HorseRacingModel

from app_core.pmu import (
    get_participants,
    get_programme,
    participants_to_df,
    programme_choices,
)


# ============================================================
# IMPORTS NEURAL #9
# ============================================================

from neural.forward_capture import (
    capture_forward_comparison,
    load_forward_neural,
)


PARIS_TZ = ZoneInfo(
    "Europe/Paris"
)


# ============================================================
# OUTILS
# ============================================================

def _rows(response) -> list[dict]:

    return list(
        getattr(
            response,
            "data",
            None,
        )
        or []
    )


def _norm_text(value) -> str:

    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value)
        .strip()
        .lower(),
    )


# ============================================================
# EXPERIENCE FORWARD ACTIVE
# ============================================================

def _active_experiment(
    client,
) -> dict | None:

    rows = _rows(
        client
        .table(
            "forward_experiments"
        )
        .select("*")
        .eq(
            "status",
            "active",
        )
        .order(
            "started_at"
        )
        .limit(1)
        .execute()
    )

    return (
        rows[0]
        if rows
        else None
    )


# ============================================================
# MODELE #8 FIGE
# ============================================================

def _model_record(
    client,
    experiment: dict,
) -> dict:

    model_id = int(
        experiment[
            "model_version_id"
        ]
    )

    rows = _rows(
        client
        .table(
            "model_versions"
        )
        .select("*")
        .eq(
            "id",
            model_id,
        )
        .limit(1)
        .execute()
    )

    if not rows:

        raise RuntimeError(
            f"Modèle #{model_id} "
            "introuvable."
        )

    record = rows[0]

    expected_hash = str(
        experiment[
            "artifact_hash"
        ]
    )

    actual_hash = str(
        record.get(
            "artifact_hash"
        )
        or ""
    )

    if (
        actual_hash
        != expected_hash
    ):

        raise RuntimeError(
            "Artifact hash différent "
            "du modèle gelé : "
            f"{actual_hash} "
            f"!= {expected_hash}"
        )

    return record


# ============================================================
# COMPTEUR FORWARD #8
# ============================================================

def _effective_count(
    client,
    experiment_id: int,
) -> int:

    rows = _rows(
        client
        .table(
            "forward_bets"
        )
        .select(
            "id,status"
        )
        .eq(
            "experiment_id",
            experiment_id,
        )
        .neq(
            "status",
            "void",
        )
        .execute()
    )

    return len(rows)


def _existing_participant_ids(
    client,
    experiment_id: int,
) -> set[int]:

    rows = _rows(
        client
        .table(
            "forward_bets"
        )
        .select(
            "participant_id"
        )
        .eq(
            "experiment_id",
            experiment_id,
        )
        .execute()
    )

    return {
        int(
            row[
                "participant_id"
            ]
        )
        for row in rows
        if (
            row.get(
                "participant_id"
            )
            is not None
        )
    }


# ============================================================
# METADONNEES COURSE
# ============================================================

def _apply_choice_metadata(
    race: pd.DataFrame,
    choice: dict,
) -> pd.DataFrame:

    out = race.copy()

    metadata = {

        "discipline":
            choice.get(
                "discipline"
            ),

        "hippodrome":
            choice.get(
                "hippodrome"
            ),

        "distance":
            choice.get(
                "distance"
            ),

        "terrain":
            choice.get(
                "terrain"
            ),

        "field_size":
            choice.get(
                "field_size"
            ),
    }

    for (
        column,
        value,
    ) in metadata.items():

        if value is not None:

            out[
                column
            ] = value

    if (
        "field_size"
        not in out.columns
        or
        pd.to_numeric(
            out[
                "field_size"
            ],
            errors="coerce",
        )
        .isna()
        .all()
    ):

        out[
            "field_size"
        ] = len(out)

    return out


# ============================================================
# SNAPSHOTS HISTORIQUES #8
# ============================================================

def _enrich_from_snapshot(
    race: pd.DataFrame,
    snapshots: dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:

    out = normalize_race_input(
        race
    )

    for (
        entity,
        column,
        prefix,
    ) in [

        (
            "horse",
            "horse_name",
            "horse",
        ),

        (
            "jockey",
            "jockey",
            "jockey",
        ),

        (
            "trainer",
            "trainer",
            "trainer",
        ),
    ]:

        snap = snapshots.get(
            entity
        )

        if (
            snap is None
            or snap.empty
        ):
            continue

        keys = (
            out[column]
            .map(
                _norm_text
            )
        )

        tmp = snap.set_index(
            f"{prefix}_key"
        )

        for metric in [

            "win_rate",
            "place_rate",
            "starts_prior",

        ]:

            target = (
                f"{prefix}_"
                f"{metric}"
            )

            mapped = keys.map(
                tmp[target]
            )

            if (
                target
                in out.columns
            ):

                out[
                    target
                ] = (
                    mapped
                    .fillna(
                        out[
                            target
                        ]
                    )
                )

            else:

                out[
                    target
                ] = mapped

    return out


# ============================================================
# SAUVEGARDE COURSE LIVE
# ============================================================

def _persist_live_race(
    client,
    race: pd.DataFrame,
) -> tuple[
    int,
    dict[int, dict],
]:

    if race.empty:

        raise ValueError(
            "Course vide."
        )

    races_df = (
        race[
            [
                "race_id",
                "race_date",
                "discipline",
                "hippodrome",
                "distance",
                "terrain",
                "field_size",
                "reunion",
                "course_number",
            ]
        ]
        .drop_duplicates(
            "race_id"
        )
        .copy()
    )

    races_df[
        "status"
    ] = "scheduled"

    upsert_races(
        races_df
    )

    upsert_participants(
        race
    )

    external_id = str(
        race[
            "race_id"
        ].iloc[0]
    )

    race_rows = _rows(
        client
        .table(
            "races"
        )
        .select(
            "id,external_id"
        )
        .eq(
            "external_id",
            external_id,
        )
        .limit(1)
        .execute()
    )

    if not race_rows:

        raise RuntimeError(
            "Course Supabase "
            "introuvable : "
            f"{external_id}"
        )

    internal_race_id = int(
        race_rows[0][
            "id"
        ]
    )

    participant_rows = _rows(
        client
        .table(
            "participants"
        )
        .select(
            "id,"
            "horse_number,"
            "horse_name"
        )
        .eq(
            "race_id",
            internal_race_id,
        )
        .execute()
    )

    mapping: dict[
        int,
        dict,
    ] = {}

    for row in participant_rows:

        number = row.get(
            "horse_number"
        )

        if number is not None:

            mapping[
                int(number)
            ] = row

    return (
        internal_race_id,
        mapping,
    )


# ============================================================
# COURSES DANS LA FENETRE 10-30 MIN
# ============================================================

def _candidate_courses(
    programme,
    target_date: date,
    now_utc: datetime,
    min_minutes: int,
    max_minutes: int,
) -> list[
    tuple[
        datetime,
        dict,
    ]
]:

    choices = (
        programme_choices(
            programme
        )
    )

    raw_courses = (
        course_objects(
            programme
        )
    )

    candidates = []

    for choice in choices:

        key = (
            int(
                choice[
                    "reunion"
                ]
            ),
            int(
                choice[
                    "course"
                ]
            ),
        )

        start = course_start_time(
            raw_courses.get(
                key
            ),
            target_date,
        )

        if start is None:
            continue

        minutes_before = (
            start
            - now_utc
        ).total_seconds() / 60.0

        if (
            min_minutes
            <= minutes_before
            <= max_minutes
        ):

            candidates.append(
                (
                    start,
                    choice,
                )
            )

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1][
                "reunion"
            ],
            item[1][
                "course"
            ],
        )
    )

    return candidates


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    client = (
        get_supabase_client()
    )

    if client is None:

        raise SystemExit(
            "Supabase non configuré."
        )

    # --------------------------------------------------------
    # EXPERIENCE #8
    # --------------------------------------------------------

    experiment = (
        _active_experiment(
            client
        )
    )

    if not experiment:

        print(
            "Aucune expérience "
            "forward active."
        )

        return

    experiment_id = int(
        experiment[
            "id"
        ]
    )

    target_bets = int(
        experiment[
            "target_bets"
        ]
    )

    current_effective = (
        _effective_count(
            client,
            experiment_id,
        )
    )

    remaining = max(
        0,
        target_bets
        - current_effective,
    )

    print(
        "=" * 72
    )

    print(
        "HORSEPRONO - "
        "FORWARD SNAPSHOT"
    )

    print(
        "=" * 72
    )

    print(
        "Expérience : "
        f"#{experiment_id} "
        f"{experiment['name']}"
    )

    print(
        "Modèle gelé : "
        f"#{experiment['model_version_id']}"
    )

    print(
        "Compteur effectif : "
        f"{current_effective}/"
        f"{target_bets}"
    )

    if remaining <= 0:

        print(
            "Objectif déjà rempli "
            "en paris non void. "
            "Aucun nouveau snapshot."
        )

        return

    # --------------------------------------------------------
    # CHARGEMENT MODELE #8
    # --------------------------------------------------------

    record = _model_record(
        client,
        experiment,
    )

    model = (
        HorseRacingModel
        .from_stored_record(
            record
        )
    )

    # --------------------------------------------------------
    # CONFIG FORWARD
    # --------------------------------------------------------

    threshold = float(
        experiment[
            "edge_threshold"
        ]
    )

    min_minutes = int(
        experiment[
            "capture_min_minutes"
        ]
    )

    max_minutes = int(
        experiment[
            "capture_max_minutes"
        ]
    )

    now_utc = datetime.now(
        timezone.utc
    )

    target_date = (
        now_utc
        .astimezone(
            PARIS_TZ
        )
        .date()
    )

    print(
        f"Date PMU : "
        f"{target_date}"
    )

    print(
        "Fenêtre de capture : "
        f"{min_minutes} à "
        f"{max_minutes} min "
        "avant départ."
    )

    # --------------------------------------------------------
    # PROGRAMME PMU
    # --------------------------------------------------------

    programme = get_programme(
        target_date
    )

    all_choices = (
        programme_choices(
            programme
        )
    )

    raw_courses = (
        course_objects(
            programme
        )
    )

    parsed_starts = 0

    for choice in all_choices:

        key = (
            int(
                choice[
                    "reunion"
                ]
            ),
            int(
                choice[
                    "course"
                ]
            ),
        )

        if (
            course_start_time(
                raw_courses.get(
                    key
                ),
                target_date,
            )
            is not None
        ):

            parsed_starts += 1

    print(
        "Programme PMU : "
        f"{len(all_choices)} "
        "courses, "
        f"{parsed_starts} "
        "horaires interprétés."
    )

    if (
        all_choices
        and parsed_starts == 0
    ):

        raise RuntimeError(
            "Aucun horaire PMU "
            "n'a pu être interprété. "
            "Le forward test reste "
            "protégé : aucun pari "
            "n'est enregistré."
        )

    # --------------------------------------------------------
    # COURSES CANDIDATES
    # --------------------------------------------------------

    candidates = (
        _candidate_courses(
            programme,
            target_date,
            now_utc,
            min_minutes,
            max_minutes,
        )
    )

    if not candidates:

        print(
            "Aucune course dans "
            "la fenêtre de capture."
        )

        return

    print(
        "Courses candidates : "
        f"{len(candidates)}"
    )

    # --------------------------------------------------------
    # CHARGEMENT NEURAL #9
    # Seulement si une course est candidate
    # --------------------------------------------------------

    try:

        neural_bundle = (
            load_forward_neural(
                client
            )
        )

        print(
            "✅ Neural #9 "
            "chargé correctement."
        )

    except Exception as exc:

        neural_bundle = None

        print(
            "⚠️ Neural #9 "
            "indisponible : "
            f"{exc}"
        )

        print(
            "Le forward #8 "
            "continue normalement."
        )

    # --------------------------------------------------------
    # HISTORIQUE STRICTEMENT ANTERIEUR
    # --------------------------------------------------------

    print(
        "Chargement historique "
        "strictement antérieur "
        "à la date..."
    )

    history = (
        get_history_as_of(
            target_date
        )
    )

    print(
        "Historique chargé : "
        f"{len(history)} lignes"
    )

    snapshots = (
        entity_snapshot_from_history(
            history,
            target_date,
        )
    )

    existing_ids = (
        _existing_participant_ids(
            client,
            experiment_id,
        )
    )

    inserts: list[
        dict
    ] = []

    # ========================================================
    # BOUCLE COURSES
    # ========================================================

    for (
        start_time,
        choice,
    ) in candidates:

        if remaining <= 0:
            break

        reunion = int(
            choice[
                "reunion"
            ]
        )

        course = int(
            choice[
                "course"
            ]
        )

        print()

        print(
            f"R{reunion}C{course} "
            f"@ "
            f"{start_time.isoformat()}"
        )

        # ----------------------------------------------------
        # PARTICIPANTS PMU
        # ----------------------------------------------------

        payload = get_participants(
            target_date,
            reunion,
            course,
        )

        race = participants_to_df(
            payload,
            target_date,
            reunion,
            course,
        )

        # ----------------------------------------------------
        # NON-PARTANTS
        # ----------------------------------------------------

        non_runners = (
            non_runner_numbers(
                payload
            )
        )

        if non_runners:

            print(
                "  Non-partants : "
                + ", ".join(
                    str(x)
                    for x
                    in sorted(
                        non_runners
                    )
                )
            )

        if race.empty:

            print(
                "  Aucun participant."
            )

            continue

        # ----------------------------------------------------
        # METADONNEES
        # ----------------------------------------------------

        race = (
            _apply_choice_metadata(
                race,
                choice,
            )
        )

        # ----------------------------------------------------
        # SAUVEGARDE COURSE LIVE
        # ----------------------------------------------------

        (
            internal_race_id,
            participant_map,
        ) = _persist_live_race(
            client,
            race,
        )

        # ----------------------------------------------------
        # MODELE #8
        # ----------------------------------------------------

        enriched = (
            _enrich_from_snapshot(
                race,
                snapshots,
            )
        )

        predictions = (
            model.predict(
                enriched
            )
        )

        # ====================================================
        # DUEL FORWARD #8 VS NEURAL #9
        # ====================================================

        if (
            neural_bundle
            is not None
        ):

            try:

                duel_result = (
                    capture_forward_comparison(
                        client=client,
                        experiment=experiment,
                        race=race,
                        internal_race_id=(
                            internal_race_id
                        ),
                        participant_map=(
                            participant_map
                        ),
                        classical_predictions=(
                            predictions
                        ),
                        scheduled_start=(
                            start_time
                        ),
                        captured_at=(
                            datetime.now(
                                timezone.utc
                            )
                        ),
                        neural_bundle=(
                            neural_bundle
                        ),
                        non_runner_numbers=(
                            non_runners
                        ),
                    )
                )

                print(
                    "  Duel enregistré : "
                    f"{duel_result['written']} "
                    "lignes"
                )

            except Exception as exc:

                print(
                    "  ⚠️ Capture duel "
                    "ignorée : "
                    f"{exc}"
                )

        # ====================================================
        # LOGIQUE PARIS #8
        # INCHANGEE
        # ====================================================

        predictions[
            "edge"
        ] = (

            pd.to_numeric(
                predictions[
                    "win_probability"
                ],
                errors="coerce",
            )

            -

            pd.to_numeric(
                predictions[
                    "market_probability"
                ],
                errors="coerce",
            )
        )

        predictions[
            "odds_num"
        ] = pd.to_numeric(
            predictions[
                "odds"
            ],
            errors="coerce",
        )

        eligible = predictions[
            predictions[
                "edge"
            ].ge(
                threshold
            )
            &
            predictions[
                "odds_num"
            ].gt(
                1.0
            )
        ].copy()

        eligible = (
            eligible
            .sort_values(
                [
                    "rank",
                    "horse_number",
                ],
                ascending=True,
            )
        )

        print(
            "  Edge >= "
            f"{threshold:.2%} : "
            f"{len(eligible)} "
            "chevaux"
        )

        # ----------------------------------------------------
        # PARIS ELIGIBLES #8
        # ----------------------------------------------------

        for _, row in (
            eligible
            .iterrows()
        ):

            if remaining <= 0:
                break

            horse_number_raw = (
                pd.to_numeric(
                    pd.Series(
                        [
                            row.get(
                                "horse_number"
                            )
                        ]
                    ),
                    errors="coerce",
                )
                .iloc[0]
            )

            if pd.isna(
                horse_number_raw
            ):
                continue

            horse_number = int(
                horse_number_raw
            )

            # Ignore explicitement
            # les non-partants.
            if (
                horse_number
                in non_runners
            ):
                continue

            participant = (
                participant_map.get(
                    horse_number
                )
            )

            if not participant:
                continue

            participant_id = int(
                participant[
                    "id"
                ]
            )

            if (
                participant_id
                in existing_ids
            ):
                continue

            odds_snapshot = float(
                row[
                    "odds_num"
                ]
            )

            model_probability = (
                float(
                    row[
                        "win_probability"
                    ]
                )
            )

            market_probability = (
                float(
                    row[
                        "market_probability"
                    ]
                )
            )

            edge = float(
                row[
                    "edge"
                ]
            )

            insert_row = {

                "experiment_id":
                    experiment_id,

                "race_id":
                    internal_race_id,

                "participant_id":
                    participant_id,

                "model_version_id":
                    int(
                        experiment[
                            "model_version_id"
                        ]
                    ),

                "artifact_hash":
                    str(
                        experiment[
                            "artifact_hash"
                        ]
                    ),

                "scheduled_start":
                    start_time
                    .isoformat(),

                "discipline":
                    str(
                        row.get(
                            "discipline"
                        )
                        or "INCONNU"
                    ),

                "horse_name":
                    str(
                        row.get(
                            "horse_name"
                        )
                        or participant.get(
                            "horse_name"
                        )
                        or "INCONNU"
                    ),

                "horse_number":
                    horse_number,

                "odds_snapshot":
                    odds_snapshot,

                "market_probability":
                    market_probability,

                "model_probability":
                    model_probability,

                "place_probability":
                    float(
                        row[
                            "place_probability"
                        ]
                    ),

                "edge":
                    edge,

                "model_rank":
                    int(
                        row[
                            "rank"
                        ]
                    ),

                "stake":
                    1.0,

                "status":
                    "pending",
            }

            inserts.append(
                insert_row
            )

            existing_ids.add(
                participant_id
            )

            remaining -= 1

            print(
                "  + "
                f"N°{horse_number} "
                f"{insert_row['horse_name']} "
                "| "
                f"cote "
                f"{odds_snapshot:.2f} "
                "| "
                f"P "
                f"{model_probability:.3f} "
                "| "
                f"marché "
                f"{market_probability:.3f} "
                "| "
                f"edge "
                f"{edge:+.3f}"
            )

    # ========================================================
    # INSERT PARIS #8
    # ========================================================

    if inserts:

        response = (
            client
            .table(
                "forward_bets"
            )
            .insert(
                inserts
            )
            .execute()
        )

        written = (
            len(
                _rows(
                    response
                )
            )
            or len(
                inserts
            )
        )

    else:

        written = 0

    final_effective = (
        _effective_count(
            client,
            experiment_id,
        )
    )

    print()

    print(
        "=" * 72
    )

    print(
        "Nouveaux paris "
        "capturés : "
        f"{written}"
    )

    print(
        "Compteur forward : "
        f"{final_effective}/"
        f"{target_bets}"
    )

    print(
        "=" * 72
    )


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    main()
