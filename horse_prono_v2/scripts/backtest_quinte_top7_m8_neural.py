from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

HORSE_PRONO_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

PROJECT_ROOT = HORSE_PRONO_ROOT.parent

sys.path.insert(
    0,
    str(HORSE_PRONO_ROOT),
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# ============================================================
# HORSEPRONO
# ============================================================

from app_core.db import (
    get_history_as_of,
    get_supabase_client,
)

from app_core.features import (
    entity_snapshot_from_history,
)

from app_core.pmu import (
    get_programme,
)


# ============================================================
# NEURAL #9
# ============================================================

from neural.predict import (
    MODEL_PATH,
    load_neural_v1,
    predict_dataframe,
    sha256_file,
)

from neural.backtest_simple_place import (
    build_dataset,
)


# ============================================================
# REUTILISATION BACKTEST QUINTE M8
# ============================================================

from backtest_quinte_simple_place_m8 import (
    START_DATE,
    END_DATE,
    MODEL_HASH,
    daterange,
    load_model_8,
    find_quinte_supports,
    load_race,
    load_participants,
    build_race_input,
    enrich_from_snapshot,
)


# ============================================================
# CONFIG
# ============================================================

NEURAL_HASH = (
    "a2d77a6767e2bfa3258d3fad98173f0bec217c9562a5a6b983651257b8d320b8"
)

NEURAL_TRAIN_END = date(
    2026,
    8,
    9,
)

NEURAL_VALID_END = date(
    2026,
    8,
    24,
)

COMMON_TEST_START = date(
    2026,
    8,
    25,
)

OUTPUT_DIR = (
    HORSE_PRONO_ROOT
    / "artifacts"
    / "quinte_top7_m8_neural"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPERS
# ============================================================

def safe_int(value):

    try:

        if pd.isna(value):
            return None

        return int(value)

    except Exception:
        return None


def neural_period(value):

    d = pd.Timestamp(
        value
    ).date()

    if d <= NEURAL_TRAIN_END:
        return "TRAIN_IN_SAMPLE"

    if d <= NEURAL_VALID_END:
        return "VALIDATION"

    return "TEST_SPENT"


# ============================================================
# PODIUM REEL
# ============================================================

def actual_podium(
    participants: pd.DataFrame,
):

    data = participants.copy()

    if "is_non_runner" in data.columns:

        data = data[
            data[
                "is_non_runner"
            ]
            .fillna(False)
            != True
        ].copy()

    data[
        "finish_num"
    ] = pd.to_numeric(
        data[
            "finish_position"
        ],
        errors="coerce",
    )

    data[
        "horse_num"
    ] = pd.to_numeric(
        data[
            "horse_number"
        ],
        errors="coerce",
    )

    data = data[
        data[
            "finish_num"
        ].notna()
        &
        data[
            "horse_num"
        ].notna()
        &
        data[
            "finish_num"
        ].gt(0)
    ].copy()

    data = data.sort_values(
        [
            "finish_num",
            "horse_num",
        ]
    )

    podium = data.head(
        3
    ).copy()

    if len(podium) < 3:
        return []

    return [
        int(x)
        for x in podium[
            "horse_num"
        ].tolist()
    ]


# ============================================================
# MODEL 8
# PIPELINE FORWARD ACTUEL
# ============================================================

def predict_m8_top7(
    model,
    race: pd.DataFrame,
    history: pd.DataFrame,
    race_date: date,
):

    snapshots = (
        entity_snapshot_from_history(
            history,
            race_date,
        )
    )

    enriched = (
        enrich_from_snapshot(
            race,
            snapshots,
        )
    )

    predictions = (
        model.predict(
            enriched
        )
    )

    predictions[
        "win_probability"
    ] = pd.to_numeric(
        predictions[
            "win_probability"
        ],
        errors="coerce",
    )

    predictions[
        "horse_number"
    ] = pd.to_numeric(
        predictions[
            "horse_number"
        ],
        errors="coerce",
    )

    predictions = (
        predictions
        .dropna(
            subset=[
                "win_probability",
                "horse_number",
            ]
        )
        .sort_values(
            [
                "win_probability",
                "horse_number",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    return [
        int(x)
        for x in predictions[
            "horse_number"
        ]
        .head(7)
        .tolist()
    ]


# ============================================================
# FORMAT COURSE NEURAL
# ============================================================

def build_neural_race(
    race_row: dict,
    participants: pd.DataFrame,
):

    races = pd.DataFrame(
        [
            {
                "race_id":
                    int(
                        race_row[
                            "id"
                        ]
                    ),

                "race_external_id":
                    race_row.get(
                        "external_id"
                    ),

                "race_date":
                    race_row.get(
                        "race_date"
                    ),

                "meeting_number":
                    race_row.get(
                        "meeting_number"
                    ),

                "race_number":
                    race_row.get(
                        "race_number"
                    ),

                "hippodrome":
                    race_row.get(
                        "hippodrome"
                    ),

                "discipline":
                    race_row.get(
                        "discipline"
                    ),

                "distance_m":
                    race_row.get(
                        "distance_m"
                    ),

                "terrain":
                    race_row.get(
                        "terrain"
                    ),

                "field_size":
                    race_row.get(
                        "field_size"
                    ),

                "status":
                    race_row.get(
                        "status"
                    ),
            }
        ]
    )

    runners = participants.copy()

    if (
        "id" in runners.columns
        and
        "participant_id"
        not in runners.columns
    ):

        runners = runners.rename(
            columns={
                "id":
                    "participant_id"
            }
        )

    return build_dataset(
        races,
        runners,
    )


# ============================================================
# NEURAL TOP 7
# ============================================================

def predict_neural_top7(
    bundle,
    data: pd.DataFrame,
):

    if data.empty:
        return []

    predictions = (
        predict_dataframe(
            data,
            bundle,
        )
    )

    predictions[
        "horse_number"
    ] = pd.to_numeric(
        predictions[
            "horse_number"
        ],
        errors="coerce",
    )

    predictions = predictions[
        predictions[
            "horse_number"
        ].notna()
    ].copy()

    predictions = (
        predictions
        .sort_values(
            [
                "neural_rank",
                "horse_number",
            ]
        )
    )

    return [
        int(x)
        for x in predictions[
            "horse_number"
        ]
        .head(7)
        .tolist()
    ]


# ============================================================
# SCORE TOP7 VS PODIUM
# ============================================================

def podium_hits(
    podium,
    top7,
):

    return len(
        set(
            podium
        )
        &
        set(
            top7
        )
    )


# ============================================================
# RESUME
# ============================================================

def summarize(
    data: pd.DataFrame,
    model_name: str,
    hit_column: str,
    segment: str,
):

    if data.empty:

        return {
            "segment":
                segment,

            "model":
                model_name,

            "races":
                0,
        }

    hits = pd.to_numeric(
        data[
            hit_column
        ],
        errors="coerce",
    )

    races = len(
        data
    )

    total_found = int(
        hits.sum()
    )

    return {
        "segment":
            segment,

        "model":
            model_name,

        "races":
            races,

        "podium_places":
            races * 3,

        "podium_found":
            total_found,

        "podium_recall_pct":
            (
                100.0
                * total_found
                / (
                    races * 3
                )
        ),

        "average_found_out_of_3":
            float(
                hits.mean()
            ),

        "0_of_3":
            int(
                hits.eq(
                    0
                ).sum()
            ),

        "1_of_3":
            int(
                hits.eq(
                    1
                ).sum()
            ),

        "2_of_3":
            int(
                hits.eq(
                    2
                ).sum()
            ),

        "3_of_3":
            int(
                hits.eq(
                    3
                ).sum()
            ),

        "all_3_in_top7_pct":
            (
                100.0
                * hits.eq(
                    3
                ).mean()
            ),

        "at_least_2_in_top7_pct":
            (
                100.0
                * hits.ge(
                    2
                ).mean()
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 78
    )

    print(
        "HORSEPRONO - QUINTE+"
    )

    print(
        "PODIUM REEL DANS TOP 7"
    )

    print(
        "MODEL #8 vs NEURAL #9"
    )

    print(
        "=" * 78
    )

    # --------------------------------------------------------
    # SUPABASE
    # --------------------------------------------------------

    client = (
        get_supabase_client()
    )

    if client is None:

        raise RuntimeError(
            "Supabase non configuré."
        )

    # --------------------------------------------------------
    # M8
    # --------------------------------------------------------

    model8 = (
        load_model_8(
            client
        )
    )

    print()

    print(
        f"✅ Model 8 : "
        f"{MODEL_HASH}"
    )

    # --------------------------------------------------------
    # NEURAL
    # --------------------------------------------------------

    neural_hash = (
        sha256_file(
            MODEL_PATH
        )
    )

    if neural_hash != NEURAL_HASH:

        raise RuntimeError(
            "Le Neural chargé "
            "n'est pas le Neural #9 gelé."
        )

    neural_bundle = (
        load_neural_v1()
    )

    print(
        f"✅ Neural 9 : "
        f"{neural_hash}"
    )

    # --------------------------------------------------------
    # HISTORIQUE M8
    # --------------------------------------------------------

    print()

    print(
        "Chargement historique..."
    )

    history = (
        get_history_as_of(
            END_DATE
            + timedelta(
                days=1
            )
        )
    )

    print(
        f"Historique : "
        f"{len(history)} lignes"
    )

    results = []

    detected = 0
    valid = 0

    # --------------------------------------------------------
    # JOUR PAR JOUR
    # --------------------------------------------------------

    for current_date in daterange(
        START_DATE,
        END_DATE,
    ):

        try:

            programme = (
                get_programme(
                    current_date
                )
            )

        except Exception as exc:

            print(
                f"{current_date} "
                f"⚠️ programme : "
                f"{exc}"
            )

            continue

        supports = (
            find_quinte_supports(
                programme
            )
        )

        for support in supports:

            detected += 1

            reunion = int(
                support[
                    "meeting_number"
                ]
            )

            course = int(
                support[
                    "race_number"
                ]
            )

            race_row = load_race(
                client,
                current_date,
                reunion,
                course,
            )

            if race_row is None:

                print(
                    f"{current_date} "
                    f"R{reunion}C{course} "
                    "⚠️ absent Supabase"
                )

                continue

            participants = (
                load_participants(
                    client,
                    int(
                        race_row[
                            "id"
                        ]
                    ),
                )
            )

            if participants.empty:
                continue

            podium = (
                actual_podium(
                    participants
                )
            )

            if len(
                podium
            ) != 3:

                print(
                    f"{current_date} "
                    f"R{reunion}C{course} "
                    "⚠️ podium incomplet"
                )

                continue

            # =================================================
            # MODEL 8
            # =================================================

            race_m8 = (
                build_race_input(
                    race_row,
                    participants,
                )
            )

            m8_top7 = (
                predict_m8_top7(
                    model8,
                    race_m8,
                    history,
                    current_date,
                )
            )

            # =================================================
            # NEURAL
            # =================================================

            neural_data = (
                build_neural_race(
                    race_row,
                    participants,
                )
            )

            neural_top7 = (
                predict_neural_top7(
                    neural_bundle,
                    neural_data,
                )
            )

            # =================================================
            # SCORE
            # =================================================

            m8_hits = (
                podium_hits(
                    podium,
                    m8_top7,
                )
            )

            neural_hits = (
                podium_hits(
                    podium,
                    neural_top7,
                )
            )

            valid += 1

            results.append(
                {
                    "race_date":
                        current_date,

                    "meeting_number":
                        reunion,

                    "race_number":
                        course,

                    "hippodrome":
                        race_row.get(
                            "hippodrome"
                        ),

                    "discipline":
                        race_row.get(
                            "discipline"
                        ),

                    "field_size":
                        race_row.get(
                            "field_size"
                        ),

                    "neural_period":
                        neural_period(
                            current_date
                        ),

                    "common_test":
                        (
                            current_date
                            >= COMMON_TEST_START
                        ),

                    "podium":
                        "-".join(
                            map(
                                str,
                                podium,
                            )
                        ),

                    "m8_top7":
                        "-".join(
                            map(
                                str,
                                m8_top7,
                            )
                        ),

                    "m8_podium_hits":
                        m8_hits,

                    "m8_all3":
                        (
                            m8_hits
                            == 3
                        ),

                    "neural_top7":
                        "-".join(
                            map(
                                str,
                                neural_top7,
                            )
                        ),

                    "neural_podium_hits":
                        neural_hits,

                    "neural_all3":
                        (
                            neural_hits
                            == 3
                        ),
                }
            )

            print(
                f"{current_date} "
                f"R{reunion}C{course} | "
                f"M8={m8_hits}/3 | "
                f"Neural={neural_hits}/3"
            )

    # ========================================================
    # DATAFRAME
    # ========================================================

    detail = pd.DataFrame(
        results
    )

    if detail.empty:

        raise RuntimeError(
            "Aucune course exploitable."
        )

    detail[
        "race_date"
    ] = pd.to_datetime(
        detail[
            "race_date"
        ]
    )

    # ========================================================
    # RESUME TOTAL
    # ========================================================

    summaries = []

    for (
        model_name,
        hit_column,
    ) in [
        (
            "MODEL_8",
            "m8_podium_hits",
        ),
        (
            "NEURAL_9",
            "neural_podium_hits",
        ),
    ]:

        summaries.append(
            summarize(
                detail,
                model_name,
                hit_column,
                "ALL_99_DESCRIPTIVE",
            )
        )

    # --------------------------------------------------------
    # FENETRE COMMUNE TEST
    # 25/08 -> 08/09
    # --------------------------------------------------------

    common_test = detail[
        detail[
            "common_test"
        ]
        == True
    ].copy()

    for (
        model_name,
        hit_column,
    ) in [
        (
            "MODEL_8",
            "m8_podium_hits",
        ),
        (
            "NEURAL_9",
            "neural_podium_hits",
        ),
    ]:

        summaries.append(
            summarize(
                common_test,
                model_name,
                hit_column,
                "COMMON_TEST_25AUG_08SEP",
            )
        )

    # --------------------------------------------------------
    # PERIODES NEURAL
    # --------------------------------------------------------

    for period in [
        "TRAIN_IN_SAMPLE",
        "VALIDATION",
        "TEST_SPENT",
    ]:

        subset = detail[
            detail[
                "neural_period"
            ]
            == period
        ].copy()

        for (
            model_name,
            hit_column,
        ) in [
            (
                "MODEL_8",
                "m8_podium_hits",
            ),
            (
                "NEURAL_9",
                "neural_podium_hits",
            ),
        ]:

            summaries.append(
                summarize(
                    subset,
                    model_name,
                    hit_column,
                    period,
                )
            )

    summary = pd.DataFrame(
        summaries
    )

    # ========================================================
    # FACE A FACE
    # ========================================================

    comparison = pd.DataFrame(
        [
            {
                "races":
                    len(
                        detail
                    ),

                "both_3_of_3":
                    int(
                        (
                            detail[
                                "m8_all3"
                            ]
                            &
                            detail[
                                "neural_all3"
                            ]
                        ).sum()
                    ),

                "m8_only_3_of_3":
                    int(
                        (
                            detail[
                                "m8_all3"
                            ]
                            &
                            ~detail[
                                "neural_all3"
                            ]
                        ).sum()
                    ),

                "neural_only_3_of_3":
                    int(
                        (
                            ~detail[
                                "m8_all3"
                            ]
                            &
                            detail[
                                "neural_all3"
                            ]
                        ).sum()
                    ),

                "neither_3_of_3":
                    int(
                        (
                            ~detail[
                                "m8_all3"
                            ]
                            &
                            ~detail[
                                "neural_all3"
                            ]
                        ).sum()
                    ),

                "m8_finds_more_podium_horses":
                    int(
                        (
                            detail[
                                "m8_podium_hits"
                            ]
                            >
                            detail[
                                "neural_podium_hits"
                            ]
                        ).sum()
                    ),

                "neural_finds_more_podium_horses":
                    int(
                        (
                            detail[
                                "neural_podium_hits"
                            ]
                            >
                            detail[
                                "m8_podium_hits"
                            ]
                        ).sum()
                    ),

                "tie":
                    int(
                        (
                            detail[
                                "m8_podium_hits"
                            ]
                            ==
                            detail[
                                "neural_podium_hits"
                            ]
                        ).sum()
                    ),
            }
        ]
    )

    # ========================================================
    # EXPORT
    # ========================================================

    detail.to_csv(
        OUTPUT_DIR
        / "quinte_top7_course_by_course.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        OUTPUT_DIR
        / "quinte_top7_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "quinte_top7_head_to_head.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()

    print(
        "=" * 78
    )

    print(
        "RESULTATS"
    )

    print(
        "=" * 78
    )

    print(
        f"Quinté détectés : "
        f"{detected}"
    )

    print(
        f"Quinté exploitables : "
        f"{valid}"
    )

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "FACE A FACE"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print()

    print(
        f"Fichiers : "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()
