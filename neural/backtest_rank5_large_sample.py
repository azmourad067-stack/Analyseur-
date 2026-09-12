from __future__ import annotations

import argparse
import time

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from neural.predict import (
    MODEL_PATH,
    load_neural_v1,
    predict_dataframe,
    sha256_file,
)

from neural.backtest_simple_place import (
    get_client,
    load_participants,
    build_dataset,
    get_final_reports,
    extract_simple_place_returns,
    max_drawdown,
)


# ============================================================
# HORSEPRONO - RANG 5 SIMPLE PLACE
# GRAND ECHANTILLON
# ============================================================

NEURAL_HASH = (
    "a2d77a6767e2bfa3258d3fad98173f0bec217c9562a5a6b983651257b8d320b8"
)

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "rank5_large_sample"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# PERIODES DU NEURAL #9
# ============================================================

TRAIN_END = date(
    2026,
    8,
    9,
)

VALID_END = date(
    2026,
    8,
    24,
)

TEST_END = date(
    2026,
    9,
    11,
)

FORWARD_START = date(
    2026,
    9,
    12,
)


# ============================================================
# OUTILS
# ============================================================

def safe_float(value):

    try:

        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    except Exception:

        return None


def safe_int(value):

    try:

        if pd.isna(value):
            return None

        return int(value)

    except Exception:

        return None


def rows(response):

    return list(
        getattr(
            response,
            "data",
            None,
        )
        or []
    )


# ============================================================
# CHARGEMENT PAGINE DES COURSES
# ============================================================

def load_races_paginated(
    client,
    start_date: date,
    end_date: date,
):

    all_rows = []

    page_size = 1000
    offset = 0

    while True:

        response = (
            client
            .table(
                "races"
            )
            .select(
                "id,"
                "external_id,"
                "race_date,"
                "meeting_number,"
                "race_number,"
                "hippodrome,"
                "discipline,"
                "distance_m,"
                "terrain,"
                "field_size,"
                "status"
            )
            .gte(
                "race_date",
                start_date.isoformat(),
            )
            .lte(
                "race_date",
                end_date.isoformat(),
            )
            .order(
                "race_date"
            )
            .order(
                "meeting_number"
            )
            .order(
                "race_number"
            )
            .range(
                offset,
                offset + page_size - 1,
            )
            .execute()
        )

        batch = rows(
            response
        )

        if not batch:
            break

        all_rows.extend(
            batch
        )

        print(
            f"Courses chargées : "
            f"{len(all_rows)}"
        )

        if len(batch) < page_size:
            break

        offset += page_size

    races = pd.DataFrame(
        all_rows
    )

    if races.empty:
        return races

    races = races.rename(
        columns={
            "id":
                "race_id",

            "external_id":
                "race_external_id",
        }
    )

    races[
        "race_id"
    ] = pd.to_numeric(
        races[
            "race_id"
        ],
        errors="coerce",
    )

    races = races[
        races[
            "race_id"
        ].notna()
    ].copy()

    races[
        "race_id"
    ] = races[
        "race_id"
    ].astype(
        int
    )

    return races


# ============================================================
# CLASSIFICATION PERIODE
# ============================================================

def period_name(
    value,
):

    d = pd.Timestamp(
        value
    ).date()

    if d <= TRAIN_END:

        return (
            "TRAIN_IN_SAMPLE"
        )

    if d <= VALID_END:

        return (
            "VALIDATION"
        )

    if d <= TEST_END:

        return (
            "TEST_SPENT"
        )

    return (
        "FORWARD"
    )


# ============================================================
# RAPPORT PMU D'UNE COURSE
# ============================================================

def fetch_place_returns(
    race_date: date,
    reunion: int,
    course: int,
):

    last_error = None

    for attempt in range(
        1,
        4,
    ):

        try:

            payload = (
                get_final_reports(
                    race_date,
                    reunion,
                    course,
                )
            )

            (
                place_returns,
                simple_place_found,
            ) = (
                extract_simple_place_returns(
                    payload
                )
            )

            return (
                place_returns,
                simple_place_found,
                None,
            )

        except Exception as exc:

            last_error = exc

            time.sleep(
                0.5
                * attempt
            )

    return (
        {},
        False,
        str(
            last_error
        ),
    )


# ============================================================
# BOOTSTRAP DU ROI
# ============================================================

def bootstrap_roi_ci(
    profits,
    n_boot=2000,
):

    profits = np.asarray(
        profits,
        dtype=float,
    )

    profits = profits[
        np.isfinite(
            profits
        )
    ]

    n = len(
        profits
    )

    if n < 10:

        return (
            np.nan,
            np.nan,
        )

    rng = np.random.default_rng(
        42
    )

    means = np.empty(
        n_boot,
        dtype=float,
    )

    for i in range(
        n_boot
    ):

        sample = rng.choice(
            profits,
            size=n,
            replace=True,
        )

        means[i] = (
            sample.mean()
        )

    low = np.percentile(
        means,
        2.5,
    )

    high = np.percentile(
        means,
        97.5,
    )

    return (
        float(low),
        float(high),
    )


# ============================================================
# RESUME
# ============================================================

def summarize(
    data: pd.DataFrame,
    label: str,
):

    if data.empty:
        return None

    data = (
        data
        .sort_values(
            [
                "race_date",
                "meeting_number",
                "race_number",
            ]
        )
        .copy()
    )

    bets = len(
        data
    )

    placed = int(
        data[
            "is_placed"
        ].sum()
    )

    stake = float(
        bets
    )

    gross = float(
        data[
            "gross_return"
        ].sum()
    )

    profit = float(
        data[
            "profit"
        ].sum()
    )

    roi = (
        profit / stake
        if stake > 0
        else np.nan
    )

    hit_rate = (
        placed / bets
        if bets > 0
        else np.nan
    )

    winner_returns = (
        data.loc[
            data[
                "is_placed"
            ],
            "place_return",
        ]
    )

    break_even_return = (
        1.0 / hit_rate
        if (
            hit_rate
            and hit_rate > 0
        )
        else np.nan
    )

    (
        roi_ci_low,
        roi_ci_high,
    ) = bootstrap_roi_ci(
        data[
            "profit"
        ]
    )

    return {
        "segment":
            label,

        "bets":
            bets,

        "placed":
            placed,

        "hit_rate":
            hit_rate,

        "stake":
            stake,

        "gross_return":
            gross,

        "profit":
            profit,

        "roi":
            roi,

        "roi_ci_low_95":
            roi_ci_low,

        "roi_ci_high_95":
            roi_ci_high,

        "avg_place_return_win":
            (
                winner_returns.mean()
                if not winner_returns.empty
                else np.nan
            ),

        "median_place_return_win":
            (
                winner_returns.median()
                if not winner_returns.empty
                else np.nan
            ),

        "break_even_return":
            break_even_return,

        "avg_odds":
            pd.to_numeric(
                data[
                    "odds"
                ],
                errors="coerce",
            ).mean(),

        "median_odds":
            pd.to_numeric(
                data[
                    "odds"
                ],
                errors="coerce",
            ).median(),

        "max_drawdown":
            max_drawdown(
                data[
                    "profit"
                ]
            ),

        "races":
            data[
                "race_id"
            ].nunique(),

        "start_date":
            data[
                "race_date"
            ].min(),

        "end_date":
            data[
                "race_date"
            ].max(),
    }


# ============================================================
# TABLE PAR PERIODE
# ============================================================

def summary_by_period(
    detail,
):

    records = []

    record = summarize(
        detail,
        "ALL_STOCK",
    )

    if record:
        records.append(
            record
        )

    post_train = detail[
        pd.to_datetime(
            detail[
                "race_date"
            ]
        ).dt.date
        > TRAIN_END
    ]

    record = summarize(
        post_train,
        "POST_TRAIN_2026-08-10+",
    )

    if record:
        records.append(
            record
        )

    for period in [
        "TRAIN_IN_SAMPLE",
        "VALIDATION",
        "TEST_SPENT",
        "FORWARD",
    ]:

        subset = detail[
            detail[
                "period"
            ]
            == period
        ]

        record = summarize(
            subset,
            period,
        )

        if record:
            records.append(
                record
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# TABLE PAR DISCIPLINE
# ============================================================

def summary_by_discipline(
    detail,
    post_train_only=True,
):

    if post_train_only:

        source = detail[
            pd.to_datetime(
                detail[
                    "race_date"
                ]
            ).dt.date
            > TRAIN_END
        ].copy()

    else:

        source = detail.copy()

    records = []

    for discipline in sorted(
        source[
            "discipline"
        ]
        .fillna(
            "INCONNU"
        )
        .astype(
            str
        )
        .unique()
    ):

        subset = source[
            source[
                "discipline"
            ]
            .fillna(
                "INCONNU"
            )
            .astype(
                str
            )
            == discipline
        ]

        record = summarize(
            subset,
            discipline,
        )

        if record:
            records.append(
                record
            )

    result = pd.DataFrame(
        records
    )

    if not result.empty:

        result = result.sort_values(
            [
                "roi",
                "bets",
            ],
            ascending=[
                False,
                False,
            ],
        )

    return result


# ============================================================
# TABLE PAR TAILLE DE PELOTON
# ============================================================

def summary_by_field_size(
    detail,
):

    source = detail[
        pd.to_datetime(
            detail[
                "race_date"
            ]
        ).dt.date
        > TRAIN_END
    ].copy()

    field_size = pd.to_numeric(
        source[
            "active_field_size"
        ],
        errors="coerce",
    )

    groups = {
        "5-7 partants":
            field_size.between(
                5,
                7,
            ),

        "8-11 partants":
            field_size.between(
                8,
                11,
            ),

        "12-15 partants":
            field_size.between(
                12,
                15,
            ),

        "16+ partants":
            (
                field_size
                >= 16
            ),
    }

    records = []

    for name, mask in (
        groups.items()
    ):

        record = summarize(
            source[
                mask
            ],
            name,
        )

        if record:
            records.append(
                record
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# TABLE PAR MOIS
# ============================================================

def summary_by_month(
    detail,
):

    source = detail.copy()

    source[
        "month"
    ] = pd.to_datetime(
        source[
            "race_date"
        ]
    ).dt.to_period(
        "M"
    ).astype(
        str
    )

    records = []

    for month in sorted(
        source[
            "month"
        ].unique()
    ):

        record = summarize(
            source[
                source[
                    "month"
                ]
                == month
            ],
            month,
        )

        if record:
            records.append(
                record
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# MAIN BACKTEST
# ============================================================

def run(
    start_date: date,
    end_date: date,
):

    print()
    print(
        "=" * 78
    )

    print(
        "HORSEPRONO NEURAL #9"
    )

    print(
        "RANG 5 SIMPLE PLACE "
        "- GRAND ECHANTILLON"
    )

    print(
        "=" * 78
    )

    print(
        f"Période demandée : "
        f"{start_date} → {end_date}"
    )

    # --------------------------------------------------------
    # VERIFICATION DU MODELE
    # --------------------------------------------------------

    model_hash = sha256_file(
        MODEL_PATH
    )

    print()
    print(
        f"Hash Neural : "
        f"{model_hash}"
    )

    if model_hash != NEURAL_HASH:

        raise RuntimeError(
            "Le Neural chargé "
            "n'est pas le modèle #9 gelé."
        )

    bundle = load_neural_v1()

    # --------------------------------------------------------
    # SUPABASE
    # --------------------------------------------------------

    client = get_client()

    races = load_races_paginated(
        client,
        start_date,
        end_date,
    )

    if races.empty:

        raise RuntimeError(
            "Aucune course."
        )

    print()
    print(
        f"Courses Supabase : "
        f"{len(races)}"
    )

    participants = (
        load_participants(
            client,
            races[
                "race_id"
            ].tolist(),
        )
    )

    print(
        f"Participants : "
        f"{len(participants)}"
    )

    data = build_dataset(
        races,
        participants,
    )

    print(
        f"Partants actifs : "
        f"{len(data)}"
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    predictions = (
        predict_dataframe(
            data,
            bundle,
        )
    )

    active_sizes = (
        predictions
        .groupby(
            "race_id"
        )
        .size()
        .to_dict()
    )

    rank5 = predictions[
        predictions[
            "neural_rank"
        ]
        == 5
    ].copy()

    rank5[
        "active_field_size"
    ] = rank5[
        "race_id"
    ].map(
        active_sizes
    )

    print()
    print(
        f"Courses avec un Rang 5 : "
        f"{len(rank5)}"
    )

    # --------------------------------------------------------
    # RAPPORTS SIMPLE PLACE
    # --------------------------------------------------------

    result_rows = []

    unavailable = 0
    no_simple_place = 0

    total = len(
        rank5
    )

    for index, (
        _,
        horse,
    ) in enumerate(
        rank5.iterrows(),
        start=1,
    ):

        race_date = (
            pd.Timestamp(
                horse[
                    "race_date"
                ]
            )
            .date()
        )

        reunion = safe_int(
            horse.get(
                "meeting_number"
            )
        )

        course = safe_int(
            horse.get(
                "race_number"
            )
        )

        number = safe_int(
            horse.get(
                "horse_number"
            )
        )

        if (
            reunion is None
            or course is None
            or number is None
        ):
            continue

        (
            place_returns,
            simple_place_found,
            error,
        ) = fetch_place_returns(
            race_date,
            reunion,
            course,
        )

        if error is not None:

            unavailable += 1

            if (
                index % 50 == 0
                or index == total
            ):

                print(
                    f"[{index}/{total}] "
                    f"rapports indisponibles="
                    f"{unavailable}"
                )

            continue

        if (
            not simple_place_found
            or not place_returns
        ):

            no_simple_place += 1
            continue

        payout = (
            place_returns.get(
                number
            )
        )

        is_placed = (
            payout is not None
        )

        gross_return = (
            float(payout)
            if is_placed
            else 0.0
        )

        result_rows.append(
            {
                "race_id":
                    int(
                        horse[
                            "race_id"
                        ]
                    ),

                "race_date":
                    race_date,

                "period":
                    period_name(
                        race_date
                    ),

                "meeting_number":
                    reunion,

                "race_number":
                    course,

                "hippodrome":
                    horse.get(
                        "hippodrome"
                    ),

                "discipline":
                    horse.get(
                        "discipline"
                    ),

                "active_field_size":
                    safe_int(
                        horse.get(
                            "active_field_size"
                        )
                    ),

                "horse_number":
                    number,

                "horse_name":
                    horse.get(
                        "horse_name"
                    ),

                "odds":
                    safe_float(
                        horse.get(
                            "odds"
                        )
                    ),

                "neural_rank":
                    5,

                "neural_top3_probability":
                    float(
                        horse[
                            "neural_top3_probability"
                        ]
                    ),

                "finish_position":
                    safe_int(
                        horse.get(
                            "finish_position"
                        )
                    ),

                "is_placed":
                    is_placed,

                "place_return":
                    (
                        float(payout)
                        if payout is not None
                        else 0.0
                    ),

                "stake":
                    1.0,

                "gross_return":
                    gross_return,

                "profit":
                    (
                        gross_return
                        - 1.0
                    ),
            }
        )

        if (
            index % 100 == 0
            or index == total
        ):

            print(
                f"[{index}/{total}] "
                f"analysées | "
                f"valides="
                f"{len(result_rows)} | "
                f"indisponibles="
                f"{unavailable}"
            )

        time.sleep(
            0.03
        )

    detail = pd.DataFrame(
        result_rows
    )

    if detail.empty:

        raise RuntimeError(
            "Aucun résultat exploitable."
        )

    # --------------------------------------------------------
    # TABLEAUX
    # --------------------------------------------------------

    period_summary = (
        summary_by_period(
            detail
        )
    )

    discipline_summary = (
        summary_by_discipline(
            detail,
            post_train_only=True,
        )
    )

    discipline_all = (
        summary_by_discipline(
            detail,
            post_train_only=False,
        )
    )

    field_summary = (
        summary_by_field_size(
            detail
        )
    )

    month_summary = (
        summary_by_month(
            detail
        )
    )

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    detail.to_csv(
        OUTPUT_DIR
        / "rank5_detail.csv",
        index=False,
    )

    period_summary.to_csv(
        OUTPUT_DIR
        / "rank5_by_period.csv",
        index=False,
    )

    discipline_summary.to_csv(
        OUTPUT_DIR
        / "rank5_by_discipline_post_train.csv",
        index=False,
    )

    discipline_all.to_csv(
        OUTPUT_DIR
        / "rank5_by_discipline_all_stock.csv",
        index=False,
    )

    field_summary.to_csv(
        OUTPUT_DIR
        / "rank5_by_field_size_post_train.csv",
        index=False,
    )

    month_summary.to_csv(
        OUTPUT_DIR
        / "rank5_by_month.csv",
        index=False,
    )

    # --------------------------------------------------------
    # AFFICHAGE
    # --------------------------------------------------------

    print()
    print(
        "=" * 78
    )

    print(
        "RANG 5 - RESULTATS PAR PERIODE"
    )

    print(
        "=" * 78
    )

    print(
        period_summary[
            [
                "segment",
                "bets",
                "placed",
                "hit_rate",
                "profit",
                "roi",
                "roi_ci_low_95",
                "roi_ci_high_95",
                "avg_place_return_win",
                "max_drawdown",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 78
    )

    print(
        "RANG 5 - DISCIPLINES "
        "POST-TRAIN"
    )

    print(
        "=" * 78
    )

    print(
        discipline_summary[
            [
                "segment",
                "bets",
                "placed",
                "hit_rate",
                "profit",
                "roi",
                "roi_ci_low_95",
                "roi_ci_high_95",
                "avg_place_return_win",
                "max_drawdown",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 78
    )

    print(
        "RANG 5 - TAILLE PELOTON "
        "POST-TRAIN"
    )

    print(
        "=" * 78
    )

    print(
        field_summary[
            [
                "segment",
                "bets",
                "placed",
                "hit_rate",
                "profit",
                "roi",
                "roi_ci_low_95",
                "roi_ci_high_95",
                "max_drawdown",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 78
    )

    print(
        "RANG 5 - PAR MOIS"
    )

    print(
        "=" * 78
    )

    print(
        month_summary[
            [
                "segment",
                "bets",
                "placed",
                "hit_rate",
                "profit",
                "roi",
                "max_drawdown",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Rapports indisponibles : "
        f"{unavailable}"
    )

    print(
        f"Courses sans Simple Placé : "
        f"{no_simple_place}"
    )

    print()
    print(
        "Résultats enregistrés dans :"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "=" * 78
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--start",
        type=str,
        default="2026-06-01",
    )

    parser.add_argument(
        "--end",
        type=str,
        default="2026-09-12",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    start_date = (
        pd.Timestamp(
            args.start
        ).date()
    )

    end_date = (
        pd.Timestamp(
            args.end
        ).date()
    )

    run(
        start_date,
        end_date,
    )


if __name__ == "__main__":

    main()
