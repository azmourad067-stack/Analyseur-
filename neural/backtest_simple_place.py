from __future__ import annotations

import argparse
import os
import time

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from supabase import create_client

from neural.predict import (
    MODEL_PATH,
    load_neural_v1,
    predict_dataframe,
    sha256_file,
)


# ============================================================
# HORSEPRONO NEURAL - SIMPLE PLACE BACKTEST
# ============================================================

NEURAL_HASH = (
    "a2d77a6767e2bfa3258d3fad98173f0bec217c9562a5a6b983651257b8d320b8"
)

PMU_BASE_URL = (
    "https://online.turfinfo.api.pmu.fr/"
    "rest/client/1"
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
    / "simple_place"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Toutes les courses à partir du 12/09 sont
# considérées comme validation forward fraîche.
FORWARD_START = date(
    2026,
    9,
    12,
)

REQUEST_TIMEOUT = 20

HEADERS = {
    "User-Agent":
        "HorseProno/1.0",
    "Accept":
        "application/json",
}


# ============================================================
# OUTILS
# ============================================================

def _rows(response):

    return list(
        getattr(
            response,
            "data",
            None,
        )
        or []
    )


def _safe_float(value):

    try:

        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    except Exception:

        return None


def _safe_int(value):

    try:

        if pd.isna(value):
            return None

        return int(value)

    except Exception:

        return None


# ============================================================
# SUPABASE
# ============================================================

def get_client():

    url = os.getenv(
        "SUPABASE_URL"
    )

    key = (
        os.getenv(
            "SUPABASE_SERVICE_KEY"
        )
        or os.getenv(
            "SUPABASE_KEY"
        )
    )

    if not url or not key:

        raise RuntimeError(
            "SUPABASE_URL / "
            "SUPABASE_SERVICE_KEY absents."
        )

    return create_client(
        url,
        key,
    )


# ============================================================
# COURSES
# ============================================================

def load_races(
    client,
    start_date: date,
    end_date: date,
):

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
        .execute()
    )

    races = pd.DataFrame(
        _rows(response)
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
        races["race_id"],
        errors="coerce",
    )

    races = races[
        races["race_id"].notna()
    ].copy()

    races[
        "race_id"
    ] = races[
        "race_id"
    ].astype(int)

    return races


# ============================================================
# PARTICIPANTS
# ============================================================

def load_participants(
    client,
    race_ids: list[int],
):

    if not race_ids:
        return pd.DataFrame()

    rows = []

    # On limite les lots pour éviter
    # la limite Supabase de résultats.
    chunk_size = 40

    for start in range(
        0,
        len(race_ids),
        chunk_size,
    ):

        chunk = race_ids[
            start:
            start + chunk_size
        ]

        response = (
            client
            .table(
                "participants"
            )
            .select(
                "id,"
                "race_id,"
                "horse_name,"
                "horse_number,"
                "jockey_name,"
                "trainer_name,"
                "odds,"
                "weight_kg,"
                "draw,"
                "age,"
                "sex,"
                "recent_form,"
                "finish_position,"
                "is_non_runner"
            )
            .in_(
                "race_id",
                chunk,
            )
            .execute()
        )

        rows.extend(
            _rows(response)
        )

    participants = pd.DataFrame(
        rows
    )

    if participants.empty:
        return participants

    participants = (
        participants.rename(
            columns={
                "id":
                    "participant_id",
            }
        )
    )

    return participants


# ============================================================
# DATASET NEURAL
# ============================================================

def build_dataset(
    races: pd.DataFrame,
    participants: pd.DataFrame,
):

    if (
        races.empty
        or participants.empty
    ):

        return pd.DataFrame()

    data = participants.merge(
        races,
        on="race_id",
        how="inner",
    )

    # --------------------------------------------------------
    # PARTANTS UNIQUEMENT
    # --------------------------------------------------------

    if (
        "is_non_runner"
        in data.columns
    ):

        data = data[
            data[
                "is_non_runner"
            ].fillna(False)
            != True
        ].copy()

    data[
        "horse_number"
    ] = pd.to_numeric(
        data["horse_number"],
        errors="coerce",
    )

    data = data[
        data[
            "horse_number"
        ].notna()
    ].copy()

    data[
        "horse_number"
    ] = data[
        "horse_number"
    ].astype(int)

    data[
        "race_date"
    ] = pd.to_datetime(
        data["race_date"],
        errors="coerce",
    )

    data = data[
        data["race_date"].notna()
    ].copy()

    return data


# ============================================================
# RAPPORTS PMU
# ============================================================

def get_final_reports(
    race_date: date,
    reunion: int,
    course: int,
):

    date_text = race_date.strftime(
        "%d%m%Y"
    )

    url = (
        f"{PMU_BASE_URL}"
        f"/programme/"
        f"{date_text}"
        f"/R{reunion}"
        f"/C{course}"
        f"/rapports-definitifs"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        params={
            "combinaisonEnTableau":
                "true",

            "specialisation":
                "INTERNET",
        },
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# PARCOURS JSON RAPPORTS
# ============================================================

def iter_bet_blocks(obj):

    if isinstance(
        obj,
        dict,
    ):

        if (
            "typePari"
            in obj
            and "rapports"
            in obj
        ):

            yield obj

        for value in obj.values():

            yield from iter_bet_blocks(
                value
            )

    elif isinstance(
        obj,
        list,
    ):

        for value in obj:

            yield from iter_bet_blocks(
                value
            )


# ============================================================
# NUMEROS DE CHEVAUX D'UNE COMBINAISON
# ============================================================

def combination_numbers(
    value,
):

    result = []

    if value is None:
        return result

    if isinstance(
        value,
        (int, np.integer),
    ):

        return [
            int(value)
        ]

    if isinstance(
        value,
        float,
    ):

        if value.is_integer():

            return [
                int(value)
            ]

        return result

    if isinstance(
        value,
        str,
    ):

        import re

        numbers = re.findall(
            r"\d+",
            value,
        )

        return [
            int(x)
            for x in numbers
        ]

    if isinstance(
        value,
        list,
    ):

        for item in value:

            result.extend(
                combination_numbers(
                    item
                )
            )

        return result

    if isinstance(
        value,
        dict,
    ):

        preferred_keys = [
            "numPmu",
            "numero",
            "numParticipant",
            "cheval",
        ]

        for key in preferred_keys:

            if key in value:

                numbers = (
                    combination_numbers(
                        value[key]
                    )
                )

                if numbers:
                    return numbers

        for item in value.values():

            result.extend(
                combination_numbers(
                    item
                )
            )

    return result


# ============================================================
# CALCUL RETOUR PAR EURO
# ============================================================

def return_per_euro(
    bet_block: dict,
    report: dict,
):

    # --------------------------------------------------------
    # Méthode la plus fiable :
    #
    # dividendePourUneMiseDeBase / miseBase
    #
    # Si les deux sont en centimes :
    # 350 / 100 = 3.50 €
    #
    # Si les deux sont déjà en euros :
    # 3.50 / 1 = 3.50 €
    # --------------------------------------------------------

    base_stake = _safe_float(
        bet_block.get(
            "miseBase"
        )
    )

    dividend_base = _safe_float(
        report.get(
            "dividendePourUneMiseDeBase"
        )
    )

    if (
        base_stake is not None
        and base_stake > 0
        and dividend_base is not None
        and dividend_base >= 0
    ):

        ratio = (
            dividend_base
            / base_stake
        )

        if (
            0 < ratio < 1000
        ):

            return float(
                ratio
            )

    # --------------------------------------------------------
    # Fallback : dividendePourUnEuro
    # --------------------------------------------------------

    per_euro = _safe_float(
        report.get(
            "dividendePourUnEuro"
        )
    )

    if (
        per_euro is not None
        and per_euro > 0
    ):

        # L'API a historiquement utilisé
        # des montants en centimes selon
        # certaines versions.
        #
        # On ne divise que les valeurs
        # manifestement incompatibles
        # avec un rapport standard.
        if per_euro > 100:

            per_euro = (
                per_euro
                / 100.0
            )

        return float(
            per_euro
        )

    # --------------------------------------------------------
    # Dernier fallback
    # --------------------------------------------------------

    dividend = _safe_float(
        report.get(
            "dividende"
        )
    )

    if (
        dividend is not None
        and dividend > 0
    ):

        if dividend > 100:

            dividend = (
                dividend
                / 100.0
            )

        return float(
            dividend
        )

    return None


# ============================================================
# EXTRACTION SIMPLE PLACE
# ============================================================

def extract_simple_place_returns(
    payload,
):

    returns = {}

    simple_place_found = False

    for block in iter_bet_blocks(
        payload
    ):

        bet_type = str(
            block.get(
                "typePari"
            )
            or ""
        ).upper()

        if (
            "SIMPLE"
            not in bet_type
            or "PLACE"
            not in bet_type
        ):

            continue

        # On exclut éventuellement
        # des variantes étrangères.
        if (
            "COUPLE"
            in bet_type
            or "TRIO"
            in bet_type
        ):
            continue

        simple_place_found = True

        reports = (
            block.get(
                "rapports"
            )
            or []
        )

        if not isinstance(
            reports,
            list,
        ):
            continue

        for report in reports:

            if not isinstance(
                report,
                dict,
            ):
                continue

            combination = (
                report.get(
                    "combinaison"
                )
            )

            numbers = (
                combination_numbers(
                    combination
                )
            )

            numbers = list(
                dict.fromkeys(
                    numbers
                )
            )

            # Simple placé =
            # une combinaison d'un seul cheval.
            if len(numbers) != 1:
                continue

            number = int(
                numbers[0]
            )

            payout = return_per_euro(
                block,
                report,
            )

            if (
                payout is None
                or payout <= 0
            ):
                continue

            returns[number] = payout

    return (
        returns,
        simple_place_found,
    )


# ============================================================
# MAX DRAWDOWN
# ============================================================

def max_drawdown(
    profits: pd.Series,
):

    if profits.empty:
        return 0.0

    cumulative = (
        profits
        .fillna(0)
        .cumsum()
    )

    peak = (
        cumulative
        .cummax()
    )

    drawdown = (
        cumulative
        - peak
    )

    return float(
        abs(
            drawdown.min()
        )
    )


# ============================================================
# RESUME STRATEGIE
# ============================================================

def summarize_mask(
    data: pd.DataFrame,
    mask,
    strategy: str,
    segment: str,
):

    subset = data[
        mask
    ].copy()

    if subset.empty:
        return None

    subset = subset.sort_values(
        [
            "race_date",
            "meeting_number",
            "race_number",
            "neural_rank",
        ]
    )

    bets = len(
        subset
    )

    wins = int(
        subset[
            "is_placed"
        ].sum()
    )

    stake = float(
        bets
    )

    gross = float(
        subset[
            "gross_return"
        ].sum()
    )

    profit = (
        gross
        - stake
    )

    roi = (
        profit
        / stake
        if stake > 0
        else np.nan
    )

    winning_returns = subset.loc[
        subset[
            "is_placed"
        ],
        "place_return",
    ]

    return {
        "strategy":
            strategy,

        "segment":
            segment,

        "bets":
            bets,

        "placed":
            wins,

        "hit_rate":
            wins / bets,

        "stake":
            stake,

        "gross_return":
            gross,

        "profit":
            profit,

        "roi":
            roi,

        "avg_place_return_win":
            (
                winning_returns.mean()
                if not winning_returns.empty
                else np.nan
            ),

        "median_place_return_win":
            (
                winning_returns.median()
                if not winning_returns.empty
                else np.nan
            ),

        "avg_win_odds":
            pd.to_numeric(
                subset["odds"],
                errors="coerce",
            ).mean(),

        "avg_neural_probability":
            subset[
                "neural_top3_probability"
            ].mean(),

        "max_drawdown":
            max_drawdown(
                subset[
                    "profit"
                ]
            ),

        "races":
            subset[
                "race_id"
            ].nunique(),
    }


# ============================================================
# GENERATION DES STRATEGIES
# ============================================================

def strategy_summary(
    data: pd.DataFrame,
):

    records = []

    # --------------------------------------------------------
    # RANGS INDIVIDUELS
    # --------------------------------------------------------

    for rank in range(
        1,
        11,
    ):

        record = summarize_mask(
            data,
            (
                data[
                    "neural_rank"
                ]
                == rank
            ),
            f"Rang {rank}",
            "ALL",
        )

        if record:
            records.append(
                record
            )

    # --------------------------------------------------------
    # GROUPES DE RANGS
    # --------------------------------------------------------

    groups = {
        "Rangs 4-5":
            [4, 5],

        "Rangs 5-6":
            [5, 6],

        "Rangs 6-7":
            [6, 7],

        "Rangs 7-8":
            [7, 8],

        "Rangs 4-8":
            [4, 5, 6, 7, 8],
    }

    for name, ranks in (
        groups.items()
    ):

        record = summarize_mask(
            data,
            data[
                "neural_rank"
            ].isin(
                ranks
            ),
            name,
            "ALL",
        )

        if record:
            records.append(
                record
            )

    # --------------------------------------------------------
    # RANGS 6-7 PAR DISCIPLINE
    # --------------------------------------------------------

    rank67 = data[
        data[
            "neural_rank"
        ].isin(
            [6, 7]
        )
    ]

    for discipline in sorted(
        rank67[
            "discipline"
        ]
        .dropna()
        .astype(str)
        .unique()
    ):

        record = summarize_mask(
            data,
            (
                data[
                    "neural_rank"
                ].isin(
                    [6, 7]
                )
                &
                (
                    data[
                        "discipline"
                    ].astype(str)
                    == discipline
                )
            ),
            "Rangs 6-7",
            discipline,
        )

        if record:
            records.append(
                record
            )

    # --------------------------------------------------------
    # OBSTACLE
    # --------------------------------------------------------

    obstacle = (
        data[
            "discipline"
        ]
        .astype(str)
        .isin(
            [
                "HAIES",
                "STEEPLECHASE",
                "CROSS_COUNTRY",
            ]
        )
    )

    record = summarize_mask(
        data,
        (
            data[
                "neural_rank"
            ].isin(
                [6, 7]
            )
            & obstacle
        ),
        "Rangs 6-7",
        "OBSTACLE",
    )

    if record:
        records.append(
            record
        )

    # --------------------------------------------------------
    # TROT
    # --------------------------------------------------------

    trot = (
        data[
            "discipline"
        ]
        .astype(str)
        .isin(
            [
                "ATTELE_AUTOSTART",
                "ATTELE_VOLTE",
                "TROT_MONTE",
            ]
        )
    )

    record = summarize_mask(
        data,
        (
            data[
                "neural_rank"
            ].isin(
                [6, 7]
            )
            & trot
        ),
        "Rangs 6-7",
        "TROT",
    )

    if record:
        records.append(
            record
        )

    # --------------------------------------------------------
    # TAILLE PELOTON
    # --------------------------------------------------------

    field_size = pd.to_numeric(
        data[
            "field_size"
        ],
        errors="coerce",
    )

    field_segments = {
        "4-7 partants":
            (
                field_size
                .between(
                    4,
                    7,
                )
            ),

        "8-11 partants":
            (
                field_size
                .between(
                    8,
                    11,
                )
            ),

        "12-15 partants":
            (
                field_size
                .between(
                    12,
                    15,
                )
            ),

        "16+ partants":
            (
                field_size
                >= 16
            ),
    }

    for label, field_mask in (
        field_segments.items()
    ):

        record = summarize_mask(
            data,
            (
                data[
                    "neural_rank"
                ].isin(
                    [6, 7]
                )
                & field_mask
            ),
            "Rangs 6-7",
            label,
        )

        if record:
            records.append(
                record
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    start_date: date,
    end_date: date,
):

    print(
        "=" * 72
    )

    print(
        "HORSEPRONO NEURAL "
        "- BACKTEST SIMPLE PLACE"
    )

    print(
        "=" * 72
    )

    print(
        f"Période : "
        f"{start_date} → {end_date}"
    )

    # --------------------------------------------------------
    # MODELE FIGE
    # --------------------------------------------------------

    current_hash = sha256_file(
        MODEL_PATH
    )

    print(
        "Hash Neural : "
        f"{current_hash}"
    )

    if (
        current_hash
        != NEURAL_HASH
    ):

        raise RuntimeError(
            "Le Neural n'est pas "
            "le modèle #9 gelé."
        )

    bundle = load_neural_v1()

    # --------------------------------------------------------
    # SUPABASE
    # --------------------------------------------------------

    client = get_client()

    races = load_races(
        client,
        start_date,
        end_date,
    )

    print(
        f"Courses Supabase : "
        f"{len(races)}"
    )

    if races.empty:

        raise RuntimeError(
            "Aucune course trouvée."
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
        f"Partants Neural : "
        f"{len(data)}"
    )

    # --------------------------------------------------------
    # PREDICTIONS NEURAL
    # --------------------------------------------------------

    predictions = (
        predict_dataframe(
            data,
            bundle,
        )
    )

    print(
        "Prédictions Neural "
        "calculées."
    )

    # --------------------------------------------------------
    # RAPPORTS SIMPLE PLACE
    # --------------------------------------------------------

    result_rows = []

    race_groups = (
        predictions.groupby(
            "race_id",
            sort=False,
        )
    )

    total_races = (
        predictions[
            "race_id"
        ].nunique()
    )

    usable_races = 0

    no_reports = 0

    no_simple_place = 0

    for index, (
        race_id,
        race,
    ) in enumerate(
        race_groups,
        start=1,
    ):

        row0 = race.iloc[0]

        race_date = (
            pd.Timestamp(
                row0[
                    "race_date"
                ]
            )
            .date()
        )

        reunion = _safe_int(
            row0.get(
                "meeting_number"
            )
        )

        course = _safe_int(
            row0.get(
                "race_number"
            )
        )

        if (
            reunion is None
            or course is None
        ):

            continue

        try:

            payload = (
                get_final_reports(
                    race_date,
                    reunion,
                    course,
                )
            )

        except Exception as exc:

            no_reports += 1

            print(
                f"[{index}/{total_races}] "
                f"{race_date} "
                f"R{reunion}C{course} "
                f"rapports indisponibles : "
                f"{exc}"
            )

            continue

        (
            place_returns,
            simple_place_found,
        ) = extract_simple_place_returns(
            payload
        )

        if not simple_place_found:

            no_simple_place += 1
            continue

        if not place_returns:

            no_simple_place += 1
            continue

        usable_races += 1

        print(
            f"[{index}/{total_races}] "
            f"{race_date} "
            f"R{reunion}C{course} "
            f"SP={place_returns}"
        )

        for _, horse in (
            race.iterrows()
        ):

            number = _safe_int(
                horse[
                    "horse_number"
                ]
            )

            if number is None:
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
                        int(race_id),

                    "race_date":
                        race_date,

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

                    "field_size":
                        horse.get(
                            "field_size"
                        ),

                    "horse_number":
                        number,

                    "horse_name":
                        horse.get(
                            "horse_name"
                        ),

                    "odds":
                        _safe_float(
                            horse.get(
                                "odds"
                            )
                        ),

                    "neural_rank":
                        int(
                            horse[
                                "neural_rank"
                            ]
                        ),

                    "neural_top3_probability":
                        float(
                            horse[
                                "neural_top3_probability"
                            ]
                        ),

                    "finish_position":
                        _safe_int(
                            horse.get(
                                "finish_position"
                            )
                        ),

                    "is_placed":
                        is_placed,

                    "place_return":
                        (
                            float(payout)
                            if payout
                            is not None
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

                    "period":
                        (
                            "FORWARD_VALIDATION"
                            if race_date
                            >= FORWARD_START
                            else "EXPLORATION"
                        ),
                }
            )

        time.sleep(
            0.05
        )

    detail = pd.DataFrame(
        result_rows
    )

    print()
    print(
        "=" * 72
    )

    print(
        f"Courses exploitables : "
        f"{usable_races}"
    )

    print(
        f"Rapports indisponibles : "
        f"{no_reports}"
    )

    print(
        f"Sans Simple Placé : "
        f"{no_simple_place}"
    )

    if detail.empty:

        raise RuntimeError(
            "Aucun rapport Simple Placé "
            "exploitable."
        )

    # --------------------------------------------------------
    # SAUVEGARDE DETAIL
    # --------------------------------------------------------

    detail_path = (
        OUTPUT_DIR
        / "simple_place_detail.csv"
    )

    detail.to_csv(
        detail_path,
        index=False,
    )

    # --------------------------------------------------------
    # EXPLORATION
    # --------------------------------------------------------

    exploration = detail[
        detail[
            "period"
        ]
        == "EXPLORATION"
    ].copy()

    if not exploration.empty:

        exploration_summary = (
            strategy_summary(
                exploration
            )
        )

        exploration_summary[
            "enough_sample_30"
        ] = (
            exploration_summary[
                "bets"
            ]
            >= 30
        )

        exploration_summary = (
            exploration_summary
            .sort_values(
                [
                    "enough_sample_30",
                    "roi",
                    "bets",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
        )

        exploration_path = (
            OUTPUT_DIR
            / "simple_place_exploration.csv"
        )

        exploration_summary.to_csv(
            exploration_path,
            index=False,
        )

        print()
        print(
            "TOP STRATEGIES "
            "EXPLORATION"
        )

        print(
            exploration_summary[
                [
                    "strategy",
                    "segment",
                    "bets",
                    "placed",
                    "hit_rate",
                    "profit",
                    "roi",
                    "avg_place_return_win",
                    "max_drawdown",
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # FORWARD VALIDATION
    # --------------------------------------------------------

    forward = detail[
        detail[
            "period"
        ]
        == "FORWARD_VALIDATION"
    ].copy()

    if not forward.empty:

        forward_summary = (
            strategy_summary(
                forward
            )
        )

        forward_summary = (
            forward_summary
            .sort_values(
                [
                    "roi",
                    "bets",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
        )

        forward_path = (
            OUTPUT_DIR
            / "simple_place_forward.csv"
        )

        forward_summary.to_csv(
            forward_path,
            index=False,
        )

        print()
        print(
            "VALIDATION FORWARD"
        )

        print(
            forward_summary[
                [
                    "strategy",
                    "segment",
                    "bets",
                    "placed",
                    "hit_rate",
                    "profit",
                    "roi",
                    "avg_place_return_win",
                    "max_drawdown",
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    print()
    print(
        "Fichiers écrits dans :"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "=" * 72
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help=(
            "Nombre de jours "
            "à analyser."
        ),
    )

    parser.add_argument(
        "--start",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--end",
        type=str,
        default=None,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if args.end:

        end_date = (
            pd.Timestamp(
                args.end
            )
            .date()
        )

    else:

        end_date = (
            date.today()
        )

    if args.start:

        start_date = (
            pd.Timestamp(
                args.start
            )
            .date()
        )

    else:

        start_date = (
            end_date
            - timedelta(
                days=max(
                    1,
                    args.days,
                )
                - 1
            )
        )

    run_backtest(
        start_date,
        end_date,
    )


if __name__ == "__main__":

    main()
