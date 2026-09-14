from __future__ import annotations

import os
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ============================================================
# PATHS
# ============================================================

HORSE_PRONO_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

PROJECT_ROOT = (
    HORSE_PRONO_ROOT
    .parent
)

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

from app_core.data import (
    normalize_race_input,
)

from app_core.db import (
    get_history_as_of,
    get_supabase_client,
)

from app_core.features import (
    entity_snapshot_from_history,
)

from app_core.model import (
    HorseRacingModel,
)

from app_core.pmu import (
    get_programme,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_VERSION_ID = 8

MODEL_HASH = (
    "78d65eeab7f9264521aabf158e1addeae0566549f3b72b2e4ea10083d7114539"
)

START_DATE = date(
    2026,
    6,
    1,
)

END_DATE = date(
    2026,
    9,
    8,
)

TEST_START = date(
    2026,
    8,
    23,
)

STAKE = 10.0

REQUEST_TIMEOUT = 20

PMU_BASE_URL = (
    "https://online.turfinfo.api.pmu.fr/"
    "rest/client/1"
)

HEADERS = {
    "User-Agent":
        "HorseProno/1.0",
    "Accept":
        "application/json",
}

OUTPUT_DIR = (
    HORSE_PRONO_ROOT
    / "artifacts"
    / "quinte_simple_place_m8"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPERS
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


def _safe_int(value):

    try:

        if pd.isna(value):
            return None

        return int(value)

    except Exception:

        return None


def _safe_float(value):

    try:

        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    except Exception:

        return None


def _norm_text(value):

    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value)
        .strip()
        .lower(),
    )


def daterange(
    start_date: date,
    end_date: date,
):

    current = start_date

    while current <= end_date:

        yield current

        current += timedelta(
            days=1
        )


# ============================================================
# MODELE #8 GELE
# ============================================================

def load_model_8(
    client,
):

    response = (
        client
        .table(
            "model_versions"
        )
        .select(
            "*"
        )
        .eq(
            "id",
            MODEL_VERSION_ID,
        )
        .limit(
            1
        )
        .execute()
    )

    rows = _rows(
        response
    )

    if not rows:

        raise RuntimeError(
            "Modèle #8 introuvable."
        )

    record = rows[0]

    actual_hash = str(
        record.get(
            "artifact_hash"
        )
        or ""
    )

    if actual_hash != MODEL_HASH:

        raise RuntimeError(
            "HASH DU MODELE #8 INCORRECT\n"
            f"Attendu : {MODEL_HASH}\n"
            f"Trouvé  : {actual_hash}"
        )

    print(
        "✅ Modèle #8 gelé vérifié"
    )

    print(
        f"   {record.get('model_name')}"
    )

    print(
        f"   hash = {actual_hash}"
    )

    model = (
        HorseRacingModel
        .from_stored_record(
            record
        )
    )

    return model


# ============================================================
# IDENTIFICATION AUTOMATIQUE DU QUINTE+
# ============================================================

def _find_reunions(
    obj,
):

    if isinstance(
        obj,
        dict,
    ):

        reunions = obj.get(
            "reunions"
        )

        if isinstance(
            reunions,
            list,
        ):

            return reunions

        for value in obj.values():

            result = _find_reunions(
                value
            )

            if result is not None:
                return result

    elif isinstance(
        obj,
        list,
    ):

        for value in obj:

            result = _find_reunions(
                value
            )

            if result is not None:
                return result

    return None


def _is_quinte_course(
    course_obj: dict,
):

    paris = (
        course_obj.get(
            "paris"
        )
        or []
    )

    if not isinstance(
        paris,
        list,
    ):

        return False

    for pari in paris:

        if not isinstance(
            pari,
            dict,
        ):

            continue

        values = [
            pari.get(
                "codePari"
            ),
            pari.get(
                "typePari"
            ),
        ]

        for value in values:

            text = str(
                value
                or ""
            ).upper()

            if (
                "QUINTE"
                in text
            ):

                return True

    return False


def find_quinte_supports(
    programme,
):

    supports = []

    reunions = _find_reunions(
        programme
    )

    if not isinstance(
        reunions,
        list,
    ):

        return supports

    for reunion_obj in reunions:

        if not isinstance(
            reunion_obj,
            dict,
        ):

            continue

        reunion = _safe_int(
            reunion_obj.get(
                "numOfficiel"
            )
            or reunion_obj.get(
                "numReunion"
            )
            or reunion_obj.get(
                "numero"
            )
        )

        if reunion is None:
            continue

        hippodrome_obj = (
            reunion_obj.get(
                "hippodrome"
            )
        )

        if isinstance(
            hippodrome_obj,
            dict,
        ):

            hippodrome = (
                hippodrome_obj.get(
                    "libelleCourt"
                )
                or hippodrome_obj.get(
                    "libelleLong"
                )
                or ""
            )

        else:

            hippodrome = str(
                hippodrome_obj
                or ""
            )

        courses = (
            reunion_obj.get(
                "courses"
            )
            or []
        )

        if not isinstance(
            courses,
            list,
        ):

            continue

        for course_obj in courses:

            if not isinstance(
                course_obj,
                dict,
            ):

                continue

            if not _is_quinte_course(
                course_obj
            ):

                continue

            course = _safe_int(
                course_obj.get(
                    "numOrdre"
                )
                or course_obj.get(
                    "numCourse"
                )
                or course_obj.get(
                    "numero"
                )
            )

            if course is None:
                continue

            supports.append(
                {
                    "meeting_number":
                        reunion,

                    "race_number":
                        course,

                    "hippodrome_pmu":
                        hippodrome,

                    "label":
                        str(
                            course_obj.get(
                                "libelle"
                            )
                            or course_obj.get(
                                "libelleCourt"
                            )
                            or ""
                        ),
                }
            )

    return supports


# ============================================================
# COURSE SUPABASE
# ============================================================

def load_race(
    client,
    race_date: date,
    meeting_number: int,
    race_number: int,
):

    response = (
        client
        .table(
            "races"
        )
        .select(
            "*"
        )
        .eq(
            "race_date",
            race_date.isoformat(),
        )
        .eq(
            "meeting_number",
            meeting_number,
        )
        .eq(
            "race_number",
            race_number,
        )
        .limit(
            1
        )
        .execute()
    )

    rows = _rows(
        response
    )

    if not rows:
        return None

    return rows[0]


def load_participants(
    client,
    race_id: int,
):

    response = (
        client
        .table(
            "participants"
        )
        .select(
            "*"
        )
        .eq(
            "race_id",
            race_id,
        )
        .order(
            "horse_number"
        )
        .execute()
    )

    return pd.DataFrame(
        _rows(
            response
        )
    )


# ============================================================
# FORMAT COURSE POUR MODEL 8
# ============================================================

def build_race_input(
    race_row: dict,
    participants: pd.DataFrame,
):

    if participants.empty:

        return pd.DataFrame()

    df = participants.copy()

    if (
        "is_non_runner"
        in df.columns
    ):

        df = df[
            df[
                "is_non_runner"
            ].fillna(
                False
            )
            != True
        ].copy()

    if df.empty:

        return df

    external_id = (
        race_row.get(
            "external_id"
        )
        or (
            f"R"
            f"{race_row.get('meeting_number')}"
            f"C"
            f"{race_row.get('race_number')}"
            f"_"
            f"{race_row.get('race_date')}"
        )
    )

    out = pd.DataFrame(
        index=df.index
    )

    out[
        "race_id"
    ] = str(
        external_id
    )

    out[
        "race_date"
    ] = race_row.get(
        "race_date"
    )

    out[
        "reunion"
    ] = race_row.get(
        "meeting_number"
    )

    out[
        "course_number"
    ] = race_row.get(
        "race_number"
    )

    out[
        "hippodrome"
    ] = race_row.get(
        "hippodrome"
    )

    out[
        "discipline"
    ] = race_row.get(
        "discipline"
    )

    out[
        "distance"
    ] = race_row.get(
        "distance_m"
    )

    out[
        "terrain"
    ] = race_row.get(
        "terrain"
    )

    out[
        "field_size"
    ] = (
        race_row.get(
            "field_size"
        )
        or len(df)
    )

    out[
        "horse_name"
    ] = df[
        "horse_name"
    ].values

    out[
        "horse_number"
    ] = df[
        "horse_number"
    ].values

    out[
        "jockey"
    ] = df[
        "jockey_name"
    ].values

    out[
        "trainer"
    ] = df[
        "trainer_name"
    ].values

    out[
        "odds"
    ] = df[
        "odds"
    ].values

    out[
        "weight"
    ] = df[
        "weight_kg"
    ].values

    out[
        "draw"
    ] = df[
        "draw"
    ].values

    out[
        "age"
    ] = df[
        "age"
    ].values

    out[
        "sex"
    ] = df[
        "sex"
    ].values

    out[
        "recent_form"
    ] = df[
        "recent_form"
    ].values

    out[
        "finish_position"
    ] = df[
        "finish_position"
    ].values

    return out.reset_index(
        drop=True
    )


# ============================================================
# ENRICHISSEMENT IDENTIQUE AU FORWARD
# ============================================================

def enrich_from_snapshot(
    race: pd.DataFrame,
    snapshots,
):

    out = normalize_race_input(
        race
    )

    definitions = [
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
    ]

    for (
        entity,
        column,
        prefix,
    ) in definitions:

        snap = snapshots.get(
            entity
        )

        if (
            snap is None
            or snap.empty
        ):

            continue

        key_column = (
            f"{prefix}_key"
        )

        if (
            key_column
            not in snap.columns
        ):

            continue

        tmp = snap.set_index(
            key_column
        )

        keys = (
            out[
                column
            ]
            .map(
                _norm_text
            )
        )

        for metric in [
            "win_rate",
            "place_rate",
            "starts_prior",
        ]:

            source = (
                f"{prefix}_"
                f"{metric}"
            )

            if source not in tmp.columns:
                continue

            mapped = keys.map(
                tmp[
                    source
                ]
            )

            if source in out.columns:

                out[
                    source
                ] = (
                    mapped
                    .fillna(
                        out[
                            source
                        ]
                    )
                )

            else:

                out[
                    source
                ] = mapped

    return out


# ============================================================
# RAPPORTS DEFINITIFS PMU
# ============================================================

def get_final_reports(
    race_date: date,
    reunion: int,
    course: int,
):

    date_text = (
        race_date.strftime(
            "%d%m%Y"
        )
    )

    url = (
        f"{PMU_BASE_URL}"
        f"/programme/"
        f"{date_text}"
        f"/R{reunion}"
        f"/C{course}"
        f"/rapports-definitifs"
    )

    last_error = None

    for attempt in range(
        3
    ):

        try:

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

        except Exception as exc:

            last_error = exc

            time.sleep(
                1.0
                + attempt
            )

    raise RuntimeError(
        f"Rapports PMU indisponibles : "
        f"{last_error}"
    )


def iter_bet_blocks(
    obj,
):

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

        for key in [
            "numPmu",
            "numero",
            "numParticipant",
            "cheval",
        ]:

            if key in value:

                numbers = (
                    combination_numbers(
                        value[
                            key
                        ]
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


def return_per_euro(
    block: dict,
    report: dict,
):

    base_stake = _safe_float(
        block.get(
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

    per_euro = _safe_float(
        report.get(
            "dividendePourUnEuro"
        )
    )

    if (
        per_euro is not None
        and per_euro > 0
    ):

        if per_euro > 100:

            per_euro /= 100.0

        return float(
            per_euro
        )

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

            dividend /= 100.0

        return float(
            dividend
        )

    return None


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

            numbers = (
                combination_numbers(
                    report.get(
                        "combinaison"
                    )
                )
            )

            numbers = list(
                dict.fromkeys(
                    numbers
                )
            )

            if len(numbers) != 1:
                continue

            payout = (
                return_per_euro(
                    block,
                    report,
                )
            )

            if (
                payout is None
                or payout <= 0
            ):

                continue

            returns[
                int(
                    numbers[0]
                )
            ] = float(
                payout
            )

    return (
        returns,
        simple_place_found,
    )


# ============================================================
# PERIODE
# ============================================================

def period_name(
    value,
):

    value = pd.Timestamp(
        value
    ).date()

    if value < TEST_START:

        return (
            "01_JUIN_22_AOUT_DESCRIPTIF"
        )

    return (
        "23_AOUT_08_SEPT_TEST_GELE"
    )


# ============================================================
# DRAWDOWN
# ============================================================

def max_drawdown(
    profits: pd.Series,
):

    if profits.empty:
        return 0.0

    cumulative = (
        profits
        .fillna(
            0.0
        )
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
# RESUME
# ============================================================

def summarize(
    data: pd.DataFrame,
    group_columns: list[str],
):

    rows = []

    if data.empty:

        return pd.DataFrame()

    grouped = (
        data.groupby(
            group_columns,
            dropna=False,
        )
    )

    for key, group in grouped:

        if (
            len(
                group_columns
            )
            == 1
        ):

            key = (
                key,
            )

        group = (
            group.sort_values(
                [
                    "race_date",
                    "meeting_number",
                    "race_number",
                ]
            )
            .copy()
        )

        bets = len(
            group
        )

        wins = int(
            group[
                "is_placed"
            ].sum()
        )

        stake_total = float(
            group[
                "stake"
            ].sum()
        )

        gross_return = float(
            group[
                "gross_return"
            ].sum()
        )

        profit = float(
            group[
                "profit"
            ].sum()
        )

        roi = (
            100.0
            * profit
            / stake_total
            if stake_total > 0
            else np.nan
        )

        winning_returns = (
            group.loc[
                group[
                    "is_placed"
                ],
                "return_per_euro",
            ]
        )

        row = {}

        for column, value in zip(
            group_columns,
            key,
        ):

            row[
                column
            ] = value

        row.update(
            {
                "bets":
                    bets,

                "placed":
                    wins,

                "hit_rate_pct":
                    (
                        100.0
                        * wins
                        / bets
                        if bets
                        else np.nan
                    ),

                "stake_total":
                    stake_total,

                "gross_return":
                    gross_return,

                "profit":
                    profit,

                "roi_pct":
                    roi,

                "avg_winning_return_per_euro":
                    (
                        float(
                            winning_returns.mean()
                        )
                        if not winning_returns.empty
                        else np.nan
                    ),

                "break_even_return_per_euro":
                    (
                        float(
                            bets
                            / wins
                        )
                        if wins > 0
                        else np.nan
                    ),

                "max_drawdown":
                    max_drawdown(
                        group[
                            "profit"
                        ]
                    ),
            }
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 76
    )

    print(
        "HORSEPRONO - MODEL #8"
    )

    print(
        "QUINTE+ / SIMPLE PLACE / RANGS 1 A 5"
    )

    print(
        "=" * 76
    )

    client = (
        get_supabase_client()
    )

    if client is None:

        raise RuntimeError(
            "Supabase non configuré."
        )

    model = load_model_8(
        client
    )

    # --------------------------------------------------------
    # HISTORIQUE UNIQUE
    # --------------------------------------------------------

    print()

    print(
        "Chargement de l'historique..."
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
        f"Historique chargé : "
        f"{len(history)} lignes"
    )

    detail_rows = []

    support_rows = []

    discovered = 0

    matched = 0

    valid_reports = 0

    # --------------------------------------------------------
    # JOUR PAR JOUR
    # --------------------------------------------------------

    for current_date in daterange(
        START_DATE,
        END_DATE,
    ):

        print()

        print(
            current_date.isoformat()
        )

        try:

            programme = (
                get_programme(
                    current_date
                )
            )

        except Exception as exc:

            print(
                f"  ⚠️ programme indisponible : "
                f"{exc}"
            )

            continue

        supports = (
            find_quinte_supports(
                programme
            )
        )

        if not supports:

            print(
                "  Aucun Quinté+ détecté."
            )

            continue

        discovered += len(
            supports
        )

        for support in supports:

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

            print(
                f"  Quinté+ : "
                f"R{reunion}C{course} "
                f"{support['label']}"
            )

            race_row = load_race(
                client,
                current_date,
                reunion,
                course,
            )

            if race_row is None:

                print(
                    "  ⚠️ course absente de Supabase"
                )

                support_rows.append(
                    {
                        "race_date":
                            current_date,

                        "meeting_number":
                            reunion,

                        "race_number":
                            course,

                        "status":
                            "RACE_NOT_IN_SUPABASE",
                    }
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

                print(
                    "  ⚠️ aucun participant"
                )

                continue

            race = build_race_input(
                race_row,
                participants,
            )

            if race.empty:

                continue

            matched += 1

            # ------------------------------------------------
            # HISTORIQUE STRICTEMENT ANTERIEUR
            # ------------------------------------------------

            snapshots = (
                entity_snapshot_from_history(
                    history,
                    current_date,
                )
            )

            enriched = (
                enrich_from_snapshot(
                    race,
                    snapshots,
                )
            )

            # ------------------------------------------------
            # MODEL #8
            # ------------------------------------------------

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
                .reset_index(
                    drop=True
                )
            )

            predictions[
                "model_rank"
            ] = (
                np.arange(
                    len(
                        predictions
                    )
                )
                + 1
            )

            top5 = (
                predictions[
                    predictions[
                        "model_rank"
                    ]
                    <= 5
                ]
                .copy()
            )

            # ------------------------------------------------
            # RAPPORTS SIMPLE PLACE
            # ------------------------------------------------

            try:

                payload = (
                    get_final_reports(
                        current_date,
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

            except Exception as exc:

                print(
                    f"  ⚠️ rapports indisponibles : "
                    f"{exc}"
                )

                support_rows.append(
                    {
                        "race_date":
                            current_date,

                        "meeting_number":
                            reunion,

                        "race_number":
                            course,

                        "status":
                            "REPORT_UNAVAILABLE",
                    }
                )

                continue

            if not simple_place_found:

                print(
                    "  ⚠️ Simple Placé absent"
                )

                support_rows.append(
                    {
                        "race_date":
                            current_date,

                        "meeting_number":
                            reunion,

                        "race_number":
                            course,

                        "status":
                            "NO_SIMPLE_PLACE",
                    }
                )

                continue

            valid_reports += 1

            # ------------------------------------------------
            # RESULTATS RANGS 1 A 5
            # ------------------------------------------------

            for _, row in top5.iterrows():

                number = int(
                    row[
                        "horse_number"
                    ]
                )

                rank = int(
                    row[
                        "model_rank"
                    ]
                )

                payout = (
                    place_returns.get(
                        number
                    )
                )

                is_placed = (
                    payout is not None
                    and payout > 0
                )

                return_value = (
                    float(
                        payout
                    )
                    if is_placed
                    else 0.0
                )

                gross_return = (
                    STAKE
                    * return_value
                )

                profit = (
                    gross_return
                    - STAKE
                )

                finish_position = (
                    pd.to_numeric(
                        pd.Series(
                            [
                                row.get(
                                    "finish_position"
                                )
                            ]
                        ),
                        errors="coerce",
                    )
                    .iloc[0]
                )

                detail_rows.append(
                    {
                        "race_date":
                            current_date,

                        "period":
                            period_name(
                                current_date
                            ),

                        "race_id":
                            race_row[
                                "id"
                            ],

                        "race_external_id":
                            race_row.get(
                                "external_id"
                            ),

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

                        "model_rank":
                            rank,

                        "horse_number":
                            number,

                        "horse_name":
                            row.get(
                                "horse_name"
                            ),

                        "win_probability":
                            row.get(
                                "win_probability"
                            ),

                        "odds":
                            row.get(
                                "odds"
                            ),

                        "finish_position":
                            (
                                int(
                                    finish_position
                                )
                                if pd.notna(
                                    finish_position
                                )
                                else None
                            ),

                        "is_placed":
                            bool(
                                is_placed
                            ),

                        "return_per_euro":
                            return_value,

                        "stake":
                            STAKE,

                        "gross_return":
                            gross_return,

                        "profit":
                            profit,

                        "model_version_id":
                            MODEL_VERSION_ID,

                        "artifact_hash":
                            MODEL_HASH,
                    }
                )

            support_rows.append(
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

                    "status":
                        "OK",

                    "simple_place_returns":
                        len(
                            place_returns
                        ),
                }
            )

            print(
                f"  ✅ Top 5 évalué / "
                f"{len(place_returns)} "
                f"rapports placés"
            )

            # petite pause API
            time.sleep(
                0.10
            )

    # ========================================================
    # DATAFRAMES
    # ========================================================

    detail = pd.DataFrame(
        detail_rows
    )

    supports_df = pd.DataFrame(
        support_rows
    )

    if detail.empty:

        raise RuntimeError(
            "Aucun résultat exploitable."
        )

    detail[
        "race_date"
    ] = pd.to_datetime(
        detail[
            "race_date"
        ]
    )

    # ========================================================
    # RESUMES
    # ========================================================

    by_rank = summarize(
        detail,
        [
            "model_rank"
        ],
    )

    by_period_rank = summarize(
        detail,
        [
            "period",
            "model_rank",
        ],
    )

    by_discipline_rank = summarize(
        detail,
        [
            "discipline",
            "model_rank",
        ],
    )

    # ========================================================
    # EXPORTS
    # ========================================================

    detail_path = (
        OUTPUT_DIR
        / "quinte_m8_simple_place_detail.csv"
    )

    rank_path = (
        OUTPUT_DIR
        / "quinte_m8_simple_place_by_rank.csv"
    )

    period_path = (
        OUTPUT_DIR
        / "quinte_m8_simple_place_by_period_rank.csv"
    )

    discipline_path = (
        OUTPUT_DIR
        / "quinte_m8_simple_place_by_discipline_rank.csv"
    )

    supports_path = (
        OUTPUT_DIR
        / "quinte_supports_audit.csv"
    )

    detail.to_csv(
        detail_path,
        index=False,
        encoding="utf-8-sig",
    )

    by_rank.to_csv(
        rank_path,
        index=False,
        encoding="utf-8-sig",
    )

    by_period_rank.to_csv(
        period_path,
        index=False,
        encoding="utf-8-sig",
    )

    by_discipline_rank.to_csv(
        discipline_path,
        index=False,
        encoding="utf-8-sig",
    )

    supports_df.to_csv(
        supports_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # CONSOLE
    # ========================================================

    print()

    print(
        "=" * 76
    )

    print(
        "RESULTAT FINAL"
    )

    print(
        "=" * 76
    )

    print(
        f"Supports Quinté détectés : "
        f"{discovered}"
    )

    print(
        f"Supports raccordés Supabase : "
        f"{matched}"
    )

    print(
        f"Courses avec rapports valides : "
        f"{valid_reports}"
    )

    print()

    print(
        "ROI SIMPLE PLACE - MODEL #8"
    )

    print()

    print(
        by_rank.to_string(
            index=False
        )
    )

    print()

    print(
        "PAR PERIODE"
    )

    print()

    print(
        by_period_rank.to_string(
            index=False
        )
    )

    print()

    print(
        f"Résultats enregistrés dans : "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()
