from __future__ import annotations

import re

from collections import (
    defaultdict,
    deque,
)

from typing import Any

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS

from .data import (
    normalize_history,
    normalize_race_input,
)


# ============================================================
# OUTILS
# ============================================================

def _norm_text(
    value: Any,
) -> str:

    text = (
        ""
        if pd.isna(
            value
        )
        else str(
            value
        )
        .strip()
        .lower()
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    )


def _discipline_bucket(
    value: Any,
) -> str:

    if value is None:
        return "OTHER"

    text = (
        str(
            value
        )
        .strip()
        .upper()
    )

    text = (
        text
        .replace(
            "É",
            "E",
        )
        .replace(
            "È",
            "E",
        )
        .replace(
            "Ê",
            "E",
        )
        .replace(
            "À",
            "A",
        )
        .replace(
            "Ô",
            "O",
        )
    )

    if "AUTOSTART" in text:
        return "AUTOSTART"

    if (
        "ATTELE" in text
        or "VOLTE" in text
    ):
        return "VOLTE"

    if "MONTE" in text:
        return "MONTE"

    if "PLAT" in text:
        return "PLAT"

    if any(
        token in text
        for token in (
            "HAIE",
            "STEEPLE",
            "CROSS",
            "OBSTACLE",
        )
    ):
        return "OBSTACLE"

    return "OTHER"


def _sex_bucket(
    value: Any,
) -> str:

    if value is None:
        return "UNKNOWN"

    text = (
        str(
            value
        )
        .strip()
        .upper()
    )

    if not text:
        return "UNKNOWN"

    if (
        "HONGRE"
        in text
    ):
        return "GELDING"

    if (
        "FEMELLE"
        in text
    ):
        return "FEMALE"

    if (
        "MALE"
        in text
        or "MÂLE"
        in text
    ):
        return "MALE"

    return "UNKNOWN"


# ============================================================
# FEATURES TEMPORELLES
# ============================================================

def build_temporal_training_frame(
    history: pd.DataFrame,
) -> pd.DataFrame:

    df = normalize_history(
        history
    ).copy()

    df = df.dropna(
        subset=[
            "finish_position",
            "race_date",
        ]
    ).copy()

    if df.empty:
        return df

    df[
        "race_date"
    ] = pd.to_datetime(
        df[
            "race_date"
        ],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "race_date"
        ]
    )

    df[
        "horse_key"
    ] = df[
        "horse_name"
    ].map(
        _norm_text
    )

    df[
        "jockey_key"
    ] = df[
        "jockey"
    ].map(
        _norm_text
    )

    df[
        "trainer_key"
    ] = df[
        "trainer"
    ].map(
        _norm_text
    )

    df[
        "race_key"
    ] = df[
        "race_id"
    ].astype(
        str
    )

    df = (
        df
        .sort_values(
            [
                "race_date",
                "race_key",
                "horse_number",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Résultats antérieurs uniquement
    # --------------------------------------------------------

    df[
        "is_win"
    ] = (
        df[
            "finish_position"
        ]
        == 1
    ).astype(
        float
    )

    df[
        "is_place"
    ] = (
        df[
            "finish_position"
        ]
        <= 3
    ).astype(
        float
    )

    for entity in [
        "horse_key",
        "jockey_key",
        "trainer_key",
    ]:

        grouped = df.groupby(
            entity,
            dropna=False,
        )

        prior_wins = (
            grouped[
                "is_win"
            ].cumsum()
            - df[
                "is_win"
            ]
        )

        prior_places = (
            grouped[
                "is_place"
            ].cumsum()
            - df[
                "is_place"
            ]
        )

        prior_starts = (
            grouped
            .cumcount()
        )

        prefix = entity.split(
            "_"
        )[0]

        df[
            f"{prefix}_win_rate"
        ] = (
            prior_wins
            + 1.0
        ) / (
            prior_starts
            + 10.0
        )

        df[
            f"{prefix}_place_rate"
        ] = (
            prior_places
            + 3.0
        ) / (
            prior_starts
            + 10.0
        )

        df[
            f"{prefix}_starts_prior"
        ] = (
            prior_starts
            .astype(
                float
            )
        )

    # --------------------------------------------------------
    # Forme récente
    # --------------------------------------------------------

    form_values: dict[
        str,
        deque[float],
    ] = defaultdict(
        lambda: deque(
            maxlen=5
        )
    )

    recent_form = []

    recent_consistency = []

    for _, row in df.iterrows():

        key = row[
            "horse_key"
        ]

        hist = list(
            form_values[
                key
            ]
        )

        if hist:

            weights = np.exp(
                -0.45
                * np.arange(
                    len(hist)
                )[::-1]
            )

            weights = (
                weights
                / weights.sum()
            )

            recent_form.append(
                float(
                    np.dot(
                        hist,
                        weights,
                    )
                )
            )

            recent_consistency.append(
                float(
                    1
                    - np.std(
                        hist
                    )
                )
            )

        else:

            recent_form.append(
                0.45
            )

            recent_consistency.append(
                0.50
            )

        field = max(
            float(
                row.get(
                    "field_size"
                )
                or 10
            ),
            2.0,
        )

        finish = float(
            row[
                "finish_position"
            ]
        )

        score = max(
            0.0,
            min(
                1.0,
                1.0
                - (
                    finish
                    - 1.0
                )
                / (
                    field
                    - 1.0
                ),
            ),
        )

        form_values[
            key
        ].append(
            score
        )

    df[
        "recent_form_score"
    ] = recent_form

    df[
        "recent_consistency"
    ] = recent_consistency

    return df


# ============================================================
# MATRICE FEATURES
# ============================================================

def make_features(
    df: pd.DataFrame,
    *,
    already_temporal: bool = False,
) -> pd.DataFrame:

    x = (
        df.copy()
        if already_temporal
        else normalize_race_input(
            df
        )
    )

    x = x.copy()

    # ========================================================
    # FIELD SIZE
    # ========================================================

    field = pd.to_numeric(
        x[
            "field_size"
        ],
        errors="coerce",
    )

    default_field = (
        field.median()
        if field.notna().any()
        else max(
            len(x),
            2,
        )
    )

    field = (
        field
        .fillna(
            default_field
        )
        .clip(
            lower=2
        )
    )

    # ========================================================
    # COTES
    # ========================================================

    odds = pd.to_numeric(
        x[
            "odds"
        ],
        errors="coerce",
    ).clip(
        lower=1.01
    )

    market = (
        1.0
        / odds
    )

    # ========================================================
    # CORDE
    # ========================================================

    draw = pd.to_numeric(
        x[
            "draw"
        ],
        errors="coerce",
    )

    draw_norm = (
        draw - 1.0
    ) / (
        field - 1.0
    )

    draw_norm = (
        draw_norm
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(
            0.50
        )
        .clip(
            0,
            1,
        )
    )

    draw_low = (
        1
        - draw_norm
    ).clip(
        0,
        1,
    )

    # ========================================================
    # POIDS
    # ========================================================

    weight = pd.to_numeric(
        x[
            "weight"
        ],
        errors="coerce",
    )

    race_groups = (
        x[
            "race_id"
        ].astype(
            str
        )
        if "race_id"
        in x.columns
        else pd.Series(
            "one",
            index=x.index,
        )
    )

    weight_ref = (
        weight
        .groupby(
            race_groups
        )
        .transform(
            "median"
        )
    )

    global_weight = (
        weight.median()
        if weight.notna().any()
        else 57.0
    )

    weight_ref = (
        weight_ref
        .fillna(
            global_weight
        )
    )

    weight_rel = (
        (
            weight
            - weight_ref
        )
        / 5.0
    )

    weight_rel = (
        weight_rel
        .fillna(
            0.0
        )
        .clip(
            -3,
            3,
        )
    )

    # ========================================================
    # DISTANCE
    # ========================================================

    distance = pd.to_numeric(
        x[
            "distance"
        ],
        errors="coerce",
    )

    distance_norm = (
        (
            distance
            .fillna(
                2100.0
            )
            - 2100.0
        )
        / 1000.0
    ).clip(
        -3,
        3,
    )

    field_norm = (
        (
            field
            - 10.0
        )
        / 10.0
    ).clip(
        -2,
        2,
    )

    # ========================================================
    # AGE
    # ========================================================

    if (
        "age"
        in x.columns
    ):

        age = pd.to_numeric(
            x[
                "age"
            ],
            errors="coerce",
        )

    else:

        age = pd.Series(
            np.nan,
            index=x.index,
            dtype=float,
        )

    age_missing = (
        age.isna()
        .astype(
            float
        )
    )

    # 5 ans est proche de la médiane globale de nos données.
    age_filled = (
        age
        .fillna(
            5.0
        )
        .clip(
            2,
            15,
        )
    )

    age_norm = (
        (
            age_filled
            - 5.0
        )
        / 3.0
    ).clip(
        -1.5,
        3.5,
    )

    # Age relatif aux adversaires de la course.
    age_race_median = (
        age
        .groupby(
            race_groups
        )
        .transform(
            "median"
        )
    )

    age_race_median = (
        age_race_median
        .fillna(
            5.0
        )
    )

    age_relative = (
        (
            age_filled
            - age_race_median
        )
        / 3.0
    ).clip(
        -3,
        3,
    )

    # ========================================================
    # SEXE
    # ========================================================

    if (
        "sex"
        in x.columns
    ):

        sex_bucket = (
            x[
                "sex"
            ]
            .map(
                _sex_bucket
            )
        )

    else:

        sex_bucket = pd.Series(
            "UNKNOWN",
            index=x.index,
        )

    sex_female = (
        sex_bucket
        == "FEMALE"
    ).astype(
        float
    )

    sex_gelding = (
        sex_bucket
        == "GELDING"
    ).astype(
        float
    )

    sex_known = (
        sex_bucket
        != "UNKNOWN"
    ).astype(
        float
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    out = pd.DataFrame(
        index=x.index
    )

    out[
        "log_odds"
    ] = np.log(
        odds.fillna(
            20.0
        )
    )

    out[
        "market_prob"
    ] = market.fillna(
        1.0
        / 20.0
    )

    out[
        "draw_norm"
    ] = draw_norm

    out[
        "draw_low_advantage"
    ] = draw_low

    out[
        "weight_rel"
    ] = weight_rel

    out[
        "distance_norm"
    ] = distance_norm

    out[
        "field_size_norm"
    ] = field_norm

    # --------------------------------------------------------
    # Helper historique
    # --------------------------------------------------------

    def col_or_default(
        name: str,
        default: float,
    ) -> pd.Series:

        if name in x.columns:

            series = pd.to_numeric(
                x[
                    name
                ],
                errors="coerce",
            )

        else:

            series = pd.Series(
                default,
                index=x.index,
                dtype=float,
            )

        return series.fillna(
            default
        )

    out[
        "horse_win_rate"
    ] = col_or_default(
        "horse_win_rate",
        0.10,
    ).clip(
        0,
        1,
    )

    if (
        "horse_win_rate"
        not in x.columns
        and
        "career_win_rate"
        in x.columns
    ):

        out[
            "horse_win_rate"
        ] = col_or_default(
            "career_win_rate",
            0.10,
        ).clip(
            0,
            1,
        )

    out[
        "horse_place_rate"
    ] = col_or_default(
        "horse_place_rate",
        0.30,
    ).clip(
        0,
        1,
    )

    if (
        "horse_place_rate"
        not in x.columns
        and
        "career_place_rate"
        in x.columns
    ):

        out[
            "horse_place_rate"
        ] = col_or_default(
            "career_place_rate",
            0.30,
        ).clip(
            0,
            1,
        )

    out[
        "jockey_win_rate"
    ] = col_or_default(
        "jockey_win_rate",
        0.10,
    ).clip(
        0,
        1,
    )

    out[
        "jockey_place_rate"
    ] = col_or_default(
        "jockey_place_rate",
        0.30,
    ).clip(
        0,
        1,
    )

    out[
        "trainer_win_rate"
    ] = col_or_default(
        "trainer_win_rate",
        0.10,
    ).clip(
        0,
        1,
    )

    out[
        "trainer_place_rate"
    ] = col_or_default(
        "trainer_place_rate",
        0.30,
    ).clip(
        0,
        1,
    )

    out[
        "recent_form_score"
    ] = col_or_default(
        "recent_form_score",
        0.45,
    ).clip(
        0,
        1,
    )

    out[
        "recent_consistency"
    ] = col_or_default(
        "recent_consistency",
        0.50,
    ).clip(
        0,
        1,
    )

    out[
        "horse_starts_prior"
    ] = col_or_default(
        "horse_starts_prior",
        0,
    ).clip(
        0,
        1000,
    )

    out[
        "jockey_starts_prior"
    ] = col_or_default(
        "jockey_starts_prior",
        0,
    ).clip(
        0,
        10000,
    )

    out[
        "trainer_starts_prior"
    ] = col_or_default(
        "trainer_starts_prior",
        0,
    ).clip(
        0,
        10000,
    )

    # ========================================================
    # AGE / SEXE
    # ========================================================

    out[
        "age_norm"
    ] = age_norm

    out[
        "age_relative"
    ] = age_relative

    out[
        "age_missing"
    ] = age_missing

    out[
        "sex_female"
    ] = sex_female

    out[
        "sex_gelding"
    ] = sex_gelding

    out[
        "sex_known"
    ] = sex_known

    # ========================================================
    # DISCIPLINE
    # ========================================================

    if (
        "discipline"
        in x.columns
    ):

        discipline = (
            x[
                "discipline"
            ]
            .map(
                _discipline_bucket
            )
        )

    else:

        discipline = pd.Series(
            "OTHER",
            index=x.index,
        )

    is_plat = (
        discipline
        == "PLAT"
    ).astype(
        float
    )

    is_autostart = (
        discipline
        == "AUTOSTART"
    ).astype(
        float
    )

    is_volte = (
        discipline
        == "VOLTE"
    ).astype(
        float
    )

    is_monte = (
        discipline
        == "MONTE"
    ).astype(
        float
    )

    is_obstacle = (
        discipline
        == "OBSTACLE"
    ).astype(
        float
    )

    out[
        "discipline_plat"
    ] = is_plat

    out[
        "discipline_autostart"
    ] = is_autostart

    out[
        "discipline_volte"
    ] = is_volte

    out[
        "discipline_monte"
    ] = is_monte

    out[
        "discipline_obstacle"
    ] = is_obstacle

    # ========================================================
    # PLAT
    # ========================================================

    out[
        "plat_weight_rel"
    ] = (
        is_plat
        * weight_rel
    )

    out[
        "plat_draw_norm"
    ] = (
        is_plat
        * draw_norm
    )

    out[
        "plat_distance_norm"
    ] = (
        is_plat
        * distance_norm
    )

    out[
        "plat_age_norm"
    ] = (
        is_plat
        * age_norm
    )

    # ========================================================
    # AUTOSTART
    # ========================================================

    out[
        "autostart_draw_norm"
    ] = (
        is_autostart
        * draw_norm
    )

    out[
        "autostart_distance_norm"
    ] = (
        is_autostart
        * distance_norm
    )

    out[
        "autostart_field_size_norm"
    ] = (
        is_autostart
        * field_norm
    )

    out[
        "autostart_age_norm"
    ] = (
        is_autostart
        * age_norm
    )

    # ========================================================
    # VOLTE
    # ========================================================

    out[
        "volte_distance_norm"
    ] = (
        is_volte
        * distance_norm
    )

    out[
        "volte_field_size_norm"
    ] = (
        is_volte
        * field_norm
    )

    out[
        "volte_age_norm"
    ] = (
        is_volte
        * age_norm
    )

    # ========================================================
    # MONTE
    # ========================================================

    out[
        "monte_weight_rel"
    ] = (
        is_monte
        * weight_rel
    )

    out[
        "monte_distance_norm"
    ] = (
        is_monte
        * distance_norm
    )

    out[
        "monte_age_norm"
    ] = (
        is_monte
        * age_norm
    )

    # ========================================================
    # OBSTACLE
    # ========================================================

    out[
        "obstacle_weight_rel"
    ] = (
        is_obstacle
        * weight_rel
    )

    out[
        "obstacle_distance_norm"
    ] = (
        is_obstacle
        * distance_norm
    )

    out[
        "obstacle_age_norm"
    ] = (
        is_obstacle
        * age_norm
    )

    return out[
        FEATURE_COLUMNS
    ]


# ============================================================
# SNAPSHOT HISTORIQUE
# ============================================================

def entity_snapshot_from_history(
    history: pd.DataFrame,
    race_date: Any,
) -> dict[str, pd.DataFrame]:

    df = build_temporal_training_frame(
        history
    )

    if df.empty:

        return {
            "horse":
                pd.DataFrame(),

            "jockey":
                pd.DataFrame(),

            "trainer":
                pd.DataFrame(),
        }

    cutoff = pd.Timestamp(
        race_date
    )

    df = df[
        df[
            "race_date"
        ]
        < cutoff
    ].copy()

    result: dict[
        str,
        pd.DataFrame,
    ] = {}

    for (
        entity,
        key_col,
        prefix,
    ) in [

        (
            "horse",
            "horse_key",
            "horse",
        ),

        (
            "jockey",
            "jockey_key",
            "jockey",
        ),

        (
            "trainer",
            "trainer_key",
            "trainer",
        ),
    ]:

        if df.empty:

            result[
                entity
            ] = pd.DataFrame()

            continue

        cols = [
            key_col,
            f"{prefix}_win_rate",
            f"{prefix}_place_rate",
            f"{prefix}_starts_prior",
        ]

        snap = (
            df
            .sort_values(
                "race_date"
            )
            .groupby(
                key_col,
                as_index=False,
            )
            .tail(1)[
                cols
            ]
            .copy()
        )

        snap = snap.rename(
            columns={
                key_col:
                    f"{prefix}_key"
            }
        )

        result[
            entity
        ] = snap

    return result


# ============================================================
# COURSE LIVE
# ============================================================

def enrich_live_race(
    race: pd.DataFrame,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:

    out = normalize_race_input(
        race
    )

    if (
        history is None
        or history.empty
        or out.empty
    ):
        return out

    race_date = pd.to_datetime(
        out[
            "race_date"
        ].iloc[0],
        errors="coerce",
    )

    if pd.isna(
        race_date
    ):
        return out

    snap = entity_snapshot_from_history(
        history,
        race_date,
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

        if snap[
            entity
        ].empty:

            continue

        keys = out[
            column
        ].map(
            _norm_text
        )

        tmp = (
            snap[
                entity
            ]
            .set_index(
                f"{prefix}_key"
            )
        )

        for metric in [
            "win_rate",
            "place_rate",
            "starts_prior",
        ]:

            target = (
                f"{prefix}_{metric}"
            )

            mapped = keys.map(
                tmp[
                    target
                ]
            )

            if target in out.columns:

                out[
                    target
                ] = mapped.fillna(
                    out[
                        target
                    ]
                )

            else:

                out[
                    target
                ] = mapped

    return out
