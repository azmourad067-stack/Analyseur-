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
        if pd.isna(value)
        else str(value)
        .strip()
        .lower()
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def _discipline_bucket(
    value: Any,
) -> str:
    """
    Regroupe les disciplines PMU dans les familles V4.

    Sorties possibles :
        PLAT
        AUTOSTART
        VOLTE
        MONTE
        OBSTACLE
        OTHER
    """

    if value is None:
        return "OTHER"

    text = str(
        value
    ).strip().upper()

    text = (
        text
        .replace("É", "E")
        .replace("È", "E")
        .replace("Ê", "E")
        .replace("À", "A")
        .replace("Ô", "O")
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

    obstacle_tokens = (
        "HAIE",
        "STEEPLE",
        "CROSS",
        "OBSTACLE",
    )

    if any(
        token in text
        for token in obstacle_tokens
    ):
        return "OBSTACLE"

    return "OTHER"


# ============================================================
# FEATURES TEMPORELLES
# ============================================================

def build_temporal_training_frame(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construit les statistiques strictement antérieures
    à la course courante.

    Règle essentielle :
    le résultat de la course courante ne doit jamais
    être utilisé pour prédire cette même course.
    """

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
        df["race_date"],
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
    ].astype(str)

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

    # ========================================================
    # RESULTATS ANTERIEURS UNIQUEMENT
    # ========================================================

    df[
        "is_win"
    ] = (
        df[
            "finish_position"
        ]
        == 1
    ).astype(float)

    df[
        "is_place"
    ] = (
        df[
            "finish_position"
        ]
        <= 3
    ).astype(float)

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
        ] = prior_starts.astype(
            float
        )

    # ========================================================
    # FORME RECENTE
    # ========================================================

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
                    finish - 1.0
                )
                / (
                    field - 1.0
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
# MATRICE DE FEATURES V4
# ============================================================

def make_features(
    df: pd.DataFrame,
    *,
    already_temporal: bool = False,
) -> pd.DataFrame:

    if already_temporal:

        x = df.copy()

    else:

        x = normalize_race_input(
            df
        )

    x = x.copy()

    # ========================================================
    # TAILLE DU PELOTON
    # ========================================================

    field = pd.to_numeric(
        x[
            "field_size"
        ],
        errors="coerce",
    )

    if field.notna().any():

        default_field = (
            field.median()
        )

    else:

        default_field = max(
            len(x),
            2,
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
    )

    odds = odds.clip(
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
    # POIDS RELATIF A LA COURSE
    # ========================================================

    weight = pd.to_numeric(
        x[
            "weight"
        ],
        errors="coerce",
    )

    if "race_id" in x.columns:

        race_groups = (
            x[
                "race_id"
            ]
            .astype(str)
        )

    else:

        race_groups = pd.Series(
            "one",
            index=x.index,
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

    if weight.notna().any():

        global_weight = (
            weight.median()
        )

    else:

        global_weight = 57.0

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

    # ========================================================
    # TAILLE PELOTON NORMALISEE
    # ========================================================

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
    # CREATION DATAFRAME
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

    # ========================================================
    # STATS HISTORIQUES
    # ========================================================

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
    # DISCIPLINE V4
    # ========================================================

    if "discipline" in x.columns:

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
    ).astype(float)

    is_autostart = (
        discipline
        == "AUTOSTART"
    ).astype(float)

    is_volte = (
        discipline
        == "VOLTE"
    ).astype(float)

    is_monte = (
        discipline
        == "MONTE"
    ).astype(float)

    is_obstacle = (
        discipline
        == "OBSTACLE"
    ).astype(float)

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
    # INTERACTIONS PLAT
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

    # ========================================================
    # INTERACTIONS AUTOSTART
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

    # ========================================================
    # INTERACTIONS VOLTE
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

    # ========================================================
    # INTERACTIONS MONTE
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

    # ========================================================
    # INTERACTIONS OBSTACLE
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

    # Ordre déterministe indispensable pour
    # le stockage du modèle JSON.
    return out[
        FEATURE_COLUMNS
    ]


# ============================================================
# SNAPSHOT POUR UNE COURSE FUTURE
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
# ENRICHISSEMENT LIVE
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
