from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .features import build_temporal_training_frame


def _safe_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None

    return x if np.isfinite(x) else None


def _mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return float(values.mean())


def _roi_proxy(
    odds: pd.Series,
    hits: pd.Series,
) -> float | None:
    """
    ROI proxy :
    mise = 1 unité
    victoire = retour égal à la cote historique.
    """

    o = pd.to_numeric(
        odds,
        errors="coerce",
    )

    h = pd.to_numeric(
        hits,
        errors="coerce",
    )

    valid = (
        o.notna()
        & h.notna()
        & (o > 1.0)
    )

    if not valid.any():
        return None

    returns = np.where(
        h.loc[valid].to_numpy(dtype=float) > 0,
        o.loc[valid].to_numpy(dtype=float),
        0.0,
    )

    return float(
        returns.sum() / valid.sum() - 1.0
    )


def _race_summary(
    group: pd.DataFrame,
) -> dict[str, Any]:

    if group.empty:
        return {
            "races": 0,
        }

    trio = group[
        group["trio_eligible"] == 1
    ]

    return {
        "races":
            int(len(group)),

        "top1_accuracy":
            _mean(group["top1_hit"]),

        "winner_in_top2":
            _mean(group["winner_in_top2"]),

        "winner_in_top3":
            _mean(group["winner_in_top3"]),

        "winner_in_top5":
            _mean(group["winner_in_top5"]),

        "avg_podium_captured_top3":
            _mean(
                group[
                    "podium_captured_top3"
                ]
            ),

        "avg_podium_captured_top5":
            _mean(
                group[
                    "podium_captured_top5"
                ]
            ),

        "at_least_2_podium_in_top3":
            _mean(
                group[
                    "at_least_2_podium_in_top3"
                ]
            ),

        "trio_complete_top3":
            _mean(
                trio[
                    "trio_complete_top3"
                ]
            ),

        "trio_complete_top5":
            _mean(
                trio[
                    "trio_complete_top5"
                ]
            ),

        "flat_stake_roi_proxy_top1":
            _roi_proxy(
                group["top1_odds"],
                group["top1_hit"],
            ),

        "top1_model_prob_mean":
            _mean(
                group[
                    "top1_model_prob"
                ]
            ),

        "top1_market_prob_mean":
            _mean(
                group[
                    "top1_market_prob"
                ]
            ),

        "top1_edge_mean":
            _mean(
                group[
                    "top1_edge"
                ]
            ),
    }


def _group_summaries(
    races: pd.DataFrame,
    column: str,
) -> dict[str, Any]:

    result = {}

    if column not in races.columns:
        return result

    for key, group in races.groupby(
        column,
        dropna=False,
    ):

        if pd.isna(key):
            label = "INCONNU"
        else:
            label = str(key)

        result[label] = _race_summary(
            group
        )

    return result


def _edge_strategy_top1(
    races: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:

    edge = pd.to_numeric(
        races["top1_edge"],
        errors="coerce",
    )

    odds = pd.to_numeric(
        races["top1_odds"],
        errors="coerce",
    )

    valid = (
        edge.ge(threshold)
        & odds.gt(1.0)
    )

    bets = races.loc[
        valid
    ].copy()

    if bets.empty:

        return {
            "edge_threshold": threshold,
            "bets": 0,
            "wins": 0,
            "hit_rate": None,
            "roi_proxy": None,
            "avg_edge": None,
            "avg_odds": None,
        }

    return {
        "edge_threshold":
            threshold,

        "bets":
            int(len(bets)),

        "wins":
            int(
                pd.to_numeric(
                    bets["top1_hit"],
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            ),

        "hit_rate":
            _mean(
                bets["top1_hit"]
            ),

        "roi_proxy":
            _roi_proxy(
                bets["top1_odds"],
                bets["top1_hit"],
            ),

        "avg_edge":
            _mean(
                bets["top1_edge"]
            ),

        "avg_odds":
            _mean(
                bets["top1_odds"]
            ),
    }


def _edge_strategy_all_runners(
    runners: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:

    edge = pd.to_numeric(
        runners["edge"],
        errors="coerce",
    )

    odds = pd.to_numeric(
        runners["odds_num"],
        errors="coerce",
    )

    valid = (
        edge.ge(threshold)
        & odds.gt(1.0)
    )

    bets = runners.loc[
        valid
    ].copy()

    if bets.empty:

        return {
            "edge_threshold": threshold,
            "bets": 0,
            "wins": 0,
            "hit_rate": None,
            "roi_proxy": None,
            "avg_edge": None,
            "avg_odds": None,
        }

    hits = (
        pd.to_numeric(
            bets["finish_position"],
            errors="coerce",
        )
        == 1
    ).astype(int)

    return {
        "edge_threshold":
            threshold,

        "bets":
            int(len(bets)),

        "wins":
            int(hits.sum()),

        "hit_rate":
            float(
                hits.mean()
            ),

        "roi_proxy":
            _roi_proxy(
                bets["odds_num"],
                hits,
            ),

        "avg_edge":
            _mean(
                bets["edge"]
            ),

        "avg_odds":
            _mean(
                bets["odds_num"]
            ),
    }


def _calibration_table(
    runners: pd.DataFrame,
) -> tuple[
    list[dict[str, Any]],
    float | None,
]:

    work = runners[
        [
            "model_p",
            "finish_position",
        ]
    ].copy()

    work["actual_win"] = (
        pd.to_numeric(
            work["finish_position"],
            errors="coerce",
        )
        == 1
    ).astype(int)

    bins = [
        0.0,
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
        0.50,
        0.70,
        1.0000001,
    ]

    labels = [
        "0-5%",
        "5-10%",
        "10-15%",
        "15-20%",
        "20-30%",
        "30-40%",
        "40-50%",
        "50-70%",
        "70-100%",
    ]

    work["bin"] = pd.cut(
        pd.to_numeric(
            work["model_p"],
            errors="coerce",
        ),
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    rows = []

    total = int(
        work[
            "bin"
        ].notna().sum()
    )

    weighted_gap = 0.0

    for label, group in work.groupby(
        "bin",
        observed=True,
    ):

        if group.empty:
            continue

        predicted = float(
            pd.to_numeric(
                group["model_p"],
                errors="coerce",
            ).mean()
        )

        observed = float(
            group[
                "actual_win"
            ].mean()
        )

        gap = (
            observed
            - predicted
        )

        n = int(
            len(group)
        )

        if total:

            weighted_gap += (
                n / total
            ) * abs(gap)

        rows.append(
            {
                "bin":
                    str(label),

                "n":
                    n,

                "mean_predicted_probability":
                    predicted,

                "observed_win_rate":
                    observed,

                "calibration_gap":
                    gap,
            }
        )

    ece = (
        float(weighted_gap)
        if total
        else None
    )

    return (
        rows,
        ece,
    )


def build_advanced_backtest(
    model,
    history: pd.DataFrame,
) -> dict[str, Any]:
    """
    Backtest V2 utilisant exactement le holdout
    chronologique du modèle.

    Le modèle doit déjà avoir été entraîné
    avec :

        model.fit(history)
    """

    temporal = (
        build_temporal_training_frame(
            history
        )
    )

    temporal = temporal.dropna(
        subset=[
            "finish_position",
            "race_id",
            "race_date",
        ]
    ).copy()

    if temporal.empty:
        return {}

    # Même découpage temporel que model.fit()
    _, _, test_races = (
        model._split_by_race_time(
            temporal,
            model.config.train_fraction,
            model.config.calibration_fraction,
        )
    )

    test = temporal[
        temporal[
            "race_id"
        ]
        .astype(str)
        .isin(test_races)
    ].copy()

    if test.empty:
        return {}

    # ========================================================
    # Probabilités modèle
    # ========================================================

    raw_win, raw_place = (
        model._raw_predictions(
            test
        )
    )

    p_win = (
        model._normalize_by_race(
            model._apply_calibrator(
                raw_win,
                model.win_calibrator,
            ),
            test["race_id"],
        )
    )

    p_place = (
        model._apply_calibrator(
            raw_place,
            model.place_calibrator,
        )
    )

    # ========================================================
    # Marché
    # ========================================================

    odds = pd.to_numeric(
        test["odds"],
        errors="coerce",
    )

    implied = (
        1.0
        / odds.clip(
            lower=1.01
        )
    )

    implied = implied.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    # Même logique que le modèle V3 :
    # cote absente -> faible probabilité implicite.
    market_input = (
        implied
        .fillna(0.01)
        .to_numpy(
            dtype=float
        )
    )

    market_p = (
        model._normalize_by_race(
            market_input,
            test["race_id"],
        )
    )

    # ========================================================
    # DataFrame individuel
    # ========================================================

    runners = test.copy()

    runners[
        "model_p"
    ] = p_win

    runners[
        "place_p"
    ] = p_place

    runners[
        "market_p"
    ] = market_p

    runners[
        "edge"
    ] = (
        runners["model_p"]
        - runners["market_p"]
    )

    runners[
        "odds_num"
    ] = odds

    # ========================================================
    # Une ligne par course
    # ========================================================

    race_rows = []

    for race_id, group in runners.groupby(
        "race_id",
        sort=False,
    ):

        g = group.sort_values(
            [
                "model_p",
                "place_p",
            ],
            ascending=False,
        ).copy()

        g[
            "model_rank"
        ] = np.arange(
            1,
            len(g) + 1,
        )

        finish = pd.to_numeric(
            g[
                "finish_position"
            ],
            errors="coerce",
        )

        winner_idx = set(
            g.index[
                finish == 1
            ]
        )

        podium_idx = set(
            g.index[
                finish.between(
                    1,
                    3,
                    inclusive="both",
                )
            ]
        )

        top1_idx = set(
            g.head(1).index
        )

        top2_idx = set(
            g.head(2).index
        )

        top3_idx = set(
            g.head(3).index
        )

        top5_idx = set(
            g.head(5).index
        )

        winner_exists = bool(
            winner_idx
        )

        trio_eligible = (
            len(podium_idx)
            >= 3
        )

        top = g.iloc[0]

        race_rows.append(
            {
                "race_id":
                    str(race_id),

                "race_date":
                    str(
                        pd.Timestamp(
                            g[
                                "race_date"
                            ].iloc[0]
                        ).date()
                    ),

                "discipline":
                    (
                        str(
                            g[
                                "discipline"
                            ].iloc[0]
                        )
                        if "discipline"
                        in g.columns
                        else "INCONNU"
                    ),

                "race_size":
                    int(len(g)),

                "top1_hit":
                    (
                        int(
                            bool(
                                top1_idx
                                & winner_idx
                            )
                        )
                        if winner_exists
                        else np.nan
                    ),

                "winner_in_top2":
                    (
                        int(
                            bool(
                                top2_idx
                                & winner_idx
                            )
                        )
                        if winner_exists
                        else np.nan
                    ),

                "winner_in_top3":
                    (
                        int(
                            bool(
                                top3_idx
                                & winner_idx
                            )
                        )
                        if winner_exists
                        else np.nan
                    ),

                "winner_in_top5":
                    (
                        int(
                            bool(
                                top5_idx
                                & winner_idx
                            )
                        )
                        if winner_exists
                        else np.nan
                    ),

                "podium_captured_top3":
                    int(
                        len(
                            top3_idx
                            & podium_idx
                        )
                    ),

                "podium_captured_top5":
                    int(
                        len(
                            top5_idx
                            & podium_idx
                        )
                    ),

                "at_least_2_podium_in_top3":
                    (
                        int(
                            len(
                                top3_idx
                                & podium_idx
                            )
                            >= 2
                        )
                        if trio_eligible
                        else np.nan
                    ),

                "trio_eligible":
                    int(
                        trio_eligible
                    ),

                "trio_complete_top3":
                    (
                        int(
                            podium_idx
                            .issubset(
                                top3_idx
                            )
                        )
                        if trio_eligible
                        else np.nan
                    ),

                "trio_complete_top5":
                    (
                        int(
                            podium_idx
                            .issubset(
                                top5_idx
                            )
                        )
                        if trio_eligible
                        else np.nan
                    ),

                "top1_model_prob":
                    _safe_float(
                        top[
                            "model_p"
                        ]
                    ),

                "top1_market_prob":
                    _safe_float(
                        top[
                            "market_p"
                        ]
                    ),

                "top1_edge":
                    _safe_float(
                        top[
                            "edge"
                        ]
                    ),

                "top1_odds":
                    _safe_float(
                        top[
                            "odds_num"
                        ]
                    ),
            }
        )

    races = pd.DataFrame(
        race_rows
    )

    if races.empty:
        return {}

    # ========================================================
    # Segmentation taille peloton
    # ========================================================

    races[
        "field_size_band"
    ] = pd.cut(
        races[
            "race_size"
        ],
        bins=[
            0,
            7,
            10,
            14,
            np.inf,
        ],
        labels=[
            "<=7",
            "8-10",
            "11-14",
            "15+",
        ],
        include_lowest=True,
    )

    # ========================================================
    # Segmentation cote du choix n°1
    # ========================================================

    races[
        "top1_odds_band"
    ] = pd.cut(
        pd.to_numeric(
            races[
                "top1_odds"
            ],
            errors="coerce",
        ),
        bins=[
            0,
            3,
            5,
            10,
            20,
            np.inf,
        ],
        labels=[
            "<=3",
            "3-5",
            "5-10",
            "10-20",
            "20+",
        ],
        include_lowest=True,
        right=False,
    )

    # ========================================================
    # Discipline
    # ========================================================

    discipline = (
        races[
            "discipline"
        ]
        .fillna(
            "INCONNU"
        )
        .astype(str)
        .str.strip()
    )

    known_discipline = (
        ~discipline
        .str.upper()
        .isin(
            {
                "",
                "INCONNU",
                "NONE",
                "NAN",
            }
        )
    )

    # ========================================================
    # Calibration
    # ========================================================

    calibration, ece = (
        _calibration_table(
            runners
        )
    )

    # 0, 2, 5 et 10 points de probabilité
    thresholds = [
        0.00,
        0.02,
        0.05,
        0.10,
    ]

    # ========================================================
    # Qualité des données
    # ========================================================

    quality = {
        "test_rows":
            int(
                len(runners)
            ),

        "test_races":
            int(
                races[
                    "race_id"
                ].nunique()
            ),

        "odds_available_rate":
            float(
                pd.to_numeric(
                    runners[
                        "odds"
                    ],
                    errors="coerce",
                )
                .notna()
                .mean()
            ),

        "draw_available_rate":
            (
                float(
                    pd.to_numeric(
                        runners[
                            "draw"
                        ],
                        errors="coerce",
                    )
                    .notna()
                    .mean()
                )
                if "draw"
                in runners.columns
                else 0.0
            ),

        "weight_available_rate":
            (
                float(
                    pd.to_numeric(
                        runners[
                            "weight"
                        ],
                        errors="coerce",
                    )
                    .notna()
                    .mean()
                )
                if "weight"
                in runners.columns
                else 0.0
            ),

        "distance_available_rate":
            (
                float(
                    pd.to_numeric(
                        runners[
                            "distance"
                        ],
                        errors="coerce",
                    )
                    .notna()
                    .mean()
                )
                if "distance"
                in runners.columns
                else 0.0
            ),

        "discipline_known_rate":
            float(
                known_discipline
                .mean()
            ),
    }

    result = {
        "version":
            2,

        "test_start":
            str(
                pd.Timestamp(
                    runners[
                        "race_date"
                    ].min()
                ).date()
            ),

        "test_end":
            str(
                pd.Timestamp(
                    runners[
                        "race_date"
                    ].max()
                ).date()
            ),

        "overall":
            _race_summary(
                races
            ),

        "by_field_size":
            _group_summaries(
                races,
                "field_size_band",
            ),

        "by_top1_odds":
            _group_summaries(
                races,
                "top1_odds_band",
            ),

        "edge_strategies": {

            "top1_only": {
                (
                    f"edge_"
                    f"{int(t * 100)}pp"
                ):
                    _edge_strategy_top1(
                        races,
                        t,
                    )

                for t
                in thresholds
            },

            "all_runners": {
                (
                    f"edge_"
                    f"{int(t * 100)}pp"
                ):
                    _edge_strategy_all_runners(
                        runners,
                        t,
                    )

                for t
                in thresholds
            },
        },

        "calibration": {
            "expected_calibration_error":
                ece,

            "bins":
                calibration,
        },

        "data_quality":
            quality,

        "roi_definition": (
            "Proxy : 1 unité par pari, "
            "retour = cote historique si gagnant, "
            "0 sinon. La cote stockée n'est pas "
            "garantie identique au rapport PMU définitif."
        ),
    }

    if known_discipline.any():

        result[
            "by_discipline"
        ] = {
            "available":
                True,

            "groups":
                _group_summaries(
                    races.loc[
                        known_discipline
                    ].copy(),
                    "discipline",
                ),
        }

    else:

        result[
            "by_discipline"
        ] = {
            "available":
                False,

            "reason": (
                "Les disciplines du jeu de test "
                "sont absentes ou marquées INCONNU."
            ),

            "groups":
                {},
        }

    return result
