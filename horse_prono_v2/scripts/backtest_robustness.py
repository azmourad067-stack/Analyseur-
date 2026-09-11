from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(__file__)
    ),
)

from app_core.backtest_v2 import (
    build_advanced_backtest,
)

from app_core.config import (
    FEATURE_COLUMNS,
)

from app_core.db import (
    get_latest_active_model,
    get_training_history,
    save_backtest_run,
)

from app_core.features import (
    build_temporal_training_frame,
)

from app_core.model import (
    HorseRacingModel,
)


EDGE_THRESHOLD = 0.02

BOOTSTRAP_SAMPLES = 2000

RANDOM_STATE = 42


def safe_float(
    value: Any,
) -> float | None:

    try:
        x = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    return (
        x
        if np.isfinite(
            x
        )
        else None
    )


def bet_summary(
    bets: pd.DataFrame,
) -> dict[str, Any]:

    if bets.empty:

        return {
            "bets": 0,
            "wins": 0,
            "hit_rate": None,
            "roi_proxy": None,
            "avg_odds": None,
            "avg_edge": None,
        }

    work = bets.copy()

    work[
        "odds_num"
    ] = pd.to_numeric(
        work[
            "odds_num"
        ],
        errors="coerce",
    )

    work[
        "finish_position"
    ] = pd.to_numeric(
        work[
            "finish_position"
        ],
        errors="coerce",
    )

    work[
        "edge"
    ] = pd.to_numeric(
        work[
            "edge"
        ],
        errors="coerce",
    )

    work = work[
        work[
            "odds_num"
        ].gt(
            1.0
        )
        &
        work[
            "finish_position"
        ].notna()
    ].copy()

    if work.empty:

        return {
            "bets": 0,
            "wins": 0,
            "hit_rate": None,
            "roi_proxy": None,
            "avg_odds": None,
            "avg_edge": None,
        }

    hits = (
        work[
            "finish_position"
        ]
        .eq(
            1
        )
        .astype(
            int
        )
    )

    gross = np.where(
        hits.to_numpy(
            dtype=int
        )
        == 1,

        work[
            "odds_num"
        ].to_numpy(
            dtype=float
        ),

        0.0,
    ).sum()

    stake = len(
        work
    )

    return {

        "bets":
            int(
                stake
            ),

        "wins":
            int(
                hits.sum()
            ),

        "hit_rate":
            float(
                hits.mean()
            ),

        "roi_proxy":
            float(
                gross
                / stake
                - 1.0
            ),

        "avg_odds":
            safe_float(
                work[
                    "odds_num"
                ].mean()
            ),

        "avg_edge":
            safe_float(
                work[
                    "edge"
                ].mean()
            ),
    }


def discipline_summary(
    bets: pd.DataFrame,
) -> dict[str, Any]:

    if (
        bets.empty
        or
        "discipline"
        not in bets.columns
    ):
        return {}

    result = {}

    for (
        discipline,
        group,
    ) in bets.groupby(
        "discipline",
        dropna=False,
    ):

        label = (
            "INCONNU"
            if pd.isna(
                discipline
            )
            else str(
                discipline
            )
        )

        result[
            label
        ] = bet_summary(
            group
        )

    return result


def top1_edge_bets(
    runners: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:

    rows = []

    for _, group in runners.groupby(
        "race_id",
        sort=False,
    ):

        group = group.sort_values(
            [
                "model_p",
                "place_p",
            ],
            ascending=False,
        )

        if group.empty:
            continue

        top = group.iloc[
            0
        ]

        odds = safe_float(
            top.get(
                "odds_num"
            )
        )

        edge = safe_float(
            top.get(
                "edge"
            )
        )

        if (
            odds is not None
            and odds > 1.0
            and edge is not None
            and edge >= threshold
        ):

            rows.append(
                top.to_dict()
            )

    return pd.DataFrame(
        rows
    )


def calendar_windows(
    runners: pd.DataFrame,
    n_windows: int = 4,
) -> list[
    tuple[
        pd.Timestamp,
        pd.Timestamp,
    ]
]:

    dates = pd.to_datetime(
        runners[
            "race_date"
        ],
        errors="coerce",
    ).dropna()

    if dates.empty:
        return []

    start = (
        dates
        .min()
        .normalize()
    )

    end = (
        dates
        .max()
        .normalize()
    )

    total_days = (
        end
        - start
    ).days + 1

    cuts = np.floor(
        np.linspace(
            0,
            total_days,
            n_windows + 1,
        )
    ).astype(
        int
    )

    windows = []

    for i in range(
        n_windows
    ):

        a = (
            start
            + pd.Timedelta(
                days=int(
                    cuts[i]
                )
            )
        )

        b = (
            start
            + pd.Timedelta(
                days=int(
                    cuts[
                        i + 1
                    ]
                    - 1
                )
            )
        )

        if b >= a:

            windows.append(
                (
                    a,
                    min(
                        b,
                        end,
                    ),
                )
            )

    return windows


def race_block_bootstrap(
    bets: pd.DataFrame,
) -> dict[str, Any]:

    if bets.empty:

        return {
            "samples": 0,
            "roi_median": None,
            "roi_ci95_low": None,
            "roi_ci95_high": None,
            "probability_roi_positive": None,
        }

    work = bets.copy()

    work[
        "odds_num"
    ] = pd.to_numeric(
        work[
            "odds_num"
        ],
        errors="coerce",
    )

    work[
        "finish_position"
    ] = pd.to_numeric(
        work[
            "finish_position"
        ],
        errors="coerce",
    )

    work = work[
        work[
            "odds_num"
        ].gt(
            1.0
        )
        &
        work[
            "finish_position"
        ].notna()
    ].copy()

    if work.empty:

        return {
            "samples": 0,
            "roi_median": None,
            "roi_ci95_low": None,
            "roi_ci95_high": None,
            "probability_roi_positive": None,
        }

    work[
        "gross"
    ] = np.where(
        work[
            "finish_position"
        ].eq(
            1
        ),

        work[
            "odds_num"
        ],

        0.0,
    )

    blocks = (
        work
        .groupby(
            "race_id",
            as_index=False,
        )
        .agg(
            stake=(
                "race_id",
                "size",
            ),
            gross=(
                "gross",
                "sum",
            ),
        )
    )

    stake = blocks[
        "stake"
    ].to_numpy(
        dtype=float
    )

    gross = blocks[
        "gross"
    ].to_numpy(
        dtype=float
    )

    n = len(
        blocks
    )

    rng = (
        np.random.default_rng(
            RANDOM_STATE
        )
    )

    rois = []

    for _ in range(
        BOOTSTRAP_SAMPLES
    ):

        take = rng.integers(
            0,
            n,
            size=n,
        )

        sampled_stake = (
            stake[
                take
            ].sum()
        )

        if sampled_stake > 0:

            rois.append(
                gross[
                    take
                ].sum()
                / sampled_stake
                - 1.0
            )

    values = np.asarray(
        rois,
        dtype=float,
    )

    return {

        "samples":
            int(
                len(
                    values
                )
            ),

        "roi_median":
            float(
                np.median(
                    values
                )
            ),

        "roi_ci95_low":
            float(
                np.quantile(
                    values,
                    0.025,
                )
            ),

        "roi_ci95_high":
            float(
                np.quantile(
                    values,
                    0.975,
                )
            ),

        "probability_roi_positive":
            float(
                np.mean(
                    values
                    > 0
                )
            ),
    }


def make_test_runners(
    model: HorseRacingModel,
    history: pd.DataFrame,
) -> pd.DataFrame:

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
        .astype(
            str
        )
        .isin(
            test_races
        )
    ].copy()

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
            test[
                "race_id"
            ],
        )
    )

    p_place = (
        model._apply_calibrator(
            raw_place,
            model.place_calibrator,
        )
    )

    odds = pd.to_numeric(
        test[
            "odds"
        ],
        errors="coerce",
    )

    implied = (
        1.0
        / odds.clip(
            lower=1.01
        )
    ).replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    market_p = (
        model._normalize_by_race(
            implied
            .fillna(
                0.01
            )
            .to_numpy(
                dtype=float
            ),
            test[
                "race_id"
            ],
        )
    )

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
        runners[
            "model_p"
        ]
        - runners[
            "market_p"
        ]
    )

    runners[
        "odds_num"
    ] = odds

    runners[
        "race_date"
    ] = pd.to_datetime(
        runners[
            "race_date"
        ],
        errors="coerce",
    ).dt.normalize()

    return runners


def build_report(
    model: HorseRacingModel,
    history: pd.DataFrame,
) -> dict[str, Any]:

    runners = make_test_runners(
        model,
        history,
    )

    all_bets = runners[
        pd.to_numeric(
            runners[
                "edge"
            ],
            errors="coerce",
        ).ge(
            EDGE_THRESHOLD
        )
        &
        pd.to_numeric(
            runners[
                "odds_num"
            ],
            errors="coerce",
        ).gt(
            1.0
        )
    ].copy()

    top1_bets = top1_edge_bets(
        runners,
        EDGE_THRESHOLD,
    )

    windows = []

    for (
        index,
        (
            start,
            end,
        ),
    ) in enumerate(
        calendar_windows(
            runners
        ),
        start=1,
    ):

        window_runners = runners[
            runners[
                "race_date"
            ].between(
                start,
                end,
                inclusive="both",
            )
        ].copy()

        window_all = all_bets[
            all_bets[
                "race_date"
            ].between(
                start,
                end,
                inclusive="both",
            )
        ].copy()

        if top1_bets.empty:

            window_top1 = (
                top1_bets.copy()
            )

        else:

            top1_dates = (
                pd.to_datetime(
                    top1_bets[
                        "race_date"
                    ],
                    errors="coerce",
                )
                .dt.normalize()
            )

            window_top1 = (
                top1_bets[
                    top1_dates
                    .between(
                        start,
                        end,
                        inclusive="both",
                    )
                ]
                .copy()
            )

        windows.append(
            {

                "window":
                    index,

                "start":
                    str(
                        start.date()
                    ),

                "end":
                    str(
                        end.date()
                    ),

                "races":
                    int(
                        window_runners[
                            "race_id"
                        ].nunique()
                    ),

                "runners":
                    int(
                        len(
                            window_runners
                        )
                    ),

                "all_runners_edge_2pp":
                    bet_summary(
                        window_all
                    ),

                "top1_edge_2pp":
                    bet_summary(
                        window_top1
                    ),

                "by_discipline":
                    discipline_summary(
                        window_all
                    ),
            }
        )

    rois = [
        window[
            "all_runners_edge_2pp"
        ][
            "roi_proxy"
        ]
        for window in windows
        if window[
            "all_runners_edge_2pp"
        ][
            "roi_proxy"
        ]
        is not None
    ]

    return {

        "version":
            "backtest_v3_robustness",

        "edge_threshold":
            EDGE_THRESHOLD,

        "test_start":
            str(
                runners[
                    "race_date"
                ]
                .min()
                .date()
            ),

        "test_end":
            str(
                runners[
                    "race_date"
                ]
                .max()
                .date()
            ),

        "test_races":
            int(
                runners[
                    "race_id"
                ].nunique()
            ),

        "test_runners":
            int(
                len(
                    runners
                )
            ),

        "overall": {

            "all_runners_edge_2pp":
                bet_summary(
                    all_bets
                ),

            "top1_edge_2pp":
                bet_summary(
                    top1_bets
                ),

            "by_discipline":
                discipline_summary(
                    all_bets
                ),
        },

        "temporal_windows":
            windows,

        "stability": {

            "positive_roi_windows":
                int(
                    sum(
                        roi > 0
                        for roi in rois
                    )
                ),

            "total_windows":
                int(
                    len(
                        rois
                    )
                ),

            "mean_window_roi_proxy":
                safe_float(
                    np.mean(
                        rois
                    )
                    if rois
                    else None
                ),

            "median_window_roi_proxy":
                safe_float(
                    np.median(
                        rois
                    )
                    if rois
                    else None
                ),

            "std_window_roi_proxy":
                safe_float(
                    np.std(
                        rois
                    )
                    if rois
                    else None
                ),

            "worst_window_roi_proxy":
                safe_float(
                    min(
                        rois
                    )
                    if rois
                    else None
                ),

            "best_window_roi_proxy":
                safe_float(
                    max(
                        rois
                    )
                    if rois
                    else None
                ),

            "race_block_bootstrap":
                race_block_bootstrap(
                    all_bets
                ),
        },

        "note": (
            "ROI proxy uniquement : les cotes historiques "
            "ne garantissent pas un prix réellement disponible "
            "au moment de la mise."
        ),
    }


def main() -> None:

    started_at = datetime.now(
        timezone.utc
    )

    print(
        "=" * 70
    )

    print(
        "HORSEPRONO - BACKTEST V3 ROBUSTESSE TEMPORELLE"
    )

    print(
        "=" * 70
    )

    active = (
        get_latest_active_model()
    )

    if not active:

        raise SystemExit(
            "Aucun modèle actif dans Supabase."
        )

    if (
        list(
            active.get(
                "features"
            )
            or []
        )
        != list(
            FEATURE_COLUMNS
        )
    ):

        raise SystemExit(
            "Le modèle actif et le code courant "
            "n'utilisent pas les mêmes features."
        )

    model_id = int(
        active[
            "id"
        ]
    )

    print(
        f"Modèle actif : "
        f"#{model_id} "
        f"{active.get('model_name')}"
    )

    history = (
        get_training_history()
    )

    if history.empty:

        raise SystemExit(
            "Historique vide."
        )

    print(
        f"Historique : "
        f"{len(history)} participants"
    )

    # L'artefact stocké en production est BT + logit.
    # Pour reproduire exactement l'ensemble offline avec
    # Gradient Boosting, on réentraîne le même modèle
    # déterministe uniquement en mémoire.
    model = (
        HorseRacingModel()
    )

    fit_metrics = (
        model.fit(
            history
        )
    )

    active_metrics = (
        active.get(
            "metrics"
        )
        or {}
    )

       # ========================================================
    # CONTROLE DE REPRODUCTION
    # ========================================================

    expected = safe_float(
        active_metrics.get(
            "win_logloss"
        )
    )

    reproduced = safe_float(
        fit_metrics.get(
            "win_logloss"
        )
    )

    gap = (
        abs(
            expected
            - reproduced
        )
        if (
            expected is not None
            and reproduced is not None
        )
        else None
    )

    relative_gap = (
        gap
        / abs(
            expected
        )
        if (
            gap is not None
            and expected not in (
                None,
                0,
            )
        )
        else None
    )

    # --------------------------------------------------------
    # Vérification structurelle du holdout
    # --------------------------------------------------------

    structural_keys = [
        "train_rows",
        "calibration_rows",
        "test_rows",
        "test_races",
    ]

    structural_match = True

    for key in structural_keys:

        active_value = (
            active_metrics.get(
                key
            )
        )

        reproduced_value = (
            fit_metrics.get(
                key
            )
        )

        try:

            active_value = int(
                active_value
            )

            reproduced_value = int(
                reproduced_value
            )

        except (
            TypeError,
            ValueError,
        ):

            structural_match = False

            print(
                f"Structure {key}: "
                f"valeur invalide"
            )

            continue

        same = (
            active_value
            == reproduced_value
        )

        structural_match = (
            structural_match
            and same
        )

        print(
            f"Structure {key}: "
            f"{active_value} / "
            f"{reproduced_value} "
            f"=> "
            f"{'OK' if same else 'DIFF'}"
        )

    # --------------------------------------------------------
    # Vérification des dates test
    # --------------------------------------------------------

    active_start = str(
        active_metrics.get(
            "test_start"
        )
    )

    reproduced_start = str(
        fit_metrics.get(
            "test_start"
        )
    )

    active_end = str(
        active_metrics.get(
            "test_end"
        )
    )

    reproduced_end = str(
        fit_metrics.get(
            "test_end"
        )
    )

    dates_match = (
        active_start
        == reproduced_start
        and
        active_end
        == reproduced_end
    )

    # --------------------------------------------------------
    # Tolérance numérique
    # --------------------------------------------------------
    #
    # On ne cherche pas une égalité bit-à-bit :
    # sklearn / NumPy peuvent produire de très petites
    # variations flottantes entre deux exécutions.
    #
    # 1e-4 est largement inférieur à une variation
    # économiquement ou statistiquement significative ici.
    # --------------------------------------------------------

    REPRO_TOLERANCE = 1e-4

    numerical_match = (
        gap is not None
        and gap <= REPRO_TOLERANCE
    )

    print()
    print(
        f"Win log-loss enregistré : "
        f"{expected}"
    )

    print(
        f"Win log-loss reproduit   : "
        f"{reproduced}"
    )

    print(
        f"Écart absolu             : "
        f"{gap}"
    )

    print(
        f"Écart relatif            : "
        f"{relative_gap}"
    )

    print(
        f"Tolérance                 : "
        f"{REPRO_TOLERANCE}"
    )

    print(
        f"Période enregistrée       : "
        f"{active_start} -> {active_end}"
    )

    print(
        f"Période reproduite        : "
        f"{reproduced_start} -> "
        f"{reproduced_end}"
    )

    if not structural_match:

        raise SystemExit(
            "Le découpage train/calibration/test "
            "ne correspond pas au modèle actif. "
            "Backtest interrompu."
        )

    if not dates_match:

        raise SystemExit(
            "La période de test ne correspond pas "
            "au modèle actif. "
            "Backtest interrompu."
        )

    if not numerical_match:

        raise SystemExit(
            "L'écart numérique avec le modèle actif "
            "dépasse la tolérance. "
            "Backtest interrompu."
        )

    print()
    print(
        "✅ Reproduction compatible avec "
        "le modèle actif."
    )
    advanced = (
        build_advanced_backtest(
            model,
            history,
        )
    )

    v2_reference = (
        advanced
        .get(
            "edge_strategies",
            {},
        )
        .get(
            "all_runners",
            {},
        )
        .get(
            "edge_2pp",
            {},
        )
    )

    report = build_report(
        model,
        history,
    )

    print()
    print(
        "RÉFÉRENCE BACKTEST V2 EDGE >= 2 PP"
    )

    print(
        json.dumps(
            v2_reference,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        "FENÊTRES TEMPORELLES"
    )

    for window in report[
        "temporal_windows"
    ]:

        stats = window[
            "all_runners_edge_2pp"
        ]

        print(
            f"{window['start']} "
            f"-> {window['end']} | "
            f"{stats['bets']} paris | "
            f"{stats['wins']} gagnants | "
            f"ROI proxy "
            f"{stats['roi_proxy']}"
        )

    print()
    print(
        "STABILITÉ"
    )

    print(
        json.dumps(
            report[
                "stability"
            ],
            indent=2,
            ensure_ascii=False,
        )
    )

    finished_at = datetime.now(
        timezone.utc
    )

    save_backtest_run(
        {

            "model_version_id":
                model_id,

            "started_at":
                started_at.isoformat(),

            "finished_at":
                finished_at.isoformat(),

            "metrics": {

                "backtest_v3_robustness":
                    report,

                "backtest_v2_edge_2pp_reference":
                    v2_reference,

                "reproduction_check": {

                    "active_win_logloss":
                        expected,

                    "reproduced_win_logloss":
                        reproduced,

                    "absolute_gap":
                        gap,
                },
            },

            "notes": (
                "Backtest V3 de robustesse temporelle. "
                "Réentraînement uniquement en mémoire ; "
                "aucune nouvelle version modèle créée."
            ),
        }
    )

    print()
    print(
        f"Backtest V3 enregistré "
        f"pour le modèle #{model_id}."
    )

    print(
        "Aucun nouveau modèle "
        "n'a été créé ni activé."
    )


if __name__ == "__main__":
    main()
