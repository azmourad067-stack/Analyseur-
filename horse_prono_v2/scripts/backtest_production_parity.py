from __future__ import annotations

import json
import os
import sys

from datetime import (
    date,
    datetime,
    timezone,
)

import numpy as np
import pandas as pd

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(__file__)
    ),
)


from app_core.db import (
    get_history_as_of,
    get_supabase_client,
    save_backtest_run,
)

from app_core.model import (
    HorseRacingModel,
)

from backtest_robustness import (
    bet_summary,
    calendar_windows,
    discipline_summary,
    make_test_runners,
    race_block_bootstrap,
    top1_edge_bets,
)


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

MODEL_VERSION_ID = 7

FROZEN_HISTORY_CUTOFF = date(
    2026,
    9,
    9,
)

EDGE_THRESHOLD = 0.02


# ============================================================
# MODEL
# ============================================================

def get_model_record() -> dict:

    client = get_supabase_client()

    if client is None:

        raise RuntimeError(
            "Supabase non configuré."
        )

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

    rows = (
        response.data
        or []
    )

    if not rows:

        raise RuntimeError(
            f"Modèle #{MODEL_VERSION_ID} "
            f"introuvable."
        )

    return rows[0]


# ============================================================
# REPORT
# ============================================================

def build_report(
    model: HorseRacingModel,
    history: pd.DataFrame,
) -> dict:

    runners = make_test_runners(
        model,
        history,
    )

    if runners.empty:

        raise RuntimeError(
            "Holdout vide."
        )

    # --------------------------------------------------------
    # Metrics probabilistes
    # --------------------------------------------------------

    y = (
        pd.to_numeric(
            runners[
                "finish_position"
            ],
            errors="coerce",
        )
        .eq(
            1
        )
        .astype(
            int
        )
        .to_numpy()
    )

    p = pd.to_numeric(
        runners[
            "model_p"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    market_p = pd.to_numeric(
        runners[
            "market_p"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    probabilistic = {

        "win_logloss":
            float(
                log_loss(
                    y,
                    p,
                    labels=[
                        0,
                        1,
                    ],
                )
            ),

        "win_brier":
            float(
                brier_score_loss(
                    y,
                    p,
                )
            ),

        "win_auc":
            (
                float(
                    roc_auc_score(
                        y,
                        p,
                    )
                )
                if len(
                    np.unique(
                        y
                    )
                ) == 2
                else None
            ),

        "market_win_logloss":
            float(
                log_loss(
                    y,
                    market_p,
                    labels=[
                        0,
                        1,
                    ],
                )
            ),

        "market_win_brier":
            float(
                brier_score_loss(
                    y,
                    market_p,
                )
            ),
    }

    # --------------------------------------------------------
    # EDGE >= 2 PP
    # --------------------------------------------------------

    edge = pd.to_numeric(
        runners[
            "edge"
        ],
        errors="coerce",
    )

    odds = pd.to_numeric(
        runners[
            "odds_num"
        ],
        errors="coerce",
    )

    all_bets = runners[
        edge.ge(
            EDGE_THRESHOLD
        )
        &
        odds.gt(
            1.0
        )
    ].copy()

    top1_bets = top1_edge_bets(
        runners,
        EDGE_THRESHOLD,
    )

    # --------------------------------------------------------
    # Fenêtres temporelles
    # --------------------------------------------------------

    temporal_windows = []

    for (
        index,
        (
            start,
            end,
        ),
    ) in enumerate(
        calendar_windows(
            runners,
            4,
        ),
        start=1,
    ):

        mask = (
            runners[
                "race_date"
            ]
            .between(
                start,
                end,
                inclusive="both",
            )
        )

        window_runners = (
            runners[
                mask
            ]
            .copy()
        )

        all_mask = (
            all_bets[
                "race_date"
            ]
            .between(
                start,
                end,
                inclusive="both",
            )
        )

        window_all = (
            all_bets[
                all_mask
            ]
            .copy()
        )

        if top1_bets.empty:

            window_top1 = (
                top1_bets.copy()
            )

        else:

            top1_dates = pd.to_datetime(
                top1_bets[
                    "race_date"
                ],
                errors="coerce",
            )

            top1_mask = (
                top1_dates
                .between(
                    start,
                    end,
                    inclusive="both",
                )
            )

            window_top1 = (
                top1_bets[
                    top1_mask
                ]
                .copy()
            )

        temporal_windows.append(
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
                        ]
                        .nunique()
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

    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    bootstrap = (
        race_block_bootstrap(
            all_bets
        )
    )

    return {

        "version":
            "production_parity_v1",

        "model_version_id":
            MODEL_VERSION_ID,

        "prediction_mode":
            (
                "portable production artifact "
                "BT + Logistic"
            ),

        "frozen_history_cutoff":
            FROZEN_HISTORY_CUTOFF.isoformat(),

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
                ]
                .nunique()
            ),

        "test_runners":
            int(
                len(
                    runners
                )
            ),

        "probabilistic":
            probabilistic,

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
            temporal_windows,

        "bootstrap":
            bootstrap,

        "note": (
            "Test de parité avec le modèle réellement "
            "chargé en production. ROI = proxy historique."
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    started_at = datetime.now(
        timezone.utc
    )

    print(
        "=" * 70
    )

    print(
        "HORSEPRONO - PRODUCTION PARITY BACKTEST"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Modèle #7 stocké
    # --------------------------------------------------------

    record = (
        get_model_record()
    )

    print(
        f"Modèle : "
        f"#{record['id']} "
        f"{record.get('model_name')}"
    )

    print(
        f"Artifact hash : "
        f"{record.get('artifact_hash')}"
    )

    model = (
        HorseRacingModel
        .from_stored_record(
            record
        )
    )

    if not getattr(
        model,
        "portable_artifact",
        False,
    ):

        raise RuntimeError(
            "Le modèle chargé n'est pas "
            "l'artefact portable de production."
        )

    # --------------------------------------------------------
    # Historique figé
    # --------------------------------------------------------

    print()
    print(
        "Chargement de l'historique figé..."
    )

    history = get_history_as_of(
        FROZEN_HISTORY_CUTOFF
    )

    print(
        f"Historique : "
        f"{len(history)} participants"
    )

    if len(
        history
    ) != 49229:

        raise RuntimeError(
            "Le dataset figé ne contient plus "
            "49 229 lignes. "
            "Test interrompu."
        )

    # --------------------------------------------------------
    # Backtest
    # --------------------------------------------------------

    report = build_report(
        model,
        history,
    )

    overall = (
        report[
            "overall"
        ][
            "all_runners_edge_2pp"
        ]
    )

    top1 = (
        report[
            "overall"
        ][
            "top1_edge_2pp"
        ]
    )

    print()
    print(
        "=" * 70
    )

    print(
        "PRODUCTION - EDGE >= 2 PP"
    )

    print(
        "=" * 70
    )

    print(
        f"Tous chevaux : "
        f"{overall['bets']} paris / "
        f"{overall['wins']} gagnants / "
        f"hit {overall['hit_rate']} / "
        f"ROI {overall['roi_proxy']}"
    )

    print(
        f"Top1 seulement : "
        f"{top1['bets']} paris / "
        f"{top1['wins']} gagnants / "
        f"hit {top1['hit_rate']} / "
        f"ROI {top1['roi_proxy']}"
    )

    print()
    print(
        "METRICS PROBABILISTES"
    )

    print(
        json.dumps(
            report[
                "probabilistic"
            ],
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
            f"ROI {stats['roi_proxy']}"
        )

    print()
    print(
        "BOOTSTRAP"
    )

    print(
        json.dumps(
            report[
                "bootstrap"
            ],
            indent=2,
            ensure_ascii=False,
        )
    )

    # --------------------------------------------------------
    # Enregistrement Supabase
    # --------------------------------------------------------

    finished_at = datetime.now(
        timezone.utc
    )

    save_backtest_run(
        {

            "model_version_id":
                MODEL_VERSION_ID,

            "started_at":
                started_at.isoformat(),

            "finished_at":
                finished_at.isoformat(),

            "metrics": {
                "production_parity":
                    report
            },

            "notes": (
                "Backtest de parité production : "
                "artefact portable exact du modèle #7. "
                "Aucun modèle créé ou modifié."
            ),
        }
    )

    print()
    print(
        "Backtest de parité enregistré."
    )

    print(
        "Aucun nouveau modèle créé."
    )


if __name__ == "__main__":
    main()
