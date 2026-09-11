from __future__ import annotations

from datetime import datetime

import pandas as pd

from neural.predict import (
    MODEL_PATH,
    load_neural_v1,
    predict_dataframe,
    sha256_file,
)


# ============================================================
# HORSEPRONO - FORWARD DUEL
# MODELE #8 VS NEURAL #9
# ============================================================

NEURAL_MODEL_VERSION_ID = 9

NEURAL_MODEL_NAME = (
    "horseprono_neural_v1"
)

NEURAL_ARTIFACT_HASH = (
    "a2d77a6767e2bfa3258d3fad98173f0bec217c9562a5a6b983651257b8d320b8"
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


def _safe_float(value):

    parsed = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(parsed):
        return None

    return float(parsed)


def _safe_probability(value):

    value = _safe_float(value)

    if value is None:
        return None

    return float(
        min(
            max(value, 0.0),
            1.0,
        )
    )


def _safe_int(value):

    parsed = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(parsed):
        return None

    return int(parsed)


def _by_horse_number(
    df: pd.DataFrame,
) -> dict[int, pd.Series]:

    mapping = {}

    if (
        df is None
        or df.empty
        or "horse_number" not in df.columns
    ):
        return mapping

    numbers = pd.to_numeric(
        df["horse_number"],
        errors="coerce",
    )

    for index, number in numbers.items():

        if pd.isna(number):
            continue

        mapping[int(number)] = (
            df.loc[index]
        )

    return mapping


# ============================================================
# VERIFICATION MODELE FIGE
# ============================================================

def load_forward_neural(
    client,
):

    rows = _rows(
        client
        .table("model_versions")
        .select(
            "id,model_name,"
            "artifact_hash"
        )
        .eq(
            "id",
            NEURAL_MODEL_VERSION_ID,
        )
        .limit(1)
        .execute()
    )

    if not rows:

        raise RuntimeError(
            "Modèle Neural #9 "
            "introuvable dans Supabase."
        )

    record = rows[0]

    if (
        str(record.get("model_name"))
        != NEURAL_MODEL_NAME
    ):

        raise RuntimeError(
            "Le modèle #9 ne correspond "
            "pas à Neural V1."
        )

    database_hash = str(
        record.get(
            "artifact_hash"
        )
        or ""
    )

    if (
        database_hash
        != NEURAL_ARTIFACT_HASH
    ):

        raise RuntimeError(
            "Hash Neural Supabase incorrect : "
            f"{database_hash}"
        )

    local_hash = sha256_file(
        MODEL_PATH
    )

    if (
        local_hash
        != NEURAL_ARTIFACT_HASH
    ):

        raise RuntimeError(
            "Le fichier Neural local "
            "n'est pas le modèle gelé.\n"
            f"Attendu : {NEURAL_ARTIFACT_HASH}\n"
            f"Trouvé  : {local_hash}"
        )

    bundle = load_neural_v1()

    print(
        "Neural forward chargé : "
        f"#{NEURAL_MODEL_VERSION_ID} "
        f"{NEURAL_MODEL_NAME}"
    )

    print(
        "Hash Neural : "
        f"{local_hash[:12]}..."
    )

    return bundle


# ============================================================
# PREDICTIONS DEJA CAPTUREES
# ============================================================

def _existing_prediction_keys(
    client,
    experiment_id: int,
    race_id: int,
    model_ids: list[int],
) -> set[tuple[int, int]]:

    rows = _rows(
        client
        .table(
            "forward_model_predictions"
        )
        .select(
            "participant_id,"
            "model_version_id"
        )
        .eq(
            "experiment_id",
            experiment_id,
        )
        .eq(
            "race_id",
            race_id,
        )
        .in_(
            "model_version_id",
            model_ids,
        )
        .execute()
    )

    return {
        (
            int(row["participant_id"]),
            int(row["model_version_id"]),
        )
        for row in rows
    }


# ============================================================
# FORMAT TOP 3
# ============================================================

def _top3_text(
    df: pd.DataFrame,
    rank_column: str,
    probability_column: str,
) -> str:

    if df.empty:
        return "aucun"

    ordered = (
        df
        .sort_values(
            rank_column,
            ascending=True,
        )
        .head(3)
    )

    values = []

    for _, row in ordered.iterrows():

        number = _safe_int(
            row.get("horse_number")
        )

        name = str(
            row.get("horse_name")
            or "INCONNU"
        )

        probability = (
            _safe_probability(
                row.get(
                    probability_column
                )
            )
        )

        if probability is None:

            probability_text = "?"

        else:

            probability_text = (
                f"{probability:.1%}"
            )

        values.append(
            f"N°{number} {name} "
            f"({probability_text})"
        )

    return " | ".join(values)


# ============================================================
# CAPTURE DU DUEL
# ============================================================

def capture_forward_comparison(
    client,
    experiment: dict,
    race: pd.DataFrame,
    internal_race_id: int,
    participant_map: dict[int, dict],
    classical_predictions: pd.DataFrame,
    scheduled_start: datetime,
    captured_at: datetime,
    neural_bundle: dict,
    non_runner_numbers: set[int] | None = None,
) -> dict:

    experiment_id = int(
        experiment["id"]
    )

    model8_id = int(
        experiment[
            "model_version_id"
        ]
    )

    model8_hash = str(
        experiment[
            "artifact_hash"
        ]
    )

    non_runner_numbers = {
        int(value)
        for value in (
            non_runner_numbers
            or set()
        )
    }

    # ========================================================
    # RETIRER LES NON-PARTANTS DU DUEL
    # ========================================================

    race_numbers = pd.to_numeric(
        race["horse_number"],
        errors="coerce",
    )

    active_mask = ~race_numbers.map(
        lambda value: (
            int(value)
            in non_runner_numbers
            if pd.notna(value)
            else True
        )
    )

    active_race = (
        race[
            active_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    if active_race.empty:

        return {
            "written": 0,
            "model8": 0,
            "neural": 0,
        }

    # ========================================================
    # MODELE #8
    # ========================================================

    classical = (
        classical_predictions
        .copy()
    )

    classical_numbers = (
        pd.to_numeric(
            classical[
                "horse_number"
            ],
            errors="coerce",
        )
    )

    classical = classical[
        ~classical_numbers.map(
            lambda value: (
                int(value)
                in non_runner_numbers
                if pd.notna(value)
                else True
            )
        )
    ].copy()

    # Recalcul des rangs sur les
    # seuls partants actifs.

    classical[
        "duel_rank_win"
    ] = (
        pd.to_numeric(
            classical[
                "win_probability"
            ],
            errors="coerce",
        )
        .rank(
            method="first",
            ascending=False,
        )
        .astype("Int64")
    )

    classical[
        "duel_rank_place"
    ] = (
        pd.to_numeric(
            classical[
                "place_probability"
            ],
            errors="coerce",
        )
        .rank(
            method="first",
            ascending=False,
        )
        .astype("Int64")
    )

    # ========================================================
    # NEURAL #9
    # ========================================================

    neural = predict_dataframe(
        active_race,
        neural_bundle,
    )

    # ========================================================
    # INDEX PAR NUMERO
    # ========================================================

    race_map = _by_horse_number(
        active_race
    )

    model8_map = _by_horse_number(
        classical
    )

    neural_map = _by_horse_number(
        neural
    )

    # ========================================================
    # EVITER LES DOUBLONS
    # ========================================================

    existing = (
        _existing_prediction_keys(
            client,
            experiment_id,
            internal_race_id,
            [
                model8_id,
                NEURAL_MODEL_VERSION_ID,
            ],
        )
    )

    inserts = []

    model8_count = 0
    neural_count = 0

    # ========================================================
    # MEME TIMESTAMP POUR LES DEUX MODELES
    # ========================================================

    predicted_at = (
        captured_at.isoformat()
    )

    scheduled_start_iso = (
        scheduled_start.isoformat()
    )

    # ========================================================
    # PARCOURS DES MEMES CHEVAUX
    # ========================================================

    for horse_number in sorted(
        participant_map.keys()
    ):

        if (
            horse_number
            in non_runner_numbers
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
            participant["id"]
        )

        race_row = (
            race_map.get(
                horse_number
            )
        )

        model8_row = (
            model8_map.get(
                horse_number
            )
        )

        neural_row = (
            neural_map.get(
                horse_number
            )
        )

        if race_row is None:
            continue

        horse_name = str(
            race_row.get(
                "horse_name"
            )
            or participant.get(
                "horse_name"
            )
            or "INCONNU"
        )

        discipline = str(
            race_row.get(
                "discipline"
            )
            or "INCONNU"
        )

        # ----------------------------------------------------
        # COTE SNAPSHOT
        # ----------------------------------------------------

        odds_snapshot = (
            _safe_float(
                race_row.get(
                    "odds"
                )
            )
        )

        if (
            odds_snapshot is not None
            and odds_snapshot <= 1.0
        ):
            odds_snapshot = None

        market_probability = None

        if model8_row is not None:

            market_probability = (
                _safe_probability(
                    model8_row.get(
                        "market_probability"
                    )
                )
            )

        if (
            market_probability
            is None
            and odds_snapshot
            is not None
        ):

            market_probability = (
                1.0
                / odds_snapshot
            )

        # ====================================================
        # MODELE #8
        # ====================================================

        key8 = (
            participant_id,
            model8_id,
        )

        if (
            model8_row is not None
            and key8 not in existing
        ):

            inserts.append(
                {
                    "experiment_id":
                        experiment_id,

                    "race_id":
                        internal_race_id,

                    "participant_id":
                        participant_id,

                    "model_version_id":
                        model8_id,

                    "artifact_hash":
                        model8_hash,

                    "scheduled_start":
                        scheduled_start_iso,

                    "predicted_at":
                        predicted_at,

                    "discipline":
                        discipline,

                    "horse_name":
                        horse_name,

                    "horse_number":
                        horse_number,

                    "odds_snapshot":
                        odds_snapshot,

                    "market_probability":
                        market_probability,

                    "win_probability":
                        _safe_probability(
                            model8_row.get(
                                "win_probability"
                            )
                        ),

                    "place_probability":
                        _safe_probability(
                            model8_row.get(
                                "place_probability"
                            )
                        ),

                    "rank_win":
                        _safe_int(
                            model8_row.get(
                                "duel_rank_win"
                            )
                        ),

                    "rank_place":
                        _safe_int(
                            model8_row.get(
                                "duel_rank_place"
                            )
                        ),
                }
            )

            existing.add(
                key8
            )

            model8_count += 1

        # ====================================================
        # NEURAL #9
        # ====================================================

        key9 = (
            participant_id,
            NEURAL_MODEL_VERSION_ID,
        )

        if (
            neural_row is not None
            and key9 not in existing
        ):

            inserts.append(
                {
                    "experiment_id":
                        experiment_id,

                    "race_id":
                        internal_race_id,

                    "participant_id":
                        participant_id,

                    "model_version_id":
                        NEURAL_MODEL_VERSION_ID,

                    "artifact_hash":
                        NEURAL_ARTIFACT_HASH,

                    "scheduled_start":
                        scheduled_start_iso,

                    "predicted_at":
                        predicted_at,

                    "discipline":
                        discipline,

                    "horse_name":
                        horse_name,

                    "horse_number":
                        horse_number,

                    "odds_snapshot":
                        odds_snapshot,

                    "market_probability":
                        market_probability,

                    # Neural V1 prédit Top3,
                    # pas directement gagnant.
                    "win_probability":
                        None,

                    "place_probability":
                        _safe_probability(
                            neural_row.get(
                                "neural_top3_probability"
                            )
                        ),

                    "rank_win":
                        None,

                    "rank_place":
                        _safe_int(
                            neural_row.get(
                                "neural_rank"
                            )
                        ),
                }
            )

            existing.add(
                key9
            )

            neural_count += 1

    # ========================================================
    # INSERT SUPABASE
    # ========================================================

    if inserts:

        response = (
            client
            .table(
                "forward_model_predictions"
            )
            .insert(
                inserts
            )
            .execute()
        )

        written = (
            len(_rows(response))
            or len(inserts)
        )

    else:

        written = 0

    # ========================================================
    # AFFICHAGE DU DUEL
    # ========================================================

    print(
        "  Duel #8 Top3 : "
        + _top3_text(
            classical,
            "duel_rank_place",
            "place_probability",
        )
    )

    print(
        "  Duel #9 Top3 : "
        + _top3_text(
            neural,
            "neural_rank",
            "neural_top3_probability",
        )
    )

    print(
        "  Snapshots duel enregistrés : "
        f"#8={model8_count} | "
        f"#9={neural_count}"
    )

    return {
        "written":
            written,

        "model8":
            model8_count,

        "neural":
            neural_count,
    }
