from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from app_core.config import FEATURE_COLUMNS
from app_core.data import (
    example_race,
    normalize_race_input,
    read_uploaded_csv,
)
from app_core.db import (
    get_history_as_of,
    get_latest_active_model,
    get_participants,
    get_training_history,
    healthcheck,
    is_configured,
    list_model_versions,
    list_races,
    save_prediction_batch,
)
from app_core.features import enrich_live_race
from app_core.io_utils import dataframe_to_csv_bytes
from app_core.model import HorseRacingModel
from app_core.pmu import (
    get_participants as pmu_get_participants,
)
from app_core.pmu import (
    get_programme,
    participants_to_df,
    programme_choices,
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="HorseProno V3",
    page_icon="🏇",
    layout="wide",
)


DISCLAIMER = """
⚠️ **Pronos probabilistes, pas certitudes.**
Une probabilité élevée n'est pas une garantie de gain.
Les paris hippiques comportent un risque de perte,
et le modèle peut se tromper, notamment lorsque les
données historiques sont incomplètes ou que les
conditions de course changent.
"""


# ============================================================
# OUTILS
# ============================================================

def _safe_int(value):

    try:

        if pd.notna(value):
            return int(value)

    except Exception:
        pass

    return None


# ============================================================
# CACHE
# ============================================================

@st.cache_data(
    ttl=600,
    show_spinner=False,
)
def cached_programme(
    d: date,
):

    return get_programme(d)


@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def cached_races(
    race_date: date | None = None,
):

    return list_races(
        race_date,
        300,
    )


@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def cached_participants(
    race_id: str,
):

    return get_participants(
        race_id
    )


@st.cache_data(
    ttl=600,
    show_spinner=False,
)
def cached_history_as_of(
    race_date: date,
):

    return get_history_as_of(
        race_date
    )


@st.cache_data(
    ttl=600,
    show_spinner=False,
)
def cached_active_model():

    return get_latest_active_model()


@st.cache_resource
def fallback_model():

    return HorseRacingModel()


# ============================================================
# AFFICHAGE PRONOSTICS
# ============================================================

def render_predictions(
    result: pd.DataFrame,
    model_version_id: int | None = None,
    race_id: str | None = None,
):

    result = result.copy()

    display = result[
        [
            "rank",
            "horse_number",
            "horse_name",
            "odds",
            "win_probability",
            "place_probability",
            "market_probability",
            "expected_value",
            "recommendation",
            "explanation",
        ]
    ].rename(
        columns={
            "rank":
                "Rang",

            "horse_number":
                "N°",

            "horse_name":
                "Cheval",

            "odds":
                "Cote",

            "win_probability":
                "P(victoire)",

            "place_probability":
                "P(placé)",

            "market_probability":
                "P(marché)",

            "expected_value":
                "EV théorique",

            "recommendation":
                "Signal",

            "explanation":
                "Lecture",
        }
    )

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={

            "P(victoire)":
                st.column_config.ProgressColumn(
                    "P(victoire)",
                    format="%.1f%%",
                    min_value=0,
                    max_value=1,
                ),

            "P(placé)":
                st.column_config.NumberColumn(
                    format="%.1f%%"
                ),

            "P(marché)":
                st.column_config.NumberColumn(
                    format="%.1f%%"
                ),

            "EV théorique":
                st.column_config.NumberColumn(
                    format="%.2f"
                ),

            "Cote":
                st.column_config.NumberColumn(
                    format="%.2f"
                ),
        },
    )

    if result.empty:
        return

    best = result.iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🏆 N°",
        f"{_safe_int(best['horse_number']) or '?'}",
    )

    c2.metric(
        "P(victoire)",
        f"{best['win_probability'] * 100:.1f}%",
    )

    c3.metric(
        "P(placé)",
        f"{best['place_probability'] * 100:.1f}%",
    )

    c4.metric(
        "EV théorique",
        f"{best['expected_value']:+.2f}",
    )

    st.caption(
        "**EV théorique** = "
        "P(victoire) × cote − 1. "
        "Il s'agit d'une estimation mathématique "
        "avant frais, limites, variation de cote "
        "et erreurs de modèle ; "
        "ce n'est pas une recommandation de mise."
    )

    st.download_button(
        "⬇️ Télécharger le pronostic CSV",
        dataframe_to_csv_bytes(
            result
        ),
        "horseprono_v3.csv",
        "text/csv",
    )

    if (
        race_id
        and is_configured()
        and st.checkbox(
            "Enregistrer ce pronostic dans Supabase",
            value=False,
        )
    ):

        if st.button(
            "💾 Enregistrer",
            type="secondary",
        ):

            try:

                count = save_prediction_batch(
                    race_id,
                    model_version_id,
                    result,
                )

                st.success(
                    f"{count} lignes enregistrées."
                )

            except Exception as exc:

                st.error(
                    "Enregistrement impossible : "
                    f"{exc}"
                )


# ============================================================
# PREVIEW ENTRAINEMENT
# ============================================================

def train_preview(
    history: pd.DataFrame,
):

    with st.spinner(
        "Entraînement V3 et validation temporelle…"
    ):

        model = HorseRacingModel()

        metrics = model.fit(
            history
        )

    return model, metrics


# ============================================================
# HEADER
# ============================================================

st.title(
    "🏇 HorseProno V3 — moteur quantitatif"
)

st.write(
    "**Architecture :** "
    "PMU/CSV → Supabase → "
    "features temporelles → "
    "Bradley-Terry + régression logistique → "
    "challenger Gradient Boosting → "
    "calibration → backtest."
)

st.info(
    DISCLAIMER
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    ok, msg = healthcheck()

    if ok:

        st.success(
            msg
        )

    else:

        st.warning(
            msg
        )

    if st.button(
        "🔄 Vider le cache"
    ):

        st.cache_data.clear()

        st.rerun()


# ============================================================
# MODELE ACTIF
# ============================================================

active_record = None

model = fallback_model()

model_id = None


if is_configured():

    try:

        active_record = (
            cached_active_model()
        )

        if active_record:

            model = (
                HorseRacingModel
                .from_stored_record(
                    active_record
                )
            )

            model_id = (
                active_record.get(
                    "id"
                )
            )

    except Exception as exc:

        st.warning(
            "Modèle actif invalide "
            "ou incompatible : "
            f"{exc}"
        )


if active_record:

    metrics = (
        active_record.get(
            "metrics",
            {},
        )
        or {}
    )

    st.caption(
        f"Modèle actif "
        f"**#{active_record['id']}** · "
        f"test "
        f"{metrics.get('test_start', '?')} "
        f"→ "
        f"{metrics.get('test_end', '?')} · "
        f"Top1 "
        f"{metrics.get('top1_accuracy', 0) * 100:.1f}% · "
        f"ROI flat top1 "
        f"{metrics.get('flat_stake_roi_top1', 0) * 100:.1f}%"
    )

else:

    st.caption(
        "Aucun modèle V3 actif : "
        "le moteur de repli est utilisé. "
        "Les statistiques de repli "
        "ne sont pas équivalentes "
        "à un modèle entraîné."
    )


# ============================================================
# ONGLETS
# ============================================================

(
    tab_prono,
    tab_manual,
    tab_history,
    tab_model,
) = st.tabs(
    [
        "🎯 Pronostiquer",
        "✍️ Course manuelle",
        "📚 Historique",
        "📈 Modèles & backtest",
    ]
)


# ============================================================
# ONGLET PRONOSTIC
# ============================================================

with tab_prono:

    st.subheader(
        "Sélection de la course"
    )

    source = st.radio(
        "Source",
        [
            "Supabase",
            "API PMU",
            "Démonstration",
        ],
        horizontal=True,
    )

    race = pd.DataFrame()

    race_id = None


    # ========================================================
    # SOURCE SUPABASE
    # ========================================================

    if source == "Supabase":

        if not is_configured():

            st.warning(
                "Supabase doit être configuré "
                "pour cette source."
            )

        else:

            races = cached_races()

            if races.empty:

                st.info(
                    "Aucune course en base. "
                    "Lance l'ETL ou importe "
                    "ton historique."
                )

            else:

                options = races.apply(
                    lambda row: (
                        f"{row.get('race_date')} · "
                        f"R{row.get('reunion') or '?'}"
                        f"C{row.get('course_number') or '?'} · "
                        f"{row.get('hippodrome', 'INCONNU')} · "
                        f"{row.get('race_id')}"
                    ),
                    axis=1,
                ).tolist()

                ix = st.selectbox(
                    "Course",
                    range(
                        len(options)
                    ),
                    format_func=lambda i: (
                        options[i]
                    ),
                )

                race_id = str(
                    races.iloc[ix][
                        "race_id"
                    ]
                )

                if st.button(
                    "Charger les partants",
                    type="primary",
                ):

                    try:

                        race = (
                            cached_participants(
                                race_id
                            )
                        )

                        st.session_state[
                            "selected_race"
                        ] = race

                        st.session_state[
                            "selected_race_id"
                        ] = race_id

                    except Exception as exc:

                        st.error(
                            "Lecture impossible : "
                            f"{exc}"
                        )

                elif (
                    st.session_state.get(
                        "selected_race_id"
                    )
                    == race_id
                ):

                    race = (
                        st.session_state.get(
                            "selected_race",
                            pd.DataFrame(),
                        )
                    )


    # ========================================================
    # SOURCE API PMU
    # ========================================================

    elif source == "API PMU":

        d = st.date_input(
            "Date",
            date.today(),
            min_value=(
                date.today()
                - timedelta(
                    days=30
                )
            ),
            max_value=(
                date.today()
                + timedelta(
                    days=2
                )
            ),
        )


        # ----------------------------------------------------
        # PROGRAMME PMU
        # ----------------------------------------------------

        if st.button(
            "Récupérer le programme"
        ):

            try:

                st.session_state[
                    "pmu_programme"
                ] = cached_programme(
                    d
                )

                st.session_state[
                    "pmu_date"
                ] = d

            except Exception as exc:

                st.error(
                    "PMU indisponible : "
                    f"{exc}"
                )


        # ----------------------------------------------------
        # COURSES PMU
        # ----------------------------------------------------

        if (
            st.session_state.get(
                "pmu_date"
            )
            == d
        ):

            choices = (
                programme_choices(
                    st.session_state.get(
                        "pmu_programme"
                    )
                )
            )

            if choices:

                labels = [

                    (
                        f"R{x['reunion']} / "
                        f"C{x['course']} — "
                        f"{x['label']}"
                    )

                    for x in choices
                ]

                ix = st.selectbox(
                    "Course PMU",
                    range(
                        len(labels)
                    ),
                    format_func=lambda i: (
                        labels[i]
                    ),
                )

                ch = choices[
                    ix
                ]

                selected_key = (
                    d.isoformat(),
                    int(
                        ch[
                            "reunion"
                        ]
                    ),
                    int(
                        ch[
                            "course"
                        ]
                    ),
                )


                # --------------------------------------------
                # CHARGEMENT COURSE
                # --------------------------------------------

                if st.button(
                    "Charger cette course",
                    type="primary",
                ):

                    try:

                        payload = (
                            pmu_get_participants(
                                d,
                                ch[
                                    "reunion"
                                ],
                                ch[
                                    "course"
                                ],
                            )
                        )

                        race = (
                            participants_to_df(
                                payload,
                                d,
                                ch[
                                    "reunion"
                                ],
                                ch[
                                    "course"
                                ],
                            )
                        )


                        # ------------------------------------
                        # METADONNEES PROGRAMME
                        # ------------------------------------

                        if not race.empty:

                            metadata = {

                                "discipline":
                                    ch.get(
                                        "discipline"
                                    ),

                                "hippodrome":
                                    ch.get(
                                        "hippodrome"
                                    ),

                                "distance":
                                    ch.get(
                                        "distance"
                                    ),

                                "terrain":
                                    ch.get(
                                        "terrain"
                                    ),

                                "field_size":
                                    ch.get(
                                        "field_size"
                                    ),
                            }


                            for (
                                column,
                                value,
                            ) in metadata.items():

                                if (
                                    value
                                    is not None
                                ):

                                    race[
                                        column
                                    ] = value


                            race_id = str(
                                race.iloc[0][
                                    "race_id"
                                ]
                            )


                            # ------------------------------
                            # MEMORISER LA COURSE
                            # ------------------------------

                            st.session_state[
                                "pmu_selected_race"
                            ] = race

                            st.session_state[
                                "pmu_selected_race_id"
                            ] = race_id

                            st.session_state[
                                "pmu_selected_key"
                            ] = selected_key


                        else:

                            st.warning(
                                "Aucun participant "
                                "trouvé pour cette course."
                            )


                    except Exception as exc:

                        st.error(
                            "Impossible de récupérer "
                            "les partants : "
                            f"{exc}"
                        )


                # --------------------------------------------
                # RESTAURER LA COURSE APRES RERUN STREAMLIT
                # --------------------------------------------

                if (
                    st.session_state.get(
                        "pmu_selected_key"
                    )
                    == selected_key
                ):

                    race = (
                        st.session_state.get(
                            "pmu_selected_race",
                            pd.DataFrame(),
                        )
                    )

                    race_id = (
                        st.session_state.get(
                            "pmu_selected_race_id"
                        )
                    )


            else:

                st.warning(
                    "Impossible d'interpréter "
                    "le programme PMU retourné."
                )


    # ========================================================
    # SOURCE DEMONSTRATION
    # ========================================================

    else:

        race = example_race()

        race_id = None


    # ========================================================
    # COURSE CHARGEE
    # ========================================================

    if not race.empty:

        race = normalize_race_input(
            race
        )


        # ----------------------------------------------------
        # ENRICHISSEMENT HISTORIQUE
        # ----------------------------------------------------

        if is_configured():

            try:

                target_date = (
                    pd.Timestamp(
                        race[
                            "race_date"
                        ].iloc[0]
                    )
                    .date()
                )

                history = (
                    cached_history_as_of(
                        target_date
                    )
                )

                race = (
                    enrich_live_race(
                        race,
                        history,
                    )
                )

            except Exception as exc:

                st.caption(
                    "Enrichissement historique "
                    "non disponible : "
                    f"{exc}"
                )


        # ----------------------------------------------------
        # TABLEAU PARTANTS
        # ----------------------------------------------------

        display_columns = [

            "horse_number",
            "horse_name",
            "age",
            "sex",
            "jockey",
            "trainer",
            "odds",
            "draw",
            "weight",
            "recent_form",

        ]

        available_columns = [

            column

            for column
            in display_columns

            if column
            in race.columns

        ]

        st.dataframe(
            race[
                available_columns
            ],
            width="stretch",
            hide_index=True,
        )


        # ----------------------------------------------------
        # PRONOSTIC
        # ----------------------------------------------------

        if st.button(
            "🧠 Calculer le pronostic",
            type="primary",
        ):

            try:

                result = model.predict(
                    race
                )

                render_predictions(
                    result,
                    model_id,
                    race_id,
                )

            except Exception as exc:

                st.error(
                    "Prédiction impossible : "
                    f"{exc}"
                )


# ============================================================
# ONGLET COURSE MANUELLE
# ============================================================

with tab_manual:

    st.subheader(
        "Entrée manuelle"
    )

    uploaded = st.file_uploader(
        "CSV de la course",
        type=[
            "csv"
        ],
    )


    if uploaded:

        try:

            manual = (
                read_uploaded_csv(
                    uploaded
                )
            )

            st.dataframe(
                manual,
                width="stretch",
                hide_index=True,
            )


            if st.button(
                "Pronostiquer le CSV",
                type="primary",
            ):

                result = model.predict(
                    manual
                )

                render_predictions(
                    result,
                    model_id,
                    None,
                )


        except Exception as exc:

            st.error(
                "CSV invalide : "
                f"{exc}"
            )


    else:

        manual = (
            example_race()
            .drop(
                columns=[
                    "finish_position"
                ]
            )
        )

        manual[
            "race_id"
        ] = "manual-race"


        edited = st.data_editor(
            manual,
            num_rows="dynamic",
            width="stretch",
        )


        if st.button(
            "Pronostiquer la course saisie",
            type="primary",
        ):

            try:

                prepared = (
                    normalize_race_input(
                        edited
                    )
                )

                result = model.predict(
                    prepared
                )

                render_predictions(
                    result,
                    model_id,
                    None,
                )

            except Exception as exc:

                st.error(
                    "Calcul impossible : "
                    f"{exc}"
                )


# ============================================================
# ONGLET HISTORIQUE
# ============================================================

with tab_history:

    st.subheader(
        "Qualité et volume des données"
    )


    if not is_configured():

        st.info(
            "Configure Supabase "
            "pour afficher "
            "les métriques réelles."
        )


    else:

        try:

            hist = (
                get_training_history(
                    200_000
                )
            )


            if hist.empty:

                st.info(
                    "📭 Aucun historique "
                    "n'est encore présent "
                    "dans Supabase."
                )


            else:

                a, b, c, d = (
                    st.columns(
                        4
                    )
                )


                a.metric(
                    "Partants terminés",
                    (
                        f"{len(hist):,}"
                        .replace(
                            ",",
                            " ",
                        )
                    ),
                )


                b.metric(
                    "Courses",
                    (
                        f"{hist['race_id'].nunique():,}"
                        .replace(
                            ",",
                            " ",
                        )
                    ),
                )


                c.metric(
                    "Chevaux",
                    (
                        f"{hist['horse_name'].nunique():,}"
                        .replace(
                            ",",
                            " ",
                        )
                    ),
                )


                d.metric(
                    "Taux résultat",
                    (
                        f"{hist['finish_position'].notna().mean() * 100:.1f}%"
                    ),
                )


                st.caption(
                    "Les features V3 temporelles "
                    "sont reconstruites à partir "
                    "des résultats antérieurs."
                )


                st.dataframe(
                    hist.head(
                        100
                    ),
                    width="stretch",
                    hide_index=True,
                )


        except Exception as exc:

            st.error(
                "Impossible de lire "
                "l'historique : "
                f"{exc}"
            )


# ============================================================
# ONGLET MODELES
# ============================================================

with tab_model:

    st.subheader(
        "Versions du modèle"
    )


    if not is_configured():

        st.info(
            "Configure Supabase "
            "pour voir les versions."
        )


    else:

        versions = (
            list_model_versions(
                30
            )
        )


        if not versions.empty:

            def metric_value(
                row,
                key,
            ):

                metrics = (
                    row.get(
                        "metrics"
                    )
                    or {}
                )

                return metrics.get(
                    key
                )


            versions[
                "test_top1"
            ] = versions.apply(
                lambda row: (
                    metric_value(
                        row,
                        "top1_accuracy",
                    )
                ),
                axis=1,
            )


            versions[
                "ROI top1"
            ] = versions.apply(
                lambda row: (
                    metric_value(
                        row,
                        "flat_stake_roi_top1",
                    )
                ),
                axis=1,
            )


            columns = [

                "id",
                "trained_at",
                "model_type",
                "test_top1",
                "ROI top1",
                "is_active",
                "artifact_hash",

            ]


            st.dataframe(
                versions[
                    columns
                ],
                width="stretch",
                hide_index=True,
            )


        st.markdown(
            "#### Lecture du backtest"
        )


        st.write(
            "**Top-1** mesure si le cheval "
            "classé n°1 par le modèle gagne. "
            "**Top-3** mesure si au moins un "
            "des trois premiers du modèle "
            "finit dans les trois premiers. "
            "**ROI flat** simule une mise "
            "uniforme de 1 unité sur le n°1 "
            "de chaque course du jeu de test."
        )


        st.caption(
            "La comparaison avec le marché "
            "utilise la probabilité implicite "
            "normalisée des cotes disponibles. "
            "Une différence modèle-marché "
            "est un écart statistique, "
            "pas une certitude d'edge exploitable."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "HorseProno V3 · "
    "Données externes soumises "
    "à leurs conditions d'utilisation. "
    "Aucune stratégie ne garantit un gain."
)
