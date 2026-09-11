from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from app_core.config import FEATURE_COLUMNS
from app_core.data import example_race, normalize_race_input, read_uploaded_csv
from app_core.db import (
    get_history_as_of, get_latest_active_model, get_participants, get_training_history,
    healthcheck, is_configured, list_model_versions, list_races, save_prediction_batch,
)
from app_core.features import enrich_live_race
from app_core.io_utils import dataframe_to_csv_bytes
from app_core.model import HorseRacingModel
from app_core.pmu import get_participants as pmu_get_participants
from app_core.pmu import get_programme, participants_to_df, programme_choices

st.set_page_config(page_title="HorseProno V3", page_icon="🏇", layout="wide")

DISCLAIMER = """⚠️ **Pronos probabilistes, pas certitudes.** Une probabilité élevée n'est pas une garantie de gain. Les paris hippiques comportent un risque de perte, et le modèle peut se tromper, notamment lorsque les données historiques sont incomplètes ou que les conditions de course changent."""


def _safe_int(v):
    try:
        return int(v) if pd.notna(v) else None
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def cached_programme(d: date):
    return get_programme(d)


@st.cache_data(ttl=300, show_spinner=False)
def cached_races(race_date: date | None = None):
    return list_races(race_date, 300)


@st.cache_data(ttl=300, show_spinner=False)
def cached_participants(race_id: str):
    return get_participants(race_id)


@st.cache_data(ttl=600, show_spinner=False)
def cached_history_as_of(race_date: date):
    return get_history_as_of(race_date)


@st.cache_data(ttl=600, show_spinner=False)
def cached_active_model():
    return get_latest_active_model()


@st.cache_resource
def fallback_model():
    return HorseRacingModel()


def render_predictions(result: pd.DataFrame, model_version_id: int | None = None, race_id: str | None = None):
    result = result.copy()
    display = result[[
        "rank", "horse_number", "horse_name", "odds", "win_probability", "place_probability",
        "market_probability", "expected_value", "recommendation", "explanation"
    ]].rename(columns={
        "rank": "Rang", "horse_number": "N°", "horse_name": "Cheval", "odds": "Cote",
        "win_probability": "P(victoire)", "place_probability": "P(placé)", "market_probability": "P(marché)",
        "expected_value": "EV théorique", "recommendation": "Signal", "explanation": "Lecture"
    })
    st.dataframe(display, width="stretch", hide_index=True, column_config={
        "P(victoire)": st.column_config.ProgressColumn("P(victoire)", format="%.1f%%", min_value=0, max_value=1),
        "P(placé)": st.column_config.NumberColumn(format="%.1f%%"),
        "P(marché)": st.column_config.NumberColumn(format="%.1f%%"),
        "EV théorique": st.column_config.NumberColumn(format="%.2f"),
        "Cote": st.column_config.NumberColumn(format="%.2f"),
    })
    best = result.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 N°", f"{_safe_int(best['horse_number']) or '?'}")
    c2.metric("P(victoire)", f"{best['win_probability']*100:.1f}%")
    c3.metric("P(placé)", f"{best['place_probability']*100:.1f}%")
    c4.metric("EV théorique", f"{best['expected_value']:+.2f}")

    st.caption("**EV théorique** = P(victoire) × cote − 1. Il s'agit d'une estimation mathématique avant frais, limites, variation de cote et erreurs de modèle; ce n'est pas une recommandation de mise.")
    st.download_button("⬇️ Télécharger le pronostic CSV", dataframe_to_csv_bytes(result), "horseprono_v3.csv", "text/csv")
    if race_id and is_configured() and st.checkbox("Enregistrer ce pronostic dans Supabase", value=False):
        if st.button("💾 Enregistrer", type="secondary"):
            try:
                count = save_prediction_batch(race_id, model_version_id, result)
                st.success(f"{count} lignes enregistrées.")
            except Exception as exc:
                st.error(f"Enregistrement impossible : {exc}")


def train_preview(history: pd.DataFrame):
    with st.spinner("Entraînement V3 et validation temporelle…"):
        model = HorseRacingModel()
        metrics = model.fit(history)
    return model, metrics


st.title("🏇 HorseProno V3 — moteur quantitatif")
st.write("**Architecture :** PMU/CSV → Supabase → features temporelles → Bradley-Terry + régression logistique → challenger Gradient Boosting → calibration → backtest.")
st.info(DISCLAIMER)

with st.sidebar:
    ok, msg = healthcheck()
    (st.success if ok else st.warning)(msg)
    if st.button("🔄 Vider le cache"):
        st.cache_data.clear()
        st.rerun()

active_record = None
model = fallback_model()
model_id = None
if is_configured():
    try:
        active_record = cached_active_model()
        if active_record:
            model = HorseRacingModel.from_stored_record(active_record)
            model_id = active_record.get("id")
    except Exception as exc:
        st.warning(f"Modèle actif invalide ou incompatible : {exc}")

if active_record:
    m = active_record.get("metrics", {}) or {}
    st.caption(f"Modèle actif **#{active_record['id']}** · test {m.get('test_start', '?')} → {m.get('test_end', '?')} · Top1 {m.get('top1_accuracy', 0)*100:.1f}% · ROI flat top1 {m.get('flat_stake_roi_top1', 0)*100:.1f}%")
else:
    st.caption("Aucun modèle V3 actif : le moteur de repli est utilisé. Les statistiques de repli ne sont pas équivalentes à un modèle entraîné.")


tab_prono, tab_manual, tab_history, tab_model = st.tabs(["🎯 Pronostiquer", "✍️ Course manuelle", "📚 Historique", "📈 Modèles & backtest"])

with tab_prono:
    st.subheader("Sélection de la course")
    source = st.radio("Source", ["Supabase", "API PMU", "Démonstration"], horizontal=True)
    race = pd.DataFrame()
    race_id = None

    if source == "Supabase":
        if not is_configured():
            st.warning("Supabase doit être configuré pour cette source.")
        else:
            races = cached_races()
            if races.empty:
                st.info("Aucune course en base. Lance l'ETL ou importe ton historique.")
            else:
                options = races.apply(lambda r: f"{r.get('race_date')} · R{r.get('reunion') or '?'}C{r.get('course_number') or '?'} · {r.get('hippodrome','INCONNU')} · {r.get('race_id')}", axis=1).tolist()
                ix = st.selectbox("Course", range(len(options)), format_func=lambda i: options[i])
                race_id = str(races.iloc[ix]["race_id"])
                if st.button("Charger les partants", type="primary"):
                    try:
                        race = cached_participants(race_id)
                        st.session_state["selected_race"] = race
                        st.session_state["selected_race_id"] = race_id
                    except Exception as exc:
                        st.error(f"Lecture impossible : {exc}")
                elif st.session_state.get("selected_race_id") == race_id:
                    race = st.session_state.get("selected_race", pd.DataFrame())

    elif source == "API PMU":
        d = st.date_input("Date", date.today(), min_value=date.today()-timedelta(days=30), max_value=date.today()+timedelta(days=2))
        if st.button("Récupérer le programme"):
            try:
                st.session_state["pmu_programme"] = cached_programme(d)
                st.session_state["pmu_date"] = d
            except Exception as exc:
                st.error(f"PMU indisponible : {exc}")
        if st.session_state.get("pmu_date") == d:
            choices = programme_choices(st.session_state.get("pmu_programme"))
            if choices:
                labels = [f"R{x['reunion']} / C{x['course']} — {x['label']}" for x in choices]
                ix = st.selectbox("Course PMU", range(len(labels)), format_func=lambda i: labels[i])
                ch = choices[ix]
                if st.button("Charger cette course", type="primary"):
                    try:
                        race = participants_to_df( pmu_get_participants(d , ch["reunion"], ch["course"], ) , d, ch["reunion"], ch["course"], )  if not race.empty: 
    metadata = { "discipline": ch.get(  "discipline" ),   "hippodrome": ch.get(  "hippodrome"), "distance":  ch.get(
                "distance" ), "terrain": ch.get(    "terrain" ),  "field_size":  ch.get(  "field_size" ), }

    for column, value in metadata.items():

        if value is not None:
            race[
                column
            ] = value

    race_id = race.iloc[
        0
    ][
        "race_id"
    ]

else:

    race_id = None
                    except Exception as exc:
                        st.error(f"Impossible de récupérer les partants : {exc}")
            else:
                st.warning("Impossible d'interpréter le programme PMU retourné.")

    else:
        race = example_race()
        race_id = None

    if not race.empty:
        race = normalize_race_input(race)
        # Enrichir avec les stats connues avant la date cible.
        if is_configured():
            try:
                h = cached_history_as_of(pd.Timestamp(race["race_date"].iloc[0]).date())
                race = enrich_live_race(race, h)
            except Exception as exc:
                st.caption(f"Enrichissement historique non disponible : {exc}")
        st.dataframe(race[["horse_number", "horse_name", "age", "sex", "jockey", "trainer", "odds", "draw", "weight", "recent_form"]], width="stretch", hide_index=True)
        if st.button("🧠 Calculer le pronostic", type="primary"):
            try:
                result = model.predict(race)
                render_predictions(result, model_id, race_id)
            except Exception as exc:
                st.error(f"Prédiction impossible : {exc}")

with tab_manual:
    st.subheader("Entrée manuelle")
    uploaded = st.file_uploader("CSV de la course", type=["csv"])
    if uploaded:
        try:
            manual = read_uploaded_csv(uploaded)
            st.dataframe(manual, width="stretch", hide_index=True)
            if st.button("Pronostiquer le CSV", type="primary"):
                render_predictions(model.predict(manual), model_id, None)
        except Exception as exc:
            st.error(f"CSV invalide : {exc}")
    else:
        manual = example_race().drop(columns=["finish_position"])
        manual["race_id"] = "manual-race"
        edited = st.data_editor(manual, num_rows="dynamic", width="stretch")
        if st.button("Pronostiquer la course saisie", type="primary"):
            try:
                render_predictions(model.predict(normalize_race_input(edited)), model_id, None)
            except Exception as exc:
                st.error(f"Calcul impossible : {exc}")

with tab_history:
    st.subheader("Qualité et volume des données")
    if not is_configured():
        st.info("Configure Supabase pour afficher les métriques réelles.")
    else:
        try:
            hist = get_training_history(200_000)
            if hist.empty:
                st.info("📭 Aucun historique n'est encore présent dans Supabase. La connexion fonctionne, mais la base contient 0 course et 0 partant. Lance l'importateur historique avant l'entraînement.")
                st.caption("Étape suivante : importer les courses historiques, puis revenir ici pour vérifier le volume avant de lancer le modèle V3.")
            else:
                a, b, c, d = st.columns(4)
                a.metric("Partants terminés", f"{len(hist):,}".replace(",", " "))
                b.metric("Courses", f"{hist['race_id'].nunique():,}".replace(",", " "))
                c.metric("Chevaux", f"{hist['horse_name'].nunique():,}".replace(",", " "))
                d.metric("Taux résultat", f"{hist['finish_position'].notna().mean()*100:.1f}%")
                missing = hist[FEATURE_COLUMNS].isna().mean() if set(FEATURE_COLUMNS).issubset(hist.columns) else pd.Series(dtype=float)
                st.caption("Les features V3 temporelles sont reconstruites à partir des résultats antérieurs ; elles ne sont donc pas stockées comme vérité métier dans le CSV.")
                st.dataframe(hist.head(100), width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"Impossible de lire l'historique : {exc}")

with tab_model:
    st.subheader("Versions du modèle")
    if not is_configured():
        st.info("Configure Supabase pour voir les versions.")
    else:
        versions = list_model_versions(30)
        if not versions.empty:
            def metric_value(row, key):
                m = row.get("metrics") or {}
                return m.get(key)
            versions["test_top1"] = versions.apply(lambda r: metric_value(r, "top1_accuracy"), axis=1)
            versions["ROI top1"] = versions.apply(lambda r: metric_value(r, "flat_stake_roi_top1"), axis=1)
            st.dataframe(versions[["id", "trained_at", "model_type", "test_top1", "ROI top1", "is_active", "artifact_hash"]], width="stretch", hide_index=True)
        st.markdown("#### Lecture du backtest")
        st.write("**Top-1** mesure si le cheval classé n°1 par le modèle gagne. **Top-3** mesure si au moins un des trois premiers du modèle finit dans les trois premiers. **ROI flat** simule une mise uniforme de 1 unité sur le n°1 de chaque course du jeu de test. Ce n'est pas une garantie d'exécution réelle des mises.")
        st.caption("La comparaison avec le marché utilise la probabilité implicite normalisée des cotes disponibles. Une différence modèle-marché est un écart statistique, pas une certitude d'edge exploitable.")

st.divider()
st.caption("HorseProno V3 · Données externes soumises à leurs conditions d'utilisation. Aucune stratégie ne garantit un gain.")
