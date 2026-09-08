from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from app_core.config import REQUIRED_HISTORY_COLUMNS
from app_core.data import example_race, normalize_history, normalize_race_input, read_uploaded_csv
from app_core.db import (
    get_latest_active_model,
    get_participants as db_get_participants,
    get_training_history,
    healthcheck,
    is_configured,
    save_prediction_batch,
    upsert_participants,
    upsert_races,
)
from app_core.io_utils import dataframe_to_csv_bytes
from app_core.model import HorseRacingModel
from app_core.pmu import get_participants as pmu_get_participants
from app_core.pmu import get_programme, participants_to_df, programme_choices

st.set_page_config(page_title="HorseProno Data Platform", page_icon="🏇", layout="wide")

DISCLAIMER = (
    "⚠️ **Important** — les probabilités sont des estimations statistiques et non des certitudes. "
    "Elles ne garantissent ni victoire ni gain. Une cote, une forme ou un score élevé peut être démenti "
    "par les événements réels de la course."
)

BASE = Path(__file__).resolve().parent


@st.cache_data(ttl=600, show_spinner=False)
def cached_programme(d: date):
    return get_programme(d)


@st.cache_data(ttl=300, show_spinner=False)
def cached_pmu_participants(d: date, r: int, c: int):
    return participants_to_df(pmu_get_participants(d, r, c), d, r, c)


@st.cache_data(ttl=300, show_spinner=False)
def cached_active_model():
    return get_latest_active_model()


@st.cache_data(ttl=300, show_spinner=False)
def cached_races(race_date: date | None):
    from app_core.db import list_races
    return list_races(race_date, limit=200)


@st.cache_data(ttl=300, show_spinner=False)
def cached_db_participants(race_id: str):
    return db_get_participants(race_id)


@st.cache_resource
def get_fallback_model():
    return HorseRacingModel()


def make_race_record(df: pd.DataFrame, source: str) -> dict:
    first = df.iloc[0]
    return {
        "race_id": str(first.get("race_id")),
        "race_date": pd.Timestamp(first.get("race_date", pd.Timestamp.today())).date().isoformat(),
        "reunion": _int(first.get("reunion")),
        "course_number": _int(first.get("course_number")),
        "discipline": str(first.get("discipline") or "INCONNU"),
        "hippodrome": str(first.get("hippodrome") or "INCONNU"),
        "distance": _int(first.get("distance")),
        "terrain": str(first.get("terrain") or "INCONNU"),
        "field_size": _int(first.get("field_size")) or len(df),
        "source": source,
    }


def _int(v):
    try:
        return int(v) if pd.notna(v) else None
    except Exception:
        return None


def render_result(result: pd.DataFrame, persisted_race_id: str | None = None, model_version_id: int | None = None):
    st.subheader("Pronostic classé")
    cols = ["rank", "horse_number", "horse_name", "odds", "recent_form", "draw", "weight", "win_probability", "place_probability", "method"]
    cols = [c for c in cols if c in result.columns]
    display = result[cols].rename(columns={
        "rank": "Rang", "horse_number": "N°", "horse_name": "Cheval", "odds": "Cote", "recent_form": "Forme",
        "draw": "Corde", "weight": "Poids", "win_probability": "Prob. victoire", "place_probability": "Prob. placé", "method": "Méthode",
    })
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Prob. victoire": st.column_config.NumberColumn(format="%.1f%%"),
            "Prob. placé": st.column_config.NumberColumn(format="%.1f%%"),
            "Cote": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    best = result.iloc[0]
    st.success(f"🏆 N°{_int(best.get('horse_number')) or '?'} — {best.get('horse_name', '')} en tête du classement statistique.")
    st.metric("Probabilité victoire du favori modèle", f"{float(best['win_probability'])*100:.1f}%")

    if persisted_race_id and is_configured() and bool(st.secrets.get("ENABLE_PUBLIC_PREDICTION_WRITES", False)):
        if st.button("💾 Enregistrer ce pronostic dans Supabase", type="secondary"):
            try:
                count = save_prediction_batch(persisted_race_id, model_version_id, result)
                st.success(f"{count} prédictions enregistrées.")
            except Exception as exc:
                st.error(f"Impossible d'enregistrer le pronostic : {exc}")

    st.download_button("⬇️ Télécharger le pronostic CSV", dataframe_to_csv_bytes(result), "pronostic.csv", "text/csv")


st.title("🏇 HorseProno — Data Platform & Pronostics probabilistes")
st.caption("Courses → Supabase → historique → entraînement hors ligne → modèle versionné → pronostic.")
st.info(DISCLAIMER)

with st.sidebar:
    st.header("Architecture")
    ok, msg = healthcheck()
    if ok:
        st.success("Supabase connecté")
    else:
        st.warning(msg)
    st.caption("Le modèle actif est lu depuis `model_versions`. L'entraînement de production se fait via le job CI/CD, pas dans la session utilisateur.")
    if st.button("🔄 Actualiser les données en cache"):
        st.cache_data.clear()
        st.rerun()

if not is_configured():
    st.warning("Mode local/fallback : configure Supabase pour activer la base persistante et le modèle versionné.")

active_record = None
active_model = get_fallback_model()
model_version_id = None
if is_configured():
    try:
        active_record = cached_active_model()
        if active_record:
            active_model = HorseRacingModel.from_stored_record(active_record)
            model_version_id = active_record.get("id")
    except Exception as exc:
        st.warning(f"Modèle actif indisponible, utilisation du score de repli : {exc}")

if active_record:
    metrics = active_record.get("metrics", {}) or {}
    st.caption(f"Modèle actif : version **{active_record.get('id')}** · hash `{str(active_record.get('artifact_hash',''))[:12]}` · validation {metrics.get('validation_start','?')} → {metrics.get('validation_end','?')}")
else:
    st.caption("Aucun modèle actif en base : un score probabiliste transparent est utilisé en attendant le premier entraînement.")


tab_course, tab_db, tab_manual, tab_ops = st.tabs(["🎯 Pronostiquer", "🗄️ Base de données", "✍️ Saisie manuelle", "⚙️ Opérations"])

with tab_course:
    st.subheader("1. Charger une course")
    if is_configured():
        races = cached_races(None)
    else:
        races = pd.DataFrame()

    source = st.radio("Source", ["Supabase", "API PMU", "Démonstration"], horizontal=True)
    race_df = None
    persisted_race_id = None

    if source == "Supabase":
        if races.empty:
            st.info("Aucune course en base. Lance l'ingestion PMU ou importe un historique.")
        else:
            labels = races.apply(lambda r: f"{r.get('race_date')} · R{r.get('reunion') or '?'}C{r.get('course_number') or '?'} · {r.get('hippodrome','INCONNU')} · {r.get('race_id')}", axis=1).tolist()
            idx = st.selectbox("Course enregistrée", range(len(labels)), format_func=lambda i: labels[i])
            race_id = str(races.iloc[idx]["race_id"])
            if st.button("Charger les partants depuis Supabase", type="primary"):
                try:
                    race_df = cached_db_participants(race_id)
                    persisted_race_id = race_id
                    st.session_state["race_df"] = race_df
                    st.session_state["race_id"] = race_id
                except Exception as exc:
                    st.error(f"Lecture impossible : {exc}")
            elif "race_df" in st.session_state and st.session_state.get("race_id") == race_id:
                race_df = st.session_state["race_df"]
                persisted_race_id = race_id

    elif source == "API PMU":
        selected_date = st.date_input("Date", value=date.today(), min_value=date.today()-timedelta(days=30), max_value=date.today()+timedelta(days=2))
        if st.button("Charger le programme PMU"):
            try:
                st.session_state["programme"] = cached_programme(selected_date)
                st.session_state["programme_date"] = selected_date
                st.success("Programme récupéré.")
            except Exception as exc:
                st.error(f"PMU indisponible : {exc}")
        if st.session_state.get("programme_date") == selected_date and "programme" in st.session_state:
            choices = programme_choices(st.session_state["programme"])
            if choices:
                labels = [f"R{x['reunion']}/C{x['course']} — {x['label']}" for x in choices]
                idx = st.selectbox("Course PMU", range(len(labels)), format_func=lambda i: labels[i])
                choice = choices[idx]
                if st.button("Charger les partants PMU", type="primary"):
                    try:
                        race_df = cached_pmu_participants(selected_date, choice["reunion"], choice["course"])
                        race_df["reunion"] = choice["reunion"]
                        race_df["course_number"] = choice["course"]
                        # Persist live course and runners when write permissions are available.
                        if is_configured():
                            upsert_races(pd.DataFrame([make_race_record(race_df, "pmu")]))
                            upsert_participants(race_df.drop(columns=["reunion", "course_number"], errors="ignore"))
                            persisted_race_id = str(race_df.iloc[0]["race_id"]) if not race_df.empty else None
                            st.cache_data.clear()
                    except Exception as exc:
                        st.error(f"Import PMU impossible : {exc}")
            else:
                st.warning("Le schéma du programme PMU n'a pas pu être interprété.")

    else:
        race_df = example_race()
        persisted_race_id = None

    if race_df is not None and not race_df.empty:
        st.subheader("2. Partants")
        shown = [c for c in ["horse_number","horse_name","jockey","trainer","odds","draw","weight","recent_form"] if c in race_df.columns]
        st.dataframe(race_df[shown], use_container_width=True, hide_index=True)
        if st.button("🔎 Calculer le pronostic", type="primary", key="predict_course"):
            try:
                result = active_model.predict(race_df)
                st.session_state["last_result"] = result
                st.session_state["last_race_id"] = persisted_race_id
            except Exception as exc:
                st.error(f"Calcul impossible : {exc}")

    if "last_result" in st.session_state:
        st.divider()
        render_result(st.session_state["last_result"], st.session_state.get("last_race_id"), model_version_id)

with tab_db:
    st.subheader("État de la base")
    if not is_configured():
        st.info("Ajoute les secrets Supabase pour afficher les données persistantes.")
    else:
        ok, msg = healthcheck()
        st.write(msg)
        if ok:
            races = cached_races(None)
            st.metric("Courses", len(races))
            if not races.empty:
                st.dataframe(races.head(100), use_container_width=True, hide_index=True)

with tab_manual:
    st.subheader("Saisie / import d'une course")
    st.caption("Ce mode reste disponible pour tester le moteur quand une source web est indisponible.")
    uploaded = st.file_uploader("CSV partants", type=["csv"], key="manual_csv")
    if uploaded:
        try:
            manual_df = normalize_race_input(read_uploaded_csv(uploaded))
            st.dataframe(manual_df, use_container_width=True, hide_index=True)
            if st.button("Pronostiquer le CSV", type="primary"):
                result = active_model.predict(manual_df)
                render_result(result)
        except Exception as exc:
            st.error(f"CSV invalide : {exc}")
    else:
        columns = ["horse_number","horse_name","jockey","trainer","odds","draw","weight","recent_form","career_runs","career_wins","career_places"]
        if "manual_editor_v2" not in st.session_state:
            st.session_state["manual_editor_v2"] = pd.DataFrame([[None, "", "", "", None, None, None, "", None, None, None] for _ in range(8)], columns=columns)
        edited = st.data_editor(st.session_state["manual_editor_v2"], num_rows="dynamic", use_container_width=True, hide_index=True, key="manual_v2")
        valid = edited[edited["horse_name"].fillna("").astype(str).str.strip().ne("")].copy()
        if not valid.empty and st.button("Pronostiquer la saisie", type="primary"):
            valid = normalize_race_input(valid.assign(
                race_id="MANUAL", race_date=pd.Timestamp.today().normalize(), discipline="INCONNU",
                hippodrome="SAISIE MANUELLE", distance=None, terrain="INCONNU", field_size=len(valid)
            ))
            render_result(active_model.predict(valid))

with tab_ops:
    st.subheader("Pipeline de données")
    st.markdown("""
**Production :**

```text
PMU / CSV historique
        ↓
      ETL Python
        ↓
 Supabase PostgreSQL
        ↓
 validation temporelle
        ↓
 régression logistique
        ↓
 artifact JSON versionné
        ↓
   Streamlit Cloud
```

L'application utilisateur ne sérialise pas de modèle Python dans la base : elle charge uniquement un artefact JSON composé des paramètres numériques du scaler et de la régression logistique. Cela réduit le risque associé au chargement de fichiers pickle provenant d'une source externe.
    """)

    st.markdown("### Entraînement")
    if is_configured():
        if st.button("📊 Vérifier l'historique disponible"):
            try:
                history = get_training_history()
                st.write(f"Lignes terminées disponibles : **{len(history):,}**")
                if not history.empty:
                    st.dataframe(history.head(20), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"Lecture historique impossible : {exc}")
    st.info("L'entraînement de production est volontairement séparé de l'interface et s'exécute par `scripts/train_model.py`, déclenchable par GitHub Actions. Cela rend les versions de modèle reproductibles.")

    st.markdown("### Import historique local")
    hist = st.file_uploader("CSV historique", type=["csv"], key="ops_history")
    if hist:
        try:
            h = normalize_history(read_uploaded_csv(hist))
            missing = [c for c in REQUIRED_HISTORY_COLUMNS if c not in h.columns]
            if missing:
                st.error(f"Colonnes indispensables absentes : {', '.join(missing)}")
            else:
                st.success(f"Historique valide : {len(h):,} lignes.")
                st.caption("Pour un import persistant, utilise `python scripts/import_history.py chemin.csv` avec SUPABASE_URL + SUPABASE_SERVICE_KEY.")
        except Exception as exc:
            st.error(f"Historique invalide : {exc}")

with st.expander("📐 Limites du modèle"):
    st.markdown("""
- La qualité dépend directement de l'historique, de sa couverture et de son absence de biais.
- Les variables disponibles après le départ ne doivent jamais servir de variables prédictives avant-course.
- Une régression logistique ne modélise pas à elle seule toute la dépendance entre les rangs d'une course ; une future version pourra ajouter Bradley-Terry / Plackett-Luce ou un modèle de learning-to-rank.
- Une bonne AUC ne signifie pas nécessairement une bonne calibration des probabilités ni une rentabilité des paris.
- Les cotes reflètent déjà une information de marché ; elles peuvent donc être très prédictives sans constituer une preuve de valeur attendue positive.
- Les incidents de course, stratégies tactiques, changements tardifs et informations non captées par les données restent imprévisibles.
    """)
