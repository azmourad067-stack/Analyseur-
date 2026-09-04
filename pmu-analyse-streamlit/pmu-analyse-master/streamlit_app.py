import os
import sys
import logging
from datetime import datetime, date, timedelta
from logging.config import fileConfig

import pandas as pd
import streamlit as st

# Chemin racine du projet (repos GitHub / Streamlit Cloud)
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# Logging
try:
    fileConfig(os.path.join(ROOT, 'logger', 'logging_config.ini'))
except Exception:
    logging.basicConfig(level=logging.INFO)

from database.setup_database import engine
from database.database import _session
from scrapping.scrapping import call_api_between_dates

st.set_page_config(page_title="PMU Analyse", page_icon="🐎", layout="wide")

# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_reunions():
    return pd.read_sql("SELECT * FROM pmu_reunions ORDER BY dateReunion DESC, numOfficiel", engine)


@st.cache_data(ttl=300, show_spinner=False)
def load_courses():
    return pd.read_sql("SELECT * FROM pmu_courses ORDER BY heureDepart DESC", engine)


@st.cache_data(ttl=300, show_spinner=False)
def load_participants():
    return pd.read_sql("SELECT * FROM pmu_participants ORDER BY dateReunion DESC, numReunion, numOrdre, numPmu", engine)


@st.cache_data(ttl=300, show_spinner=False)
def load_hippodromes():
    return pd.read_sql("SELECT * FROM pmu_hippodromes ORDER BY libelleCourt", engine)


def invalidate():
    load_reunions.clear()
    load_courses.clear()
    load_participants.clear()
    load_hippodromes.clear()


def fmt_euros(v):
    try:
        return f"{int(v):,}".replace(",", " ") + " €"
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------------------
# Barre latérale : navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🐎 PMU Analyse")
page = st.sidebar.radio(
    "Navigation",
    ["📥 Collecte des données", "🗂️ Explorer la base", "📊 Analyses", "ℹ️ À propos"],
)

# ---------------------------------------------------------------------------
# Page 1 : Collecte des données
# ---------------------------------------------------------------------------
if page == "📥 Collecte des données":
    st.title("📥 Collecte des données PMU")
    st.caption("Récupère les programmes (réunions, courses, participants) depuis l'API publique PMU turfinfo.")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Date de début", value=date.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("Date de fin", value=date.today())

    if start_date > end_date:
        st.error("La date de début doit être antérieure à la date de fin.")
    else:
        if st.button("🚀 Lancer la collecte", type="primary"):
            progress_bar = st.progress(0.0)
            status = st.empty()

            def progress_callback(i, total, current_date, reunion_number):
                pct = min(i / total, 1.0)
                progress_bar.progress(pct)
                status.info(f"Jour {i + 1}/{total} — {current_date.strftime('%d/%m/%Y')} — Réunion R{reunion_number}…")

            try:
                counts = call_api_between_dates(
                    datetime.combine(start_date, datetime.min.time()),
                    datetime.combine(end_date, datetime.min.time()),
                    progress_callback=progress_callback,
                )
                progress_bar.progress(1.0)
                invalidate()
                st.success(
                    f"Collecte terminée : {counts['reunions']} réunions, "
                    f"{counts['courses']} courses, {counts['participants']} participants enregistrés."
                )
            except Exception as exc:
                st.error(f"Erreur pendant la collecte : {exc}")

    st.divider()
    st.subheader("État de la base")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Réunions", len(load_reunions()))
    c2.metric("Courses", len(load_courses()))
    c3.metric("Participants", len(load_participants()))
    c4.metric("Hippodromes", len(load_hippodromes()))

# ---------------------------------------------------------------------------
# Page 2 : Explorer la base
# ---------------------------------------------------------------------------
elif page == "🗂️ Explorer la base":
    st.title("🗂️ Explorer la base")
    tab_reunions, tab_courses, tab_participants = st.tabs(["Réunions", "Courses", "Participants"])

    with tab_reunions:
        df = load_reunions()
        if df.empty:
            st.info("Aucune réunion en base. Lancez une collecte.")
        else:
            cols = [c for c in ["dateReunion", "numOfficiel", "nature", "statut", "hippodrome_code", "pays_code"] if c in df.columns]
            st.dataframe(df[cols], use_container_width=True, hide_index=True)

    with tab_courses:
        df = load_courses()
        if df.empty:
            st.info("Aucune course en base.")
        else:
            cols = [c for c in ["heureDepart", "numReunion", "numOrdre", "libelle", "distance", "discipline", "specialite", "hippodrome_code", "nombreDeclaresPartants"] if c in df.columns]
            st.dataframe(df[cols], use_container_width=True, hide_index=True)

    with tab_participants:
        df = load_participants()
        if df.empty:
            st.info("Aucun participant en base.")
        else:
            cols = [c for c in ["dateReunion", "numReunion", "numOrdre", "numPmu", "nom", "age", "sexe", "entraineur", "driver", "statut", "gainsCarriere"] if c in df.columns]
            st.dataframe(df[cols], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Page 3 : Analyses
# ---------------------------------------------------------------------------
elif page == "📊 Analyses":
    st.title("📊 Analyses")
    participants = load_participants()
    courses = load_courses()

    if participants.empty:
        st.info("Aucune donnée participant en base. Lancez une collecte pour alimenter les analyses.")
    else:
        # Filtres
        f1, f2 = st.columns(2)
        with f1:
            disciplines = ["Toutes"] + sorted(courses["discipline"].dropna().unique().tolist()) if "discipline" in courses.columns else ["Toutes"]
            discipline = st.selectbox("Discipline", disciplines)
        with f2:
            hippos = ["Tous"] + sorted(participants["hippodrome_code"].dropna().unique().tolist()) if "hippodrome_code" in participants.columns else ["Tous"]
            hippo = st.selectbox("Hippodrome", hippos)

        df = participants.copy()
        if discipline != "Toutes" and "discipline" in courses.columns:
            course_ids = courses[courses["discipline"] == discipline][["numReunion", "numOrdre"]]
            df = df.merge(course_ids, on=["numReunion", "numOrdre"], how="inner")
        if hippo != "Tous":
            df = df[df["hippodrome_code"] == hippo]

        st.subheader("Vue d'ensemble")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Participants", len(df))
        m2.metric("Victoires cumulées", int(df["nombreVictoires"].fillna(0).sum()))
        m3.metric("Places cumulées", int(df["nombrePlaces"].fillna(0).sum()))
        m4.metric("Gains carrière (total)", fmt_euros(df["gainsCarriere"].fillna(0).sum()))

        # Top entraineurs
        st.subheader("Top 10 entraîneurs (par gains de carrière)")
        if "entraineur" in df.columns:
            top = (df.groupby("entraineur")
                     .agg(nb_participants=("nom", "count"),
                          victoires=("nombreVictoires", "sum"),
                          gains=("gainsCarriere", "sum"))
                     .sort_values("gains", ascending=False)
                     .head(10))
            top["gains"] = top["gains"].apply(fmt_euros)
            st.dataframe(top, use_container_width=True)

        # Top chevaux
        st.subheader("Top 10 chevaux (par gains de carrière)")
        top_horses = (df.groupby("nom")
                        .agg(nb_courses=("nom", "count"),
                             victoires=("nombreVictoires", "max"),
                             gains=("gainsCarriere", "max"))
                        .sort_values("gains", ascending=False)
                        .head(10))
        top_horses["gains"] = top_horses["gains"].apply(fmt_euros)
        st.dataframe(top_horses, use_container_width=True)

        # Répartition par discipline
        st.subheader("Répartition des participants par discipline")
        if "discipline" in courses.columns:
            dist = (df.merge(courses[["numReunion", "numOrdre", "discipline"]], on=["numReunion", "numOrdre"], how="left")
                      .groupby("discipline").size().sort_values(ascending=False))
            st.bar_chart(dist)

# ---------------------------------------------------------------------------
# Page 4 : À propos
# ---------------------------------------------------------------------------
else:
    st.title("ℹ️ À propos")
    st.markdown(
        """
        **PMU Analyse** est un outil d'aide à la décision pour les paris hippiques.

        - **Source de données** : API publique PMU *turfinfo* (programmes officiels).
        - **Stockage** : base SQLite locale (`database/db/pmu_data.db`), gérée via SQLAlchemy.
        - **Fonctionnalités** : collecte des réunions/courses/participants sur une période,
          exploration de la base et analyses statistiques (gains, victoires, entraîneurs…).

        ⚠️ *Les données fournies ne constituent pas un conseil de jeu. Jouez de manière responsable.*
        """
    )
