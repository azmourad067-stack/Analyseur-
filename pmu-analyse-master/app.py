import threading
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

from logger.logging_setup import setup_logging

setup_logging()

from database.setup_database import engine  # noqa: E402  (doit venir après setup_logging)
from scrapping.scrapping import call_api_between_dates  # noqa: E402

MAX_DAYS = 14  # borne de sécurité : l'app est publique et partagée, on évite
               # un scraping trop long qui bloquerait l'instance pour tout le monde.

# Verrou global : empêche deux scrapings simultanés sur la même instance
# (SQLite gère mal les écritures concurrentes, et l'app Streamlit Cloud est
# partagée entre tous les visiteurs qui la chargent en même temps).
_SCRAPE_LOCK = threading.Lock()

st.set_page_config(page_title="PMU Analyse", page_icon="🐎", layout="wide")
st.title("🐎 PMU Analyse")
st.caption(
    "Collecte et exploration des programmes de courses PMU. "
    "⚠️ Application de démo : l'API PMU utilisée est publique et non officielle, "
    "elle peut changer ou être limitée sans préavis."
)

tab_scraping, tab_donnees = st.tabs(["📥 Scraper de nouvelles données", "📊 Explorer les données"])

# ----------------------------------------------------------------------------
# Onglet scraping
# ----------------------------------------------------------------------------
with tab_scraping:
    st.subheader("Lancer une collecte")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Date de début", value=date(2024, 1, 1), format="DD/MM/YYYY"
        )
    with col2:
        end_date = st.date_input(
            "Date de fin", value=date(2024, 1, 1), format="DD/MM/YYYY"
        )

    nb_days = (end_date - start_date).days + 1 if end_date >= start_date else 0

    if end_date < start_date:
        st.error("La date de fin doit être postérieure ou égale à la date de début.")
    elif nb_days > MAX_DAYS:
        st.warning(
            f"Plage trop large ({nb_days} jours). Pour rester raisonnable sur une "
            f"instance gratuite partagée, limite-toi à {MAX_DAYS} jours par lancement."
        )
    else:
        st.write(f"{nb_days} jour(s) seront interrogés.")

        launch = st.button("Lancer le scraping", type="primary", disabled=nb_days == 0)

        if launch:
            if not _SCRAPE_LOCK.acquire(blocking=False):
                st.error(
                    "Un autre scraping est déjà en cours sur cette instance. "
                    "Réessaie dans quelques instants."
                )
            else:
                try:
                    progress_bar = st.progress(0.0)
                    status_placeholder = st.empty()
                    log_placeholder = st.container(height=250)

                    total_days = nb_days
                    counters = {"ok": 0, "end_of_day": 0, "error": 0}

                    def on_progress(current_date, reunion_number, status, data):
                        counters[status] = counters.get(status, 0) + 1
                        days_done = (current_date - start_date).days
                        # petite estimation de progression : jours déjà terminés
                        # + fraction du jour courant (approximative, juste pour
                        # donner un retour visuel, pas une mesure exacte)
                        progress_bar.progress(min(1.0, days_done / max(total_days, 1)))
                        status_placeholder.write(
                            f"📅 {current_date.strftime('%d/%m/%Y')} — réunion R{reunion_number} : {status}"
                        )
                        if status == "ok":
                            libelle = data.get("hippodrome", {}).get("libelleCourt", "?")
                            log_placeholder.write(f"✅ {current_date.strftime('%d/%m/%Y')} R{reunion_number} — {libelle}")
                        elif status == "error":
                            log_placeholder.write(f"⚠️ Erreur sur {current_date.strftime('%d/%m/%Y')} R{reunion_number}")

                    with st.spinner("Scraping en cours..."):
                        call_api_between_dates(
                            datetime.combine(start_date, datetime.min.time()),
                            datetime.combine(end_date, datetime.min.time()),
                            progress_callback=on_progress,
                        )

                    progress_bar.progress(1.0)
                    st.success(
                        f"Terminé : {counters.get('ok', 0)} réunion(s) récupérée(s), "
                        f"{counters.get('error', 0)} erreur(s)."
                    )
                    st.cache_data.clear()  # les données affichées dans l'autre onglet ont changé
                except Exception as exc:  # noqa: BLE001 — on veut afficher l'erreur dans l'UI
                    st.error(f"Le scraping s'est arrêté suite à une erreur : {exc}")
                finally:
                    _SCRAPE_LOCK.release()

# ----------------------------------------------------------------------------
# Onglet exploration des données déjà collectées
# ----------------------------------------------------------------------------
with tab_donnees:
    st.subheader("Données en base")

    @st.cache_data(ttl=60)
    def load_table(table_name: str) -> pd.DataFrame:
        return pd.read_sql_table(table_name, engine)

    try:
        reunions = load_table("pmu_reunions")
        courses = load_table("pmu_courses")
        hippodromes = load_table("pmu_hippodromes")
        pays = load_table("pmu_pays")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Impossible de lire la base : {exc}")
        reunions = courses = hippodromes = pays = pd.DataFrame()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Réunions", len(reunions))
    m2.metric("Courses", len(courses))
    m3.metric("Hippodromes", len(hippodromes))
    m4.metric("Pays", len(pays))

    if not reunions.empty:
        st.markdown("**Réunions**")
        st.dataframe(reunions, width='stretch', hide_index=True)

    if not courses.empty:
        st.markdown("**Courses**")
        st.dataframe(courses, width='stretch', hide_index=True)

    with st.expander("Hippodromes et pays"):
        col_a, col_b = st.columns(2)
        col_a.dataframe(hippodromes, width='stretch', hide_index=True)
        col_b.dataframe(pays, width='stretch', hide_index=True)

    if reunions.empty and courses.empty:
        st.info("Aucune donnée pour l'instant — lance un scraping dans le premier onglet.")
