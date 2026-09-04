import streamlit as st
import pandas as pd
from datetime import date, timedelta

from src.database.supabase import is_configured, get_courses, get_course_participants, get_performance_stats
from src.api.pmu_api import fetch_day
from src.core.processor import process_day_races
from src.core.bankroll import calculer_mise_optimale
from src.database.supabase import insert_courses

st.set_page_config(page_title="PMU Elite Punter", page_icon="🏇", layout="wide")

st.title("🏇 PMU Elite Punter — Python / Streamlit")
st.caption("Migration complète Node.js → Python • Streamlit + Supabase")

if not is_configured():
    st.warning("Supabase n'est pas encore configuré. Ajoute SUPABASE_URL et SUPABASE_KEY dans les secrets Streamlit.")
    st.code('SUPABASE_URL = "https://xxxxx.supabase.co"\nSUPABASE_KEY = "..."', language="toml")
    st.stop()

with st.sidebar:
    st.header("Navigation")
    page = st.radio("", ["🏠 Tableau de bord", "📅 Courses", "🔄 Synchronisation", "💰 Bankroll", "🤖 Performance IA"])

if page == "🏠 Tableau de bord":
    st.subheader("Vue d'ensemble")
    today = date.today().isoformat()
    courses = get_courses(target_date=today, limit=100)
    c1, c2, c3 = st.columns(3)
    c1.metric("Courses aujourd'hui", len(courses))
    c2.metric("Participants stockés", sum(int(c.get("nb_participants_stockes") or 0) for c in courses))
    c3.metric("Statut", "Supabase connecté")
    if courses:
        df = pd.DataFrame(courses)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Aucune course pour aujourd'hui. Utilise Synchronisation pour importer les données PMU.")

elif page == "📅 Courses":
    st.subheader("Recherche de courses")
    col1, col2 = st.columns(2)
    selected_date = col1.date_input("Date", value=date.today())
    discipline = col2.selectbox("Discipline", ["Toutes", "TROT", "PLAT", "OBSTACLE", "MONTE", "ATTELE"])
    courses = get_courses(target_date=selected_date.isoformat(), limit=100)
    if discipline != "Toutes":
        courses = [c for c in courses if str(c.get("discipline", "")).upper() == discipline]
    if not courses:
        st.info("Aucune course trouvée.")
    else:
        labels = [f"R{c.get('reunion_num')}C{c.get('course_num')} — {c.get('hippodrome')} — {c.get('heure')}" for c in courses]
        idx = st.selectbox("Course", range(len(labels)), format_func=lambda i: labels[i])
        course = courses[idx]
        st.json({k: v for k, v in course.items() if k not in {"id"}})
        participants = get_course_participants(course["id"])
        if participants:
            df = pd.DataFrame(participants)
            df = df.sort_values("prediction_score", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)

elif page == "🔄 Synchronisation":
    st.subheader("Importer l'historique PMU")
    days = st.number_input("Nombre de jours", min_value=1, max_value=365, value=1)
    if st.button("🚀 Lancer la synchronisation", type="primary"):
        progress = st.progress(0)
        status = st.empty()
        total_inserted = 0
        end = date.today()
        for i in range(int(days)):
            current = end - timedelta(days=i)
            status.info(f"Récupération {current.isoformat()} ({i+1}/{days})…")
            try:
                raw = fetch_day(current)
                processed = process_day_races(raw, current.isoformat())
                inserted = insert_courses(processed)
                total_inserted += inserted
                status.success(f"{current.isoformat()} : {inserted} course(s) insérée(s).")
            except Exception as exc:
                st.error(f"{current.isoformat()} : {exc}")
            progress.progress((i + 1) / int(days))
        st.success(f"Synchronisation terminée — {total_inserted} course(s) insérée(s).")

elif page == "💰 Bankroll":
    st.subheader("Kelly Criterion")
    col1, col2, col3 = st.columns(3)
    cote = col1.number_input("Cote", min_value=1.01, value=4.5, step=0.1)
    score = col2.number_input("Score IA", min_value=0.0, max_value=100.0, value=85.0)
    bankroll = col3.number_input("Bankroll (€)", min_value=1.0, value=1000.0, step=10.0)
    result = calculer_mise_optimale(cote, score, bankroll)
    a, b, c, d = st.columns(4)
    a.metric("Recommandation", result["recommandation"])
    b.metric("Mise", f'{result["mise"]:.2f} €')
    c.metric("Edge", f'{result["edge"]*100:.2f} %')
    d.metric("ROI attendu", f'{result["roi_attendu"]:.2f} %')
    st.json(result)

else:
    st.subheader("Performance IA")
    stats = get_performance_stats()
    g = stats.get("global", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Courses", g.get("total_courses", 0))
    c2.metric("Victoires", g.get("wins", 0))
    c3.metric("Win rate", f'{g.get("win_rate", 0)} %')
    c4.metric("ROI", f'{g.get("roi", 0)} %')
    history = stats.get("history", [])
    if history:
        df = pd.DataFrame(history)
        df["date"] = pd.to_datetime(df["date"])
        st.line_chart(df.set_index("date")["cumulative"])
        st.dataframe(df, use_container_width=True, hide_index=True)
