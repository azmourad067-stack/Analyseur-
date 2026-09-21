from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
HORSE_PRONO_ROOT = HERE if (HERE / "app_core").exists() else HERE.parent

for p in (HORSE_PRONO_ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app_core.db import get_history_as_of, get_supabase_client
from app_core.features import entity_snapshot_from_history
from app_core.pmu import get_programme, programme_choices

try:
    from scripts.m8_1_top3_place_r1 import (
        MODEL_VERSION_ID,
        R1_PLACED_ODDS_MEDIAN,
        _load_frozen_m8,
        predict_r1_course,
    )
except Exception:
    from m8_1_top3_place_r1 import (
        MODEL_VERSION_ID,
        R1_PLACED_ODDS_MEDIAN,
        _load_frozen_m8,
        predict_r1_course,
    )

PARIS_TZ = ZoneInfo("Europe/Paris")

st.set_page_config(
    page_title="HorseProno M8.1 Place",
    page_icon="🏇",
    layout="wide",
)

st.title("🏇 HorseProno M8.1 — Top 3 Placés R1")
st.caption(
    "Application séparée dédiée à la construction de 3 chevaux susceptibles "
    "de terminer dans les trois premiers. Modèle M8 #8 figé + logique M8.1 spécialisée R1."
)


def safe_int(value, default=None):
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def course_label(choice: dict) -> str:
    c = safe_int(choice.get("course"), 0)
    hippodrome = str(choice.get("hippodrome") or "Hippodrome")
    discipline = str(choice.get("discipline") or "Discipline inconnue")
    distance = safe_float(choice.get("distance"))
    field_size = safe_int(choice.get("field_size"))
    bits = [f"R1C{c}", hippodrome, discipline]
    if distance is not None:
        bits.append(f"{int(distance)} m")
    if field_size is not None:
        bits.append(f"{field_size} partants")
    return " — ".join(bits)


@st.cache_resource(show_spinner=False)
def load_model():
    client = get_supabase_client()
    if client is None:
        raise RuntimeError(
            "Supabase non configuré. Ajoute SUPABASE_URL et SUPABASE_KEY dans les secrets Streamlit."
        )
    return client, _load_frozen_m8(client)


@st.cache_data(ttl=300, show_spinner=False)
def load_r1_program(date_iso: str) -> list[dict]:
    target_date = datetime.strptime(date_iso, "%Y-%m-%d").date()
    programme = get_programme(target_date)
    choices = [
        dict(choice)
        for choice in programme_choices(programme)
        if safe_int(choice.get("reunion"), 0) == 1
    ]
    choices.sort(key=lambda x: safe_int(x.get("course"), 999))
    return choices


@st.cache_data(ttl=900, show_spinner=False)
def load_snapshots(date_iso: str):
    target_date = datetime.strptime(date_iso, "%Y-%m-%d").date()
    history = get_history_as_of(target_date)
    return entity_snapshot_from_history(history, target_date)


def top3_table(top3: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Rang"] = top3["m81_rank"].astype(int)
    out["N°"] = top3["horse_number"].astype(int)
    out["Cheval"] = top3["horse_name"].astype(str)
    out["Cote"] = pd.to_numeric(top3["odds_num"], errors="coerce").round(2)
    out["P(placé) M8"] = (
        pd.to_numeric(top3["place_probability_num"], errors="coerce") * 100
    ).round(1)
    out["Rang M8 Place"] = top3["m8_place_rank"].astype(int)
    out["Rang marché"] = top3["market_rank"].astype(int)
    out["Consensus"] = top3["consensus_top3"].map({True: "✅ Oui", False: "—"})
    out["Confiance"] = top3["m81_confidence"].astype(str)
    out["Profil cote"] = top3["odds_vs_r1_placed_median"].replace(
        {
            "SOUS_MEDIANE_7_45": "≤ médiane 7,45",
            "AU_DESSUS_MEDIANE_7_45": "> médiane 7,45",
            "COTE_INCONNUE": "Cote inconnue",
        }
    )
    return out


with st.sidebar:
    st.header("⚙️ Sélection")
    selected_date = st.date_input(
        "Date des courses",
        value=datetime.now(PARIS_TZ).date(),
        format="DD/MM/YYYY",
    )
    st.info("Cette application analyse uniquement la réunion R1.")
    st.divider()
    st.markdown(f"**Modèle :** M8 #{MODEL_VERSION_ID}")
    st.markdown(f"**Médiane cote des placés R1 :** {R1_PLACED_ODDS_MEDIAN:.2f}")
    st.caption("La médiane est un repère descriptif, pas un filtre obligatoire.")


date_iso = selected_date.isoformat()

try:
    r1_choices = load_r1_program(date_iso)
except Exception as exc:
    st.error(f"Impossible de charger le programme PMU : {exc}")
    st.stop()

if not r1_choices:
    st.warning("Aucune course R1 n'a été trouvée pour cette date.")
    st.stop()

course_map = {course_label(choice): choice for choice in r1_choices}
selected_label = st.selectbox(
    "🏁 Choisis la course R1 à analyser",
    options=list(course_map.keys()),
)
selected_choice = course_map[selected_label]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Réunion", "R1")
with c2:
    st.metric("Course", f"C{safe_int(selected_choice.get('course'), '?')}")
with c3:
    st.metric("Discipline", str(selected_choice.get("discipline") or "—"))
with c4:
    fs = safe_int(selected_choice.get("field_size"))
    st.metric("Partants", str(fs) if fs is not None else "—")

if st.button(
    "🔎 Construire les 3 chevaux placés",
    type="primary",
    use_container_width=True,
):
    try:
        client, model = load_model()
        snapshots = load_snapshots(date_iso)
        result = predict_r1_course(
            client=client,
            model=model,
            snapshots=snapshots,
            target_date=selected_date,
            choice=selected_choice,
        )
        st.session_state["m81_place_result"] = result
    except Exception as exc:
        st.error(f"Analyse impossible : {exc}")

result = st.session_state.get("m81_place_result")

if result:
    current_course = safe_int(selected_choice.get("course"))
    if result.get("date") == date_iso and safe_int(result.get("course")) == current_course:
        top3 = result["top3"].copy()
        full = result["full"].copy()

        st.divider()
        st.subheader("🏆 Top 3 Placés M8.1")

        numbers = "  —  ".join(str(int(x)) for x in top3["horse_number"].tolist())
        st.success(f"**Sélection : {numbers}**")

        cols = st.columns(3)
        for idx, (_, row) in enumerate(top3.iterrows()):
            with cols[idx]:
                number = safe_int(row.get("horse_number"), "?")
                horse = str(row.get("horse_name") or "Cheval")
                odds = safe_float(row.get("odds_num"))
                p_place = safe_float(row.get("place_probability_num"))
                st.markdown(f"### #{number} · {horse}")
                st.metric("Cote", f"{odds:.2f}" if odds is not None else "—")
                st.metric(
                    "P(placé) M8",
                    f"{p_place*100:.1f} %" if p_place is not None else "—",
                )
                st.caption(
                    f"Consensus : {'✅' if bool(row.get('consensus_top3')) else '—'} · "
                    f"Confiance : {row.get('m81_confidence', '—')}"
                )

        st.markdown(f"**Consensus M8/marché dans le trio : {result['consensus_count']}/3**")
        display = top3_table(top3)
        st.dataframe(display, use_container_width=True, hide_index=True)

        csv_bytes = display.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Télécharger le Top 3 en CSV",
            data=csv_bytes,
            file_name=f"m8_1_place_{date_iso}_R1C{current_course}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("📊 Voir le classement complet"):
            cols_full = [
                "m81_rank", "horse_number", "horse_name", "odds_num",
                "place_probability_num", "m8_place_rank", "market_rank",
                "m81_place_score", "consensus_top3", "m81_confidence",
            ]
            full_table = full[[c for c in cols_full if c in full.columns]].copy()
            full_table = full_table.rename(columns={
                "m81_rank": "Rang M8.1",
                "horse_number": "N°",
                "horse_name": "Cheval",
                "odds_num": "Cote",
                "place_probability_num": "P(placé) M8",
                "m8_place_rank": "Rang M8 Place",
                "market_rank": "Rang marché",
                "m81_place_score": "Score M8.1",
                "consensus_top3": "Consensus",
                "m81_confidence": "Confiance",
            })
            if "P(placé) M8" in full_table.columns:
                full_table["P(placé) M8"] = (
                    pd.to_numeric(full_table["P(placé) M8"], errors="coerce") * 100
                ).round(1)
            if "Cote" in full_table.columns:
                full_table["Cote"] = pd.to_numeric(full_table["Cote"], errors="coerce").round(2)
            if "Score M8.1" in full_table.columns:
                full_table["Score M8.1"] = pd.to_numeric(full_table["Score M8.1"], errors="coerce").round(3)
            st.dataframe(full_table, use_container_width=True, hide_index=True)

        with st.expander("🧠 Comment M8.1 construit le trio ?"):
            discipline = str(result.get("discipline") or "").upper()
            if discipline == "PLAT":
                st.markdown(
                    "**R1 Plat :** 70 % rang M8 Place + 30 % rang marché."
                )
            else:
                st.markdown(
                    "**R1 autres disciplines :** 10 % rang M8 Place + 90 % rang marché."
                )
            st.markdown(
                f"La médiane de cote observée chez les chevaux réellement placés en R1 était "
                f"**{R1_PLACED_ODDS_MEDIAN:.2f}**. Elle sert seulement de repère visuel."
            )

        st.caption(
            "Les performances historiques ne garantissent pas les résultats futurs. "
            "Cette application sert à tester et comparer une méthode de classement."
        )
