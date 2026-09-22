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
    from scripts.m8_2_top3_place import (
        ALL_PLACED_ODDS_MEDIAN,
        MODEL_VERSION_ID,
        OUTSIDER_MAX_ODDS_EXCLUSIVE,
        OUTSIDER_MIN_ODDS,
        R1_PLACED_ODDS_MEDIAN,
        _load_frozen_m8,
        predict_course,
    )
except Exception:
    from m8_2_top3_place import (
        ALL_PLACED_ODDS_MEDIAN,
        MODEL_VERSION_ID,
        OUTSIDER_MAX_ODDS_EXCLUSIVE,
        OUTSIDER_MIN_ODDS,
        R1_PLACED_ODDS_MEDIAN,
        _load_frozen_m8,
        predict_course,
    )

PARIS_TZ = ZoneInfo("Europe/Paris")

st.set_page_config(
    page_title="HorseProno M8.2 Place",
    page_icon="🏇",
    layout="wide",
)

st.title("🏇 HorseProno M8.2 — Top 3 Placés")
st.caption(
    "M8 #8 figé + classement Place adaptatif par réunion/discipline + "
    "signal outsider placé. Toutes les réunions du programme sont sélectionnables."
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


def meeting_label(meeting: int, choices: list[dict]) -> str:
    hippos = sorted(
        {
            str(x.get("hippodrome") or "").strip()
            for x in choices
            if safe_int(x.get("reunion")) == meeting and str(x.get("hippodrome") or "").strip()
        }
    )
    suffix = f" — {', '.join(hippos[:2])}" if hippos else ""
    return f"R{meeting}{suffix}"


def course_label(choice: dict) -> str:
    r = safe_int(choice.get("reunion"), 0)
    c = safe_int(choice.get("course"), 0)
    discipline = str(choice.get("discipline") or "Discipline inconnue")
    distance = safe_float(choice.get("distance"))
    field_size = safe_int(choice.get("field_size"))

    bits = [f"R{r}C{c}", discipline]
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
            "Supabase non configuré. Ajoute SUPABASE_URL et SUPABASE_KEY "
            "dans les secrets Streamlit."
        )
    return _load_frozen_m8(client)


@st.cache_data(ttl=300, show_spinner=False)
def load_program(date_iso: str) -> list[dict]:
    target_date = datetime.strptime(date_iso, "%Y-%m-%d").date()
    programme = get_programme(target_date)
    choices = [dict(choice) for choice in programme_choices(programme)]
    choices.sort(
        key=lambda x: (
            safe_int(x.get("reunion"), 999),
            safe_int(x.get("course"), 999),
        )
    )
    return choices


@st.cache_data(ttl=900, show_spinner=False)
def load_snapshots(date_iso: str):
    target_date = datetime.strptime(date_iso, "%Y-%m-%d").date()
    history = get_history_as_of(target_date)
    return entity_snapshot_from_history(history, target_date)


def top3_table(top3: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Rang"] = top3["m82_rank"].astype(int)
    out["N°"] = top3["horse_number"].astype(int)
    out["Cheval"] = top3["horse_name"].astype(str)
    out["Cote"] = pd.to_numeric(top3["odds_num"], errors="coerce").round(2)
    out["P(placé) M8"] = (
        pd.to_numeric(top3["place_probability_num"], errors="coerce") * 100
    ).round(1)
    out["Rang M8 Place"] = top3["m8_place_rank"].astype(int)
    out["Rang marché"] = top3["market_rank"].astype(int)
    out["Consensus"] = top3["consensus_top3"].map({True: "✅ Oui", False: "—"})
    out["Confiance"] = top3["m82_confidence"].astype(str)
    return out


def outsider_table(outsider: pd.DataFrame) -> pd.DataFrame:
    if outsider.empty:
        return pd.DataFrame()
    row = outsider.iloc[0]
    return pd.DataFrame(
        [
            {
                "N°": safe_int(row.get("horse_number")),
                "Cheval": str(row.get("horse_name") or ""),
                "Cote": round(safe_float(row.get("odds_num"), 0.0), 2),
                "P(placé) M8": round(safe_float(row.get("place_probability_num"), 0.0) * 100, 1),
                "Rang M8 Place": safe_int(row.get("m8_place_rank")),
                "Rang marché": safe_int(row.get("market_rank")),
                "Signal": str(row.get("outsider_signal") or "MODÉRÉ"),
            }
        ]
    )


with st.sidebar:
    st.header("⚙️ Sélection")
    selected_date = st.date_input(
        "Date des courses",
        value=datetime.now(PARIS_TZ).date(),
        format="DD/MM/YYYY",
    )
    st.divider()
    st.markdown(f"**Modèle ML :** M8 #{MODEL_VERSION_ID}")
    st.markdown("**Version de sélection :** M8.2 Place")
    st.caption(
        f"Repère cote placés : R1 {R1_PLACED_ODDS_MEDIAN:.2f} · "
        f"toutes réunions {ALL_PLACED_ODDS_MEDIAN:.2f}."
    )


date_iso = selected_date.isoformat()

try:
    choices = load_program(date_iso)
except Exception as exc:
    st.error(f"Impossible de charger le programme PMU : {exc}")
    st.stop()

if not choices:
    st.warning("Aucune course trouvée pour cette date.")
    st.stop()

meetings = sorted(
    {
        safe_int(choice.get("reunion"))
        for choice in choices
        if safe_int(choice.get("reunion")) is not None
    }
)

meeting_map = {meeting_label(m, choices): m for m in meetings}
selected_meeting_label = st.selectbox(
    "📍 Choisis la réunion",
    options=list(meeting_map.keys()),
)
selected_meeting = meeting_map[selected_meeting_label]

meeting_choices = [
    choice for choice in choices if safe_int(choice.get("reunion")) == selected_meeting
]

course_map = {course_label(choice): choice for choice in meeting_choices}
selected_course_label = st.selectbox(
    "🏁 Choisis la course",
    options=list(course_map.keys()),
)
selected_choice = course_map[selected_course_label]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Réunion", f"R{selected_meeting}")
with c2:
    st.metric("Course", f"C{safe_int(selected_choice.get('course'), '?')}")
with c3:
    st.metric("Discipline", str(selected_choice.get("discipline") or "—"))
with c4:
    fs = safe_int(selected_choice.get("field_size"))
    st.metric("Partants annoncés", str(fs) if fs is not None else "—")

if st.button(
    "🔎 Construire le Top 3 M8.2",
    type="primary",
    use_container_width=True,
):
    try:
        model = load_model()
        snapshots = load_snapshots(date_iso)
        result = predict_course(
            model=model,
            snapshots=snapshots,
            target_date=selected_date,
            choice=selected_choice,
        )
        st.session_state["m82_place_result"] = result
    except Exception as exc:
        st.error(f"Analyse impossible : {exc}")

result = st.session_state.get("m82_place_result")

if result:
    current_course = safe_int(selected_choice.get("course"))
    if (
        result.get("date") == date_iso
        and safe_int(result.get("reunion")) == selected_meeting
        and safe_int(result.get("course")) == current_course
    ):
        top3 = result["top3"].copy()
        full = result["full"].copy()
        outsider = result["outsider"].copy()

        st.divider()
        st.subheader("🏆 Top 3 Placés M8.2")

        numbers = "  —  ".join(str(int(x)) for x in top3["horse_number"].tolist())
        st.success(f"**Sélection : {numbers}**")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Confiance course", result.get("race_confidence", "—"))
        with m2:
            st.metric("Consensus M8 / marché", f"{result.get('consensus_count', 0)}/3")
        with m3:
            st.metric(
                "Pondération",
                f"M8 {result.get('place_weight', 0)*100:.0f}% / marché {result.get('market_weight', 0)*100:.0f}%",
            )

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
                    f"Confiance : {row.get('m82_confidence', '—')}"
                )

        display = top3_table(top3)
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.subheader("🎯 Outsider placé à surveiller")
        if outsider.empty:
            st.info(
                f"Aucun outsider répondant au filtre cote {OUTSIDER_MIN_ODDS:.0f}–"
                f"{OUTSIDER_MAX_ODDS_EXCLUSIVE:.0f} et Top 8 M8/marché sur cette course."
            )
        else:
            row = outsider.iloc[0]
            st.warning(
                f"**N°{safe_int(row.get('horse_number'))} — {row.get('horse_name', '')}** · "
                f"cote {safe_float(row.get('odds_num'), 0):.2f} · "
                f"signal {row.get('outsider_signal', 'MODÉRÉ')}"
            )
            st.dataframe(outsider_table(outsider), use_container_width=True, hide_index=True)
            st.caption(
                "Ce cheval n'est pas forcé dans le trio principal. Historiquement, le signal outsider "
                "8–25 / Top 8 M8 & marché a terminé placé dans ~32 % des cas sur l'échantillon étudié."
            )

        csv_bytes = display.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Télécharger le Top 3 en CSV",
            data=csv_bytes,
            file_name=f"m8_2_place_{date_iso}_R{selected_meeting}C{current_course}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("📊 Voir le classement complet"):
            cols_full = [
                "m82_rank",
                "horse_number",
                "horse_name",
                "odds_num",
                "place_probability_num",
                "m8_place_rank",
                "market_rank",
                "m82_place_score",
                "consensus_top3",
                "m82_confidence",
            ]
            table = full[[c for c in cols_full if c in full.columns]].copy()
            table = table.rename(
                columns={
                    "m82_rank": "Rang M8.2",
                    "horse_number": "N°",
                    "horse_name": "Cheval",
                    "odds_num": "Cote",
                    "place_probability_num": "P(placé) M8",
                    "m8_place_rank": "Rang M8 Place",
                    "market_rank": "Rang marché",
                    "m82_place_score": "Score M8.2",
                    "consensus_top3": "Consensus",
                    "m82_confidence": "Confiance",
                }
            )
            if "P(placé) M8" in table.columns:
                table["P(placé) M8"] = (
                    pd.to_numeric(table["P(placé) M8"], errors="coerce") * 100
                ).round(1)
            if "Cote" in table.columns:
                table["Cote"] = pd.to_numeric(table["Cote"], errors="coerce").round(2)
            if "Score M8.2" in table.columns:
                table["Score M8.2"] = pd.to_numeric(
                    table["Score M8.2"], errors="coerce"
                ).round(3)
            st.dataframe(table, use_container_width=True, hide_index=True)

        with st.expander("🧠 Comment M8.2 choisit ?"):
            st.markdown(
                f"""
**Course analysée :** R{result['reunion']}C{result['course']} · {result.get('discipline') or '—'}

- Poids M8 Place : **{result.get('place_weight', 0)*100:.0f} %**
- Poids marché : **{result.get('market_weight', 0)*100:.0f} %**
- Taille réelle du lot après non-partants : **{result.get('field_size', '—')}**
- Consensus dans le trio : **{result.get('consensus_count', 0)}/3**
- Confiance course : **{result.get('race_confidence', '—')}**

M8.2 utilise des **rangs** plutôt que d'additionner directement des probabilités sur des échelles différentes.
R1 conserve son profil spécialisé. Les autres réunions utilisent une pondération adaptée à la discipline.
                """
            )

        with st.expander("🧪 Repères du rétro-test M8.2"):
            st.markdown(
                """
Sur **340 courses** avec prédictions M8 stockées (12–20 septembre 2026) :

- **M8.2 : 16,18 % de 3/3 exacts**
- **M8.2 : 62,65 % avec au moins 2/3**
- Marché : 13,82 % de 3/3
- M8 Place pur : 14,41 % de 3/3

Sur R1, le réglage spécifique reste celui validé précédemment : 15,0 % de 3/3 sur 60 courses.
Ces chiffres décrivent un échantillon historique et ne garantissent pas les résultats futurs.
                """
            )
