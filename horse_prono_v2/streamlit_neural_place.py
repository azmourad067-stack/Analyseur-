from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent

# Ce fichier est prévu dans : repo/horse_prono_v2/streamlit_neural_place.py
HORSE_PRONO_ROOT = HERE if (HERE / "app_core").exists() else HERE.parent
REPO_ROOT = HORSE_PRONO_ROOT.parent

for p in (REPO_ROOT, HORSE_PRONO_ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# =============================================================================
# IMPORTS PROJET
# =============================================================================

from app_core.db import get_supabase_client
from app_core.forward_utils import non_runner_numbers
from app_core.pmu import (
    get_participants,
    get_programme,
    participants_to_df,
    programme_choices,
)

from neural.forward_capture import (
    NEURAL_ARTIFACT_HASH,
    NEURAL_MODEL_NAME,
    NEURAL_MODEL_VERSION_ID,
    load_forward_neural,
)
from neural.predict import predict_dataframe

PARIS_TZ = ZoneInfo("Europe/Paris")

# =============================================================================
# STREAMLIT
# =============================================================================

st.set_page_config(
    page_title="HorseProno Neural #9 — Simple Placé",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 HorseProno Neural #9 — Simple Placé")
st.caption(
    "Neural V1 figé · classement P(Top 3) · toutes réunions / toutes courses · "
    "mise en avant expérimentale du rang Neural #5."
)

# =============================================================================
# UTILS
# =============================================================================


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
            if safe_int(x.get("reunion")) == meeting
            and str(x.get("hippodrome") or "").strip()
        }
    )
    suffix = f" — {', '.join(hippos[:2])}" if hippos else ""
    return f"R{meeting}{suffix}"


def course_label(choice: dict) -> str:
    r = safe_int(choice.get("reunion"), 0)
    c = safe_int(choice.get("course"), 0)
    label = str(choice.get("label") or "").strip()
    discipline = str(choice.get("discipline") or "INCONNU")
    distance = safe_float(choice.get("distance"))
    field_size = safe_int(choice.get("field_size"))

    bits = [f"R{r}C{c}"]
    if label:
        bits.append(label)
    bits.append(discipline)

    if distance is not None:
        bits.append(f"{int(distance)} m")
    if field_size is not None:
        bits.append(f"{field_size} partants")

    return " — ".join(bits)


def enrich_race_with_choice(race: pd.DataFrame, choice: dict) -> pd.DataFrame:
    """Complète les métadonnées PMU nécessaires au Neural."""
    race = race.copy()

    if race.empty:
        return race

    metadata = {
        "discipline": choice.get("discipline") or "INCONNU",
        "hippodrome": choice.get("hippodrome") or "INCONNU",
        "distance": choice.get("distance"),
        "terrain": choice.get("terrain") or "INCONNU",
        "field_size": len(race),
    }

    for column, value in metadata.items():
        if column not in race.columns:
            race[column] = value
            continue

        # Les données du programme sont plus fiables que les placeholders
        # "INCONNU" / None éventuellement présents dans le payload participants.
        current = race[column]
        missing = current.isna()

        if current.dtype == object:
            text = current.fillna("").astype(str).str.strip().str.upper()
            missing = missing | text.isin({"", "INCONNU", "UNKNOWN", "NONE", "NAN"})

        race.loc[missing, column] = value

    race["field_size"] = len(race)
    return race


def clean_active_runners(race: pd.DataFrame, non_runners: set[int]) -> pd.DataFrame:
    if race.empty:
        return race

    out = race.copy()
    out["horse_number_num"] = pd.to_numeric(
        out.get("horse_number"),
        errors="coerce",
    )

    if non_runners:
        out = out[
            ~out["horse_number_num"].map(
                lambda x: int(x) in non_runners if pd.notna(x) else False
            )
        ].copy()

    out = out[out["horse_number_num"].notna()].copy()
    out["horse_number"] = out["horse_number_num"].astype(int)
    out = out.drop(columns=["horse_number_num"])

    out["field_size"] = len(out)
    return out.reset_index(drop=True)


def display_table(ranked: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    out["Rang Neural"] = pd.to_numeric(
        ranked["neural_rank"], errors="coerce"
    ).astype("Int64")
    out["N°"] = pd.to_numeric(
        ranked["horse_number"], errors="coerce"
    ).astype("Int64")
    out["Cheval"] = ranked["horse_name"].fillna("").astype(str)

    out["P(Top 3) Neural"] = (
        pd.to_numeric(
            ranked["neural_top3_probability"],
            errors="coerce",
        )
        * 100
    ).round(1)

    if "odds" in ranked.columns:
        out["Cote"] = pd.to_numeric(
            ranked["odds"],
            errors="coerce",
        ).round(2)

    if "jockey" in ranked.columns:
        out["Jockey / Driver"] = ranked["jockey"].fillna("").astype(str)

    if "trainer" in ranked.columns:
        out["Entraîneur"] = ranked["trainer"].fillna("").astype(str)

    if "recent_form" in ranked.columns:
        out["Musique"] = ranked["recent_form"].fillna("").astype(str)

    return out


# =============================================================================
# CACHE
# =============================================================================


@st.cache_resource(show_spinner=False)
def load_neural_model():
    client = get_supabase_client()

    if client is None:
        raise RuntimeError(
            "Supabase non configuré. Ajoute SUPABASE_URL et SUPABASE_KEY "
            "(ou SUPABASE_SERVICE_KEY selon ta configuration) dans les Secrets Streamlit."
        )

    # Cette fonction contrôle :
    # - l'ID Supabase #9
    # - le nom du modèle
    # - le hash stocké en base
    # - le hash du .pt présent dans le dépôt
    return load_forward_neural(client)


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


@st.cache_data(ttl=120, show_spinner=False)
def load_race(date_iso: str, reunion: int, course: int, choice: dict):
    target_date = datetime.strptime(date_iso, "%Y-%m-%d").date()

    payload = get_participants(
        target_date,
        int(reunion),
        int(course),
    )

    race = participants_to_df(
        payload,
        target_date,
        int(reunion),
        int(course),
    )

    race = enrich_race_with_choice(race, choice)

    non_runners = non_runner_numbers(payload)
    race = clean_active_runners(race, non_runners)

    return race, sorted(non_runners)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("⚙️ Neural #9")

    selected_date = st.date_input(
        "Date des courses",
        value=datetime.now(PARIS_TZ).date(),
        format="DD/MM/YYYY",
    )

    st.divider()

    st.markdown(f"**Modèle :** #{NEURAL_MODEL_VERSION_ID} `{NEURAL_MODEL_NAME}`")
    st.markdown("**Cible ML :** probabilité de finir dans le Top 3")
    st.markdown("**Signal suivi :** rang Neural #5")
    st.caption(f"Hash attendu : `{NEURAL_ARTIFACT_HASH[:14]}…`")

    st.info(
        "Le rang #5 est un signal expérimental issu de notre validation historique. "
        "Il ne garantit pas qu'un cheval sera placé."
    )

# =============================================================================
# PROGRAMME
# =============================================================================

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

meeting_map = {
    meeting_label(meeting, choices): meeting
    for meeting in meetings
}

selected_meeting_label = st.selectbox(
    "📍 Choisis la réunion",
    options=list(meeting_map.keys()),
)
selected_meeting = meeting_map[selected_meeting_label]

meeting_choices = [
    choice
    for choice in choices
    if safe_int(choice.get("reunion")) == selected_meeting
]

course_map = {
    course_label(choice): choice
    for choice in meeting_choices
}

selected_course_label = st.selectbox(
    "🏁 Choisis la course",
    options=list(course_map.keys()),
)
selected_choice = course_map[selected_course_label]

selected_course = safe_int(selected_choice.get("course"))

# =============================================================================
# RESUME COURSE
# =============================================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Réunion", f"R{selected_meeting}")

with c2:
    st.metric("Course", f"C{selected_course}")

with c3:
    st.metric(
        "Discipline",
        str(selected_choice.get("discipline") or "—"),
    )

with c4:
    announced = safe_int(selected_choice.get("field_size"))
    st.metric(
        "Partants annoncés",
        str(announced) if announced is not None else "—",
    )

st.caption(
    f"{selected_choice.get('hippodrome') or 'Hippodrome inconnu'} · "
    f"{selected_choice.get('distance') or '—'} m · "
    f"terrain : {selected_choice.get('terrain') or 'INCONNU'}"
)

# =============================================================================
# ANALYSE
# =============================================================================

if st.button(
    "🧠 Lancer Neural #9",
    type="primary",
    use_container_width=True,
):
    try:
        with st.spinner("Chargement du Neural #9 et calcul du classement…"):
            bundle = load_neural_model()

            race, non_runners = load_race(
                date_iso,
                selected_meeting,
                selected_course,
                selected_choice,
            )

            if race.empty:
                raise RuntimeError(
                    "Aucun partant actif exploitable sur cette course."
                )

            ranked = predict_dataframe(
                race,
                bundle,
            ).copy()

            ranked["neural_top3_probability"] = pd.to_numeric(
                ranked["neural_top3_probability"],
                errors="coerce",
            )

            ranked = (
                ranked
                .sort_values(
                    ["neural_top3_probability", "horse_number"],
                    ascending=[False, True],
                    na_position="last",
                )
                .reset_index(drop=True)
            )

            # Recalcul explicite du rang sur les seuls partants actifs.
            ranked["neural_rank"] = range(1, len(ranked) + 1)

            st.session_state["neural_place_result"] = {
                "date": date_iso,
                "reunion": selected_meeting,
                "course": selected_course,
                "choice": dict(selected_choice),
                "non_runners": non_runners,
                "ranked": ranked,
            }

    except Exception as exc:
        st.error(f"Analyse Neural impossible : {exc}")

# =============================================================================
# RESULTATS
# =============================================================================

result = st.session_state.get("neural_place_result")

if result:
    same_race = (
        result.get("date") == date_iso
        and safe_int(result.get("reunion")) == selected_meeting
        and safe_int(result.get("course")) == selected_course
    )

    if same_race:
        ranked = result["ranked"].copy()

        st.divider()
        st.subheader("🏆 Classement Neural #9")

        top_count = min(5, len(ranked))
        top5 = ranked.head(top_count)

        top_numbers = " — ".join(
            str(int(x))
            for x in top5["horse_number"].tolist()
        )
        st.success(f"**Top {top_count} Neural : {top_numbers}**")

        if result.get("non_runners"):
            st.caption(
                "Non-partants retirés avant calcul : "
                + ", ".join(
                    f"N°{x}" for x in result["non_runners"]
                )
            )

        # ---------------------------------------------------------------------
        # TOP 3
        # ---------------------------------------------------------------------

        st.markdown("### 🎯 Top 3 Neural")

        top3 = ranked.head(min(3, len(ranked)))
        cols = st.columns(max(1, len(top3)))

        for idx, (_, row) in enumerate(top3.iterrows()):
            with cols[idx]:
                number = safe_int(row.get("horse_number"), "?")
                horse = str(row.get("horse_name") or "Cheval")
                probability = safe_float(
                    row.get("neural_top3_probability")
                )
                odds = safe_float(row.get("odds"))

                st.markdown(f"#### #{number} · {horse}")
                st.metric(
                    "P(Top 3) Neural",
                    f"{probability * 100:.1f} %"
                    if probability is not None
                    else "—",
                )
                st.caption(
                    "Cote : "
                    + (
                        f"{odds:.2f}"
                        if odds is not None
                        else "indisponible"
                    )
                )

        # ---------------------------------------------------------------------
        # RANG 5 — SIGNAL SIMPLE PLACE
        # ---------------------------------------------------------------------

        st.markdown("### 💰 Signal expérimental Simple Placé — rang Neural #5")

        if len(ranked) >= 5:
            candidate = ranked.iloc[4]

            number = safe_int(candidate.get("horse_number"), "?")
            horse = str(candidate.get("horse_name") or "Cheval")
            probability = safe_float(
                candidate.get("neural_top3_probability")
            )
            odds = safe_float(candidate.get("odds"))
            form = str(candidate.get("recent_form") or "—")
            jockey = str(candidate.get("jockey") or "—")

            st.warning(
                f"**N°{number} — {horse}**"
            )

            a, b, c, d = st.columns(4)

            with a:
                st.metric("Rang Neural", "#5")

            with b:
                st.metric(
                    "P(Top 3)",
                    f"{probability * 100:.1f} %"
                    if probability is not None
                    else "—",
                )

            with c:
                st.metric(
                    "Cote actuelle",
                    f"{odds:.2f}"
                    if odds is not None
                    else "—",
                )

            with d:
                st.metric(
                    "Partants actifs",
                    len(ranked),
                )

            st.caption(
                f"Jockey/driver : {jockey} · Musique : {form}"
            )

            st.info(
                "Repère historique : le rang Neural #5 avait terminé placé "
                "20 fois sur 68 paris dans notre validation forward, avec "
                "+18,40 unités sur cet échantillon. Ce résultat historique "
                "n'est pas une garantie de rendement futur."
            )

        else:
            st.info(
                "Moins de 5 partants actifs : aucun signal rang Neural #5."
            )

        # ---------------------------------------------------------------------
        # TABLEAU COMPLET
        # ---------------------------------------------------------------------

        st.markdown("### 📊 Classement complet")

        display = display_table(ranked)

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = display.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "⬇️ Télécharger le classement Neural en CSV",
            data=csv_bytes,
            file_name=(
                f"neural_place_{date_iso}_"
                f"R{selected_meeting}C{selected_course}.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("🧠 Comment lire Neural #9 ?"):
            st.markdown(
                """
Le modèle Neural V1 prédit pour **chaque cheval** une probabilité de faire partie
des **trois premiers**. Il ne prédit pas directement le rapport Simple Placé PMU.

Le classement affiché trie les chevaux par `neural_top3_probability`,
puis recalcule les rangs après retrait des non-partants.

Le **rang #5** est affiché séparément car il a montré un signal intéressant
dans notre validation historique. Cette app permet surtout de poursuivre ce
test prospectivement sans réentraîner le modèle.
                """
            )

        with st.expander("🔒 Contrôle du modèle figé"):
            st.markdown(
                f"""
- ID Supabase attendu : **#{NEURAL_MODEL_VERSION_ID}**
- Nom attendu : **`{NEURAL_MODEL_NAME}`**
- SHA256 attendu : **`{NEURAL_ARTIFACT_HASH}`**

`load_forward_neural()` vérifie l'enregistrement Supabase et le hash du fichier
`models/neural/horseprono_neural_v1.pt` avant de charger le réseau.
                """
            )
