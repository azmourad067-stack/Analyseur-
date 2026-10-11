from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent

# Compatible avec :
# - fichier placé dans horse_prono_v2/
# - exécution depuis un dossier voisin lors d'un test local
if (HERE / "app_core").exists():
    HORSE_PRONO_ROOT = HERE
else:
    candidates = [HERE.parent / "horse_prono_v2", HERE.parent, HERE]
    HORSE_PRONO_ROOT = next((p for p in candidates if (p / "app_core").exists()), HERE)

REPO_ROOT = HORSE_PRONO_ROOT.parent
for p in (REPO_ROOT, HORSE_PRONO_ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app_core.db import get_history_as_of, get_supabase_client
from app_core.features import entity_snapshot_from_history
from app_core.forward_utils import non_runner_numbers
from app_core.pmu import (
    get_participants,
    get_programme,
    participants_to_df,
    programme_choices,
)
from neural.forward_capture import NEURAL_ARTIFACT_HASH
from neural.predict import MODEL_PATH, load_neural_v1, predict_dataframe, sha256_file
from scripts.m8_2_top3_place import MODEL_VERSION_ID, _load_frozen_m8, predict_course

PARIS_TZ = ZoneInfo("Europe/Paris")

OFFICIAL_DISCIPLINES = {"ATTELE_AUTOSTART", "ATTELE_VOLTE"}
ODDS_MIN = 5.0
ODDS_MAX_EXCLUSIVE = 8.0
FIELD_MIN = 8
FIELD_MAX = 14
M8_REQUIRED_RANK = 2
NEURAL_MAX_RANK = 3

st.set_page_config(
    page_title="HorseProno — Cheval à jouer",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 HorseProno — Cheval à jouer")
st.caption(
    "Tu choisis une course. L'app regarde uniquement le cheval M8 #2, puis répond : "
    "🟢 BET ou 🔴 NO BET. Aucun autre cheval n'est substitué si le M8 #2 échoue."
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
            if safe_int(x.get("reunion")) == meeting
            and str(x.get("hippodrome") or "").strip()
        }
    )
    suffix = f" — {', '.join(hippos[:2])}" if hippos else ""
    return f"R{meeting}{suffix}"


def course_label(choice: dict) -> str:
    r = safe_int(choice.get("reunion"), 0)
    c = safe_int(choice.get("course"), 0)
    discipline = str(choice.get("discipline") or "INCONNU")
    field_size = safe_int(choice.get("field_size"))
    label = str(choice.get("label") or "").strip()
    bits = [f"R{r}C{c}"]
    if label:
        bits.append(label)
    bits.append(discipline)
    if field_size is not None:
        bits.append(f"{field_size} partants")
    return " — ".join(bits)


def enrich_race_with_choice(race: pd.DataFrame, choice: dict) -> pd.DataFrame:
    out = race.copy()
    if out.empty:
        return out

    metadata = {
        "discipline": choice.get("discipline") or "INCONNU",
        "hippodrome": choice.get("hippodrome") or "INCONNU",
        "distance": choice.get("distance"),
        "terrain": choice.get("terrain") or "INCONNU",
        "field_size": len(out),
    }

    for column, value in metadata.items():
        if column not in out.columns:
            out[column] = value
            continue
        current = out[column]
        missing = current.isna()
        if current.dtype == object:
            text = current.fillna("").astype(str).str.strip().str.upper()
            missing = missing | text.isin({"", "INCONNU", "UNKNOWN", "NONE", "NAN"})
        out.loc[missing, column] = value

    out["field_size"] = len(out)
    return out


def clean_active_runners(race: pd.DataFrame, non_runners: set[int]) -> pd.DataFrame:
    if race.empty:
        return race

    out = race.copy()
    out["horse_number_num"] = pd.to_numeric(out.get("horse_number"), errors="coerce")

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


def _iter_course_objects(programme: dict):
    if not isinstance(programme, dict):
        return
    root = programme.get("programme")
    if not isinstance(root, dict):
        root = programme
    reunions = root.get("reunions") or []
    if not isinstance(reunions, list):
        return
    for reunion in reunions:
        if not isinstance(reunion, dict):
            continue
        reunion_number = safe_int(
            reunion.get("numOfficiel")
            or reunion.get("numReunion")
            or reunion.get("numero")
        )
        for course in reunion.get("courses") or []:
            if not isinstance(course, dict):
                continue
            course_number = safe_int(
                course.get("numOrdre")
                or course.get("numCourse")
                or course.get("numOfficiel")
                or course.get("numero")
            )
            yield reunion_number, course_number, course


def is_quinte_plus(programme: dict, reunion: int, course: int) -> bool:
    for r, c, course_obj in _iter_course_objects(programme):
        if r != reunion or c != course:
            continue
        paris = course_obj.get("paris") or course_obj.get("typesParis") or []
        if not isinstance(paris, list):
            return False
        for pari in paris:
            if not isinstance(pari, dict):
                continue
            code = str(
                pari.get("codePari")
                or pari.get("typePari")
                or pari.get("code")
                or ""
            ).strip().upper()
            if code in {"E_QUINTE_PLUS", "QUINTE_PLUS"}:
                return True
        return False
    return False


@st.cache_resource(show_spinner=False)
def load_m8_model():
    client = get_supabase_client()
    if client is None:
        raise RuntimeError(
            "Supabase non configuré. Ajoute SUPABASE_URL et SUPABASE_KEY "
            "dans les secrets Streamlit."
        )
    return _load_frozen_m8(client)


@st.cache_resource(show_spinner=False)
def load_neural_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Fichier Neural absent : {MODEL_PATH}")

    local_hash = sha256_file(MODEL_PATH)
    if local_hash != NEURAL_ARTIFACT_HASH:
        raise RuntimeError(
            "Le Neural local n'est pas le modèle #9 figé attendu. "
            f"Attendu : {NEURAL_ARTIFACT_HASH} | Trouvé : {local_hash}"
        )
    return load_neural_v1()


@st.cache_data(ttl=300, show_spinner=False)
def load_program(date_iso: str):
    target_date = datetime.strptime(date_iso, "%Y-%m-%d").date()
    programme = get_programme(target_date)
    choices = [dict(x) for x in programme_choices(programme)]
    choices.sort(
        key=lambda x: (
            safe_int(x.get("reunion"), 999),
            safe_int(x.get("course"), 999),
        )
    )
    return programme, choices


@st.cache_data(ttl=900, show_spinner=False)
def load_snapshots(date_iso: str):
    target_date = datetime.strptime(date_iso, "%Y-%m-%d").date()
    history = get_history_as_of(target_date)
    return entity_snapshot_from_history(history, target_date)


@st.cache_data(ttl=120, show_spinner=False)
def load_neural_race(date_iso: str, reunion: int, course: int, choice: dict):
    target_date = datetime.strptime(date_iso, "%Y-%m-%d").date()
    payload = get_participants(target_date, int(reunion), int(course))
    race = participants_to_df(payload, target_date, int(reunion), int(course))
    race = enrich_race_with_choice(race, choice)
    non_runners = non_runner_numbers(payload)
    race = clean_active_runners(race, non_runners)
    return race, sorted(non_runners)


def normalize_m8_rank(m8_full: pd.DataFrame) -> pd.DataFrame:
    out = m8_full.copy()
    if "rank" in out.columns:
        rank = pd.to_numeric(out["rank"], errors="coerce")
    else:
        win_p = pd.to_numeric(out.get("win_probability"), errors="coerce")
        rank = win_p.rank(method="first", ascending=False)
    out["m8_win_rank"] = rank.astype("Int64")
    return out


def build_verdict(
    *,
    m8_full: pd.DataFrame,
    neural_ranked: pd.DataFrame,
    discipline: str,
    field_size: int,
    quinté_plus: bool,
) -> dict:
    m8 = normalize_m8_rank(m8_full)
    m8_candidate = m8[m8["m8_win_rank"].eq(M8_REQUIRED_RANK)].copy()

    if m8_candidate.empty:
        return {
            "official_bet": False,
            "quinte_shadow": False,
            "reason": "Impossible d'identifier le cheval M8 #2.",
        }

    row = m8_candidate.iloc[0]
    horse_number = safe_int(row.get("horse_number"))
    horse_name = str(row.get("horse_name") or "Cheval")
    odds = safe_float(row.get("odds"))
    win_probability = safe_float(row.get("win_probability"))

    n9 = neural_ranked.copy()
    n9["horse_number_num"] = pd.to_numeric(n9.get("horse_number"), errors="coerce")
    nrow = n9[n9["horse_number_num"].eq(horse_number)]
    neural_rank = safe_int(nrow.iloc[0].get("neural_rank")) if not nrow.empty else None
    neural_probability = (
        safe_float(nrow.iloc[0].get("neural_top3_probability")) if not nrow.empty else None
    )

    discipline_ok = str(discipline or "").strip().upper() in OFFICIAL_DISCIPLINES
    field_ok = FIELD_MIN <= int(field_size) <= FIELD_MAX
    odds_ok = odds is not None and ODDS_MIN <= odds < ODDS_MAX_EXCLUSIVE
    neural_ok = neural_rank is not None and neural_rank <= NEURAL_MAX_RANK

    official_bet = discipline_ok and field_ok and odds_ok and neural_ok
    quinte_shadow = quinté_plus and discipline_ok and odds_ok and neural_ok

    checks = {
        "discipline_ok": discipline_ok,
        "field_ok": field_ok,
        "odds_ok": odds_ok,
        "neural_ok": neural_ok,
    }

    failed = []
    if not discipline_ok:
        failed.append("discipline différente d'Autostart/Volté")
    if not field_ok:
        failed.append(f"{field_size} partants au lieu de 8–14")
    if not odds_ok:
        failed.append(
            "cote M8 #2 hors 5–8"
            if odds is not None
            else "cote M8 #2 indisponible"
        )
    if not neural_ok:
        failed.append("M8 #2 hors Top3 Neural")

    return {
        "official_bet": official_bet,
        "quinte_shadow": quinte_shadow,
        "horse_number": horse_number,
        "horse_name": horse_name,
        "odds": odds,
        "win_probability": win_probability,
        "neural_rank": neural_rank,
        "neural_probability": neural_probability,
        "discipline": discipline,
        "field_size": field_size,
        "is_quinte": quinté_plus,
        "checks": checks,
        "reason": "; ".join(failed) if failed else "Toutes les conditions V1 sont validées.",
    }


with st.sidebar:
    st.header("⚙️ Course")
    selected_date = st.date_input(
        "Date",
        value=datetime.now(PARIS_TZ).date(),
        format="DD/MM/YYYY",
    )
    st.divider()
    st.markdown("**Règle officielle figée :**")
    st.caption(
        "Autostart/Volté · 8–14 partants · cheval M8 #2 · cote 5–8 · Top3 Neural."
    )
    st.markdown("**Quinté shadow :**")
    st.caption(
        "Même règle, mais sans la limite 8–14 partants. Signal expérimental uniquement."
    )


date_iso = selected_date.isoformat()

try:
    raw_programme, choices = load_program(date_iso)
except Exception as exc:
    st.error(f"Impossible de charger le programme PMU : {exc}")
    st.stop()

if not choices:
    st.warning("Aucune course disponible.")
    st.stop()

meetings = sorted(
    {
        safe_int(c.get("reunion"))
        for c in choices
        if safe_int(c.get("reunion")) is not None
    }
)
meeting_map = {meeting_label(m, choices): m for m in meetings}
selected_meeting_label = st.selectbox("📍 Réunion", list(meeting_map.keys()))
selected_meeting = meeting_map[selected_meeting_label]

meeting_choices = [c for c in choices if safe_int(c.get("reunion")) == selected_meeting]
course_map = {course_label(c): c for c in meeting_choices}
selected_course_label = st.selectbox("🏁 Course", list(course_map.keys()))
selected_choice = course_map[selected_course_label]
selected_course = safe_int(selected_choice.get("course"))

is_quinte_selected = is_quinte_plus(raw_programme, selected_meeting, selected_course)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Course", f"R{selected_meeting}C{selected_course}")
m2.metric("Discipline", str(selected_choice.get("discipline") or "—"))
m3.metric("Partants annoncés", safe_int(selected_choice.get("field_size"), "—"))
m4.metric("Quinté+", "✅ Oui" if is_quinte_selected else "Non")

if st.button("🎯 DONNE-MOI LE CHEVAL", type="primary", use_container_width=True):
    try:
        with st.spinner("M8 #8 + Neural #9…"):
            target_date = datetime.strptime(date_iso, "%Y-%m-%d").date()
            m8_model = load_m8_model()
            neural_bundle = load_neural_model()
            snapshots = load_snapshots(date_iso)

            m8_result = predict_course(
                model=m8_model,
                snapshots=snapshots,
                target_date=target_date,
                choice=selected_choice,
            )

            neural_race, non_runners = load_neural_race(
                date_iso,
                selected_meeting,
                selected_course,
                dict(selected_choice),
            )
            if neural_race.empty:
                raise RuntimeError("Aucun partant actif exploitable.")

            neural_ranked = predict_dataframe(neural_race, neural_bundle).copy()
            neural_ranked["neural_top3_probability"] = pd.to_numeric(
                neural_ranked["neural_top3_probability"], errors="coerce"
            )
            neural_ranked = neural_ranked.sort_values(
                ["neural_top3_probability", "horse_number"],
                ascending=[False, True],
                na_position="last",
            ).reset_index(drop=True)
            neural_ranked["neural_rank"] = range(1, len(neural_ranked) + 1)

            verdict = build_verdict(
                m8_full=m8_result["full"],
                neural_ranked=neural_ranked,
                discipline=str(m8_result.get("discipline") or selected_choice.get("discipline") or ""),
                field_size=len(neural_ranked),
                quinté_plus=is_quinte_selected,
            )

            st.session_state["direct_bet_result"] = {
                "date": date_iso,
                "reunion": selected_meeting,
                "course": selected_course,
                "verdict": verdict,
                "non_runners": non_runners,
            }

    except Exception as exc:
        st.error(f"Analyse impossible : {exc}")


result = st.session_state.get("direct_bet_result")
if result:
    same_course = (
        result.get("date") == date_iso
        and safe_int(result.get("reunion")) == selected_meeting
        and safe_int(result.get("course")) == selected_course
    )

    if same_course:
        v = result["verdict"]
        st.divider()

        if v.get("horse_number") is None:
            st.error("🔴 NO BET — cheval M8 #2 introuvable.")
        else:
            horse = f"N°{v['horse_number']} — {v['horse_name']}"
            odds_text = f"{v['odds']:.2f}" if v.get("odds") is not None else "—"

            if v["official_bet"]:
                st.success(f"# 🟢 PARIER : {horse}")
                st.markdown(
                    f"### Simple Gagnant · cote actuelle **{odds_text}** · "
                    f"Neural **#{v['neural_rank']}**"
                )
                st.caption(
                    "Ce cheval est le M8 #2. Tous les filtres de la règle frozen sont validés."
                )
            else:
                st.error("# 🔴 NO BET")
                st.markdown(f"### M8 #2 = {horse} · cote **{odds_text}**")
                st.write(f"**Pourquoi :** {v['reason']}")
                st.caption(
                    "On ne remplace jamais le M8 #2 par un autre cheval coté 5–8."
                )

            # Signal Quinté séparé, sans le présenter comme règle validée.
            if v.get("is_quinte") and v.get("quinte_shadow") and not v.get("official_bet"):
                st.warning(
                    f"🟡 **QUINTÉ SHADOW : {horse}** · cote {odds_text}. "
                    "Même signal M8 #2 + Neural Top3 + cote 5–8, mais la course sort de la "
                    "limite 8–14 partants. À suivre séparément, pas à confondre avec le BET V1."
                )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("M8", "#2")
            c2.metric(
                "Cote M8 #2",
                odds_text,
                "✅ 5–8" if v["checks"]["odds_ok"] else "❌ hors filtre",
            )
            c3.metric(
                "Rang Neural",
                f"#{v['neural_rank']}" if v.get("neural_rank") is not None else "—",
                "✅ Top3" if v["checks"]["neural_ok"] else "❌ hors Top3",
            )
            c4.metric(
                "Partants actifs",
                str(v["field_size"]),
                "✅ 8–14" if v["checks"]["field_ok"] else "❌ hors filtre",
            )

            d_ok = v["checks"]["discipline_ok"]
            st.caption(
                f"Discipline : **{v['discipline']}** {'✅' if d_ok else '❌'} · "
                f"Quinté+ : **{'oui' if v['is_quinte'] else 'non'}**"
            )

            if result.get("non_runners"):
                st.caption(
                    "Non-partants retirés avant calcul : "
                    + ", ".join(f"N°{x}" for x in result["non_runners"])
                )

        st.info(
            "Le signal est un outil expérimental de sélection. Les résultats historiques ne "
            "garantissent pas un profit futur ; le forward 100 paris sert justement à le vérifier."
        )
