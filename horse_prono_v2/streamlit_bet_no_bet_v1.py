from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
HORSE_PRONO_ROOT = HERE if (HERE / "app_core").exists() else HERE.parent
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
from bet_no_bet_v1 import evaluate_bet_no_bet
from neural.forward_capture import NEURAL_ARTIFACT_HASH
from neural.predict import MODEL_PATH, load_neural_v1, predict_dataframe, sha256_file
from scripts.m8_2_top3_place import MODEL_VERSION_ID, _load_frozen_m8, predict_course

PARIS_TZ = ZoneInfo("Europe/Paris")

st.set_page_config(
    page_title="HorseProno BET / NO BET V1",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 HorseProno — BET / NO BET V1")
st.caption(
    "M8 #8 + Neural #9 · le système décide d'abord si la course mérite un pari. "
    "Par défaut : NO BET."
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
    distance = safe_float(choice.get("distance"))
    field_size = safe_int(choice.get("field_size"))

    bits = [f"R{r}C{c}", discipline]
    if distance is not None:
        bits.append(f"{int(distance)} m")
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
            missing = missing | text.isin(
                {"", "INCONNU", "UNKNOWN", "NONE", "NAN"}
            )
        out.loc[missing, column] = value

    out["field_size"] = len(out)
    return out


def clean_active_runners(
    race: pd.DataFrame,
    non_runners: set[int],
) -> pd.DataFrame:
    if race.empty:
        return race

    out = race.copy()
    out["horse_number_num"] = pd.to_numeric(
        out.get("horse_number"), errors="coerce"
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
            "Le Neural présent dans GitHub n'est pas le modèle figé attendu. "
            f"Attendu : {NEURAL_ARTIFACT_HASH} | Trouvé : {local_hash}"
        )
    return load_neural_v1()


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


@st.cache_data(ttl=120, show_spinner=False)
def load_neural_race(
    date_iso: str,
    reunion: int,
    course: int,
    choice: dict,
):
    target_date = datetime.strptime(date_iso, "%Y-%m-%d").date()
    payload = get_participants(target_date, int(reunion), int(course))
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


def decision_icon(value: str) -> str:
    return {
        "BET": "🟢",
        "PRUDENT": "🟡",
        "NO BET": "🔴",
    }.get(str(value), "⚪")


def decision_table(decisions: list[dict]) -> pd.DataFrame:
    rows = []
    for d in decisions:
        horse = ""
        if d.get("horse_number") is not None:
            horse = f"N°{d['horse_number']}"
            if d.get("horse_name"):
                horse += f" · {d['horse_name']}"

        rows.append(
            {
                "Jeu": d.get("game"),
                "Décision": (
                    f"{decision_icon(d.get('decision'))} {d.get('decision')}"
                ),
                "Confiance": f"{safe_int(d.get('confidence'), 0)}/100",
                "Cheval": horse or "—",
                "Mise": d.get("stake_level") or "—",
                "Pourquoi": d.get("reason") or "",
            }
        )
    return pd.DataFrame(rows)


def ranking_comparison(
    m8_full: pd.DataFrame,
    neural_ranked: pd.DataFrame,
) -> pd.DataFrame:
    m8 = m8_full.copy()
    n9 = neural_ranked.copy()

    m8["horse_number"] = pd.to_numeric(
        m8["horse_number"], errors="coerce"
    ).astype("Int64")
    n9["horse_number"] = pd.to_numeric(
        n9["horse_number"], errors="coerce"
    ).astype("Int64")

    m8_small = pd.DataFrame(
        {
            "N°": m8["horse_number"],
            "Cheval": m8["horse_name"].fillna("").astype(str),
            "Rang M8 Place": pd.to_numeric(
                m8["m8_place_rank"], errors="coerce"
            ).astype("Int64"),
            "P(placé) M8": (
                pd.to_numeric(
                    m8["place_probability"], errors="coerce"
                )
                * 100
            ).round(1),
            "P(gagnant) M8": (
                pd.to_numeric(
                    m8["win_probability"], errors="coerce"
                )
                * 100
            ).round(1),
        }
    )

    n_small = pd.DataFrame(
        {
            "N°": n9["horse_number"],
            "Rang Neural": pd.to_numeric(
                n9["neural_rank"], errors="coerce"
            ).astype("Int64"),
            "P(Top3) Neural": (
                pd.to_numeric(
                    n9["neural_top3_probability"], errors="coerce"
                )
                * 100
            ).round(1),
        }
    )

    out = m8_small.merge(n_small, on="N°", how="left")
    return out.sort_values(
        ["Rang M8 Place", "Rang Neural"],
        na_position="last",
    ).reset_index(drop=True)


with st.sidebar:
    st.header("⚙️ BET / NO BET")

    selected_date = st.date_input(
        "Date des courses",
        value=datetime.now(PARIS_TZ).date(),
        format="DD/MM/YYYY",
    )

    st.divider()
    st.markdown(f"**Moteur principal :** M8 #{MODEL_VERSION_ID}")
    st.markdown("**Contrôle :** Neural #9")
    st.markdown("**Version décision :** V1")
    st.caption("Aucun pari automatique : l'app affiche seulement une décision.")

    st.info(
        "Règle placé forte : P M8 ≥ 0,70 + même n°1 + même Top 3. "
        "Historique forward observé : 78,8 % placés sur 100 courses."
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
    c for c in choices
    if safe_int(c.get("reunion")) == selected_meeting
]
course_map = {course_label(c): c for c in meeting_choices}

selected_course_label = st.selectbox(
    "🏁 Choisis la course",
    options=list(course_map.keys()),
)
selected_choice = course_map[selected_course_label]
selected_course = safe_int(selected_choice.get("course"))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Réunion", f"R{selected_meeting}")
c2.metric("Course", f"C{selected_course}")
c3.metric("Discipline", str(selected_choice.get("discipline") or "—"))
announced = safe_int(selected_choice.get("field_size"))
c4.metric(
    "Partants annoncés",
    str(announced) if announced is not None else "—",
)

st.caption(
    f"{selected_choice.get('hippodrome') or 'Hippodrome inconnu'} · "
    f"{selected_choice.get('distance') or '—'} m · "
    f"terrain : {selected_choice.get('terrain') or 'INCONNU'}"
)

if st.button(
    "🚦 Analyser BET / NO BET",
    type="primary",
    use_container_width=True,
):
    try:
        with st.spinner("M8 + Neural + filtre de décision…"):
            target_date = datetime.strptime(
                date_iso, "%Y-%m-%d"
            ).date()

            m8_model = load_m8_model()
            neural_bundle = load_neural_model()
            snapshots = load_snapshots(date_iso)

            m8_result = predict_course(
                model=m8_model,
                snapshots=snapshots,
                target_date=target_date,
                choice=selected_choice,
            )
            m8_full = m8_result["full"].copy()

            neural_race, non_runners = load_neural_race(
                date_iso,
                selected_meeting,
                selected_course,
                dict(selected_choice),
            )
            if neural_race.empty:
                raise RuntimeError(
                    "Aucun partant actif exploitable pour Neural."
                )

            neural_ranked = predict_dataframe(
                neural_race,
                neural_bundle,
            ).copy()
            neural_ranked["neural_top3_probability"] = pd.to_numeric(
                neural_ranked["neural_top3_probability"],
                errors="coerce",
            )
            neural_ranked = neural_ranked.sort_values(
                ["neural_top3_probability", "horse_number"],
                ascending=[False, True],
                na_position="last",
            ).reset_index(drop=True)
            neural_ranked["neural_rank"] = range(
                1, len(neural_ranked) + 1
            )

            decision_result = evaluate_bet_no_bet(
                m8_full,
                neural_ranked,
                field_size=len(neural_ranked),
                discipline=str(
                    selected_choice.get("discipline") or ""
                ),
            )

            st.session_state["bet_no_bet_v1_result"] = {
                "date": date_iso,
                "reunion": selected_meeting,
                "course": selected_course,
                "m8": m8_result,
                "neural": neural_ranked,
                "non_runners": non_runners,
                "decision": decision_result,
            }

    except Exception as exc:
        st.error(f"Analyse BET / NO BET impossible : {exc}")


result = st.session_state.get("bet_no_bet_v1_result")

if result:
    same_race = (
        result.get("date") == date_iso
        and safe_int(result.get("reunion")) == selected_meeting
        and safe_int(result.get("course")) == selected_course
    )

    if same_race:
        decision = result["decision"]
        signals = decision["signals"]
        decisions = decision["decisions"]

        st.divider()
        st.subheader("🚦 Décision par type de pari")

        cols = st.columns(5)
        for idx, d in enumerate(decisions):
            with cols[idx]:
                icon = decision_icon(d["decision"])
                st.markdown(f"### {icon} {d['game']}")
                st.metric(
                    "Confiance",
                    f"{safe_int(d['confidence'], 0)}/100",
                )
                st.markdown(f"**{d['decision']}**")
                if d.get("horse_number") is not None:
                    horse_text = f"N°{d['horse_number']}"
                    if d.get("horse_name"):
                        horse_text += f" · {d['horse_name']}"
                    st.caption(horse_text)

        best = sorted(
            [
                d for d in decisions
                if d["decision"] in {"BET", "PRUDENT"}
            ],
            key=lambda d: (
                1 if d["decision"] == "BET" else 0,
                safe_int(d.get("confidence"), 0),
            ),
            reverse=True,
        )

        if best:
            top = best[0]
            if top["decision"] == "BET":
                st.success(
                    f"Signal principal : {top['game']} — "
                    f"{top['confidence']}/100. {top['reason']}"
                )
            else:
                st.warning(
                    "Aucun BET fort. Meilleur signal prudent : "
                    f"{top['game']} — {top['confidence']}/100."
                )
        else:
            st.error(
                "🔴 NO BET global — aucune stratégie ne franchit "
                "le filtre V1."
            )

        st.markdown("### 📊 Tableau des décisions")
        st.dataframe(
            decision_table(decisions),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### 🧠 Pourquoi le système décide ainsi ?")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric(
            "Même n°1 M8/Neural",
            "✅ Oui" if signals["top1_agree"] else "❌ Non",
        )
        s2.metric(
            "Top 3 commun",
            f"{signals['top3_overlap']}/3",
        )
        s3.metric(
            "Top 5 commun",
            f"{signals['top5_overlap']}/5",
        )
        s4.metric(
            "P(placé) M8 n°1",
            f"{signals['m8_place_p1'] * 100:.1f} %",
        )
        s5.metric(
            "Concentration Top 7",
            f"{signals['top7_concentration'] * 100:.1f} %",
        )

        st.caption(
            "Écart P(placé) Top1–Top2 : "
            f"{signals['place_gap'] * 100:.1f} pp · "
            "edge gagnant Top1 M8 : "
            f"{signals['win_edge'] * 100:+.1f} pp."
        )

        with st.expander("📚 Justification et niveau de preuve"):
            for d in decisions:
                st.markdown(
                    f"**{decision_icon(d['decision'])} "
                    f"{d['game']} — {d['decision']} "
                    f"({d['confidence']}/100)**"
                )
                st.write(d["reason"])
                st.caption(d["evidence"])

            st.warning(
                "Les taux historiques décrivent les courses observées. "
                "Ils ne garantissent ni un placement futur ni un profit. "
                "Le V1 est volontairement plus strict en Simple Gagnant "
                "et Quinté."
            )

        st.markdown("### 🔬 Comparaison M8 / Neural")
        comparison = ranking_comparison(
            result["m8"]["full"],
            result["neural"],
        )
        st.dataframe(
            comparison.head(min(10, len(comparison))),
            use_container_width=True,
            hide_index=True,
        )

        if result.get("non_runners"):
            st.caption(
                "Non-partants retirés : "
                + ", ".join(
                    f"N°{x}" for x in result["non_runners"]
                )
            )

        with st.expander("📈 Repères forward utilisés pour V1"):
            stats = decision["reference_stats"]
            st.markdown(
                f"""
- Courses avec résultats exploités : **{stats['forward_result_races']}**
- Top1 M8 placé sans filtre : **{stats['place_baseline_rate'] * 100:.1f} %**
- Même n°1 + même Top3 : **{stats['place_full_agreement_rate'] * 100:.1f} %**
- Règle placé forte : **{stats['place_strong_rule_rate'] * 100:.1f} %**
  sur **{stats['place_strong_rule_n']} courses**
- Podium réel dans Top7 M8 sans filtre : **{stats['trio_top7_baseline_rate'] * 100:.1f} %**
- Avec accord complet : **{stats['trio_top7_full_agreement_rate'] * 100:.1f} %**
                """
            )

        export = decision_table(decisions).to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Exporter les décisions CSV",
            data=export,
            file_name=(
                f"bet_no_bet_v1_{date_iso}_"
                f"R{selected_meeting}C{selected_course}.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )
