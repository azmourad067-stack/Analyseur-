from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from app_core.config import REQUIRED_HISTORY_COLUMNS
from app_core.data import example_race, normalize_history, normalize_race_input, read_uploaded_csv
from app_core.io_utils import dataframe_to_csv_bytes
from app_core.model import HorseRacingModel
from app_core.pmu import get_participants, get_programme, participants_to_df, programme_choices

st.set_page_config(page_title="HorseProno", page_icon="🏇", layout="wide")

DISCLAIMER = (
    "⚠️ **Avertissement** — les probabilités affichées sont des estimations statistiques, "
    "pas des certitudes. Les courses hippiques comportent une forte incertitude et aucun "
    "modèle ne garantit un gain. N'utilisez ces résultats ni comme conseil financier ni comme garantie de pari."
)

@st.cache_resource
def get_model():
    return HorseRacingModel()

@st.cache_data(ttl=600, show_spinner=False)
def cached_programme(d: date):
    return get_programme(d)

@st.cache_data(ttl=300, show_spinner=False)
def cached_participants(d: date, r: int, c: int):
    return participants_to_df(get_participants(d, r, c), d, r, c)


def render_result(result: pd.DataFrame):
    st.subheader("Pronostic classé")
    display_cols = ["rank", "horse_number", "horse_name", "odds", "recent_form", "draw", "weight", "win_probability", "place_probability", "method"]
    st.dataframe(
        result[display_cols].rename(columns={
            "rank": "Rang", "horse_number": "N°", "horse_name": "Cheval", "odds": "Cote",
            "recent_form": "Forme", "draw": "Corde", "weight": "Poids",
            "win_probability": "Prob. victoire", "place_probability": "Prob. placé", "method": "Méthode"
        }),
        use_container_width=True,
        column_config={
            "Prob. victoire": st.column_config.NumberColumn(format="%.1f%%"),
            "Prob. placé": st.column_config.NumberColumn(format="%.1f%%"),
            "Cote": st.column_config.NumberColumn(format="%.2f"),
        },
        hide_index=True,
    )

    best = result.iloc[0]
    st.success(f"🎯 Base du classement : n°{int(best['horse_number']) if pd.notna(best['horse_number']) else '?'} — {best['horse_name']}")

    if "log_odds" not in result.columns:
        from app_core.features import explain_contributions
        contrib = explain_contributions(result)
    else:
        contrib = pd.DataFrame(index=result.index)
    if not contrib.empty:
        st.caption("Lecture du score de repli : la cote, la forme récente, les taux de réussite et la corde sont combinés de façon explicite lorsqu'aucun historique entraîné n'est disponible.")
        show = contrib.copy()
        show.insert(0, "Cheval", result["horse_name"])
        st.dataframe(show.round(3), use_container_width=True, hide_index=True)

    st.download_button("⬇️ Télécharger le pronostic CSV", dataframe_to_csv_bytes(result), "pronostic.csv", "text/csv")

st.title("🏇 HorseProno — Pronostics hippiques probabilistes")
st.markdown("Classement des partants par **probabilité estimée de victoire et de placement**, avec explication des facteurs utilisés.")
st.info(DISCLAIMER)

model = get_model()

with st.sidebar:
    st.header("Mode d'utilisation")
    mode = st.radio("Source de la course", ["API PMU", "Saisie/import CSV", "Course de démonstration"], index=0)
    st.divider()
    st.caption("Le modèle entraîné reste en mémoire pendant la session Streamlit. Pour une utilisation reproductible, fournissez votre historique CSV à chaque nouvelle session ou ajoutez une base persistante.")

race_df = None

if mode == "API PMU":
    st.subheader("Sélection d'une course PMU")
    selected_date = st.date_input("Date", value=date.today(), min_value=date.today() - timedelta(days=30), max_value=date.today() + timedelta(days=2))
    if st.button("Charger le programme", type="secondary"):
        try:
            st.session_state["programme"] = cached_programme(selected_date)
            st.session_state["programme_date"] = selected_date
            st.success("Programme chargé.")
        except Exception as exc:
            st.error(f"Impossible de récupérer le programme PMU : {exc}")

    if "programme" in st.session_state and st.session_state.get("programme_date") == selected_date:
        choices = programme_choices(st.session_state["programme"])
        if not choices:
            st.warning("Le format de réponse du programme PMU n'a pas pu être interprété. Utilise l'import CSV ou la saisie manuelle.")
        else:
            labels = [f"R{x['reunion']}/C{x['course']} — {x['label']}" for x in choices]
            idx = st.selectbox("Course", range(len(labels)), format_func=lambda i: labels[i])
            choice = choices[idx]
            if st.button("Charger les partants", type="primary"):
                try:
                    race_df = cached_participants(selected_date, choice["reunion"], choice["course"])
                    st.session_state["race_df"] = race_df
                except Exception as exc:
                    st.error(f"Impossible de récupérer les partants : {exc}")
            elif "race_df" in st.session_state:
                race_df = st.session_state["race_df"]

elif mode == "Saisie/import CSV":
    st.subheader("Entrer ou importer les partants")
    st.download_button("Télécharger le modèle CSV", open("data/template_historique.csv", "rb").read(), "template_horse_prono.csv", "text/csv")
    uploaded = st.file_uploader("CSV des partants", type=["csv"])
    if uploaded:
        try:
            race_df = normalize_race_input(read_uploaded_csv(uploaded))
            st.success(f"{len(race_df)} partants importés.")
        except Exception as exc:
            st.error(f"CSV invalide : {exc}")
    else:
        st.caption("Saisis directement les colonnes les plus utiles. Les champs statistiques facultatifs peuvent rester vides.")
        columns = ["horse_number", "horse_name", "jockey", "trainer", "odds", "draw", "weight", "recent_form", "career_runs", "career_wins", "career_places"]
        if "manual_race_editor" not in st.session_state:
            st.session_state["manual_race_editor"] = pd.DataFrame([[None, "", "", "", None, None, None, "", None, None, None] for _ in range(8)], columns=columns)
        edited = st.data_editor(st.session_state["manual_race_editor"], num_rows="dynamic", use_container_width=True, hide_index=True, key="manual_editor")
        valid = edited[edited["horse_name"].fillna("").astype(str).str.strip().ne("")].copy()
        if not valid.empty:
            race_df = normalize_race_input(valid.assign(
                race_id="MANUAL", race_date=pd.Timestamp.today().normalize(), discipline="INCONNU",
                hippodrome="SAISIE MANUELLE", distance=None, terrain="INCONNU", field_size=len(valid)
            ))

else:
    st.subheader("Course de démonstration")
    st.caption("Les données ci-dessous sont synthétiques et servent uniquement à tester l'interface.")
    race_df = example_race()

if race_df is not None and not race_df.empty:
    st.divider()
    st.subheader("Partants")
    st.dataframe(race_df[[c for c in ["horse_number","horse_name","jockey","trainer","odds","draw","weight","recent_form"] if c in race_df.columns]], use_container_width=True, hide_index=True)
    if st.button("🔎 Calculer le pronostic", type="primary"):
        try:
            with st.spinner("Calcul des probabilités…"):
                result = model.predict(race_df)
            st.session_state["last_result"] = result
        except Exception as exc:
            st.error(f"Erreur pendant le calcul : {exc}")

if "last_result" in st.session_state:
    st.divider()
    render_result(st.session_state["last_result"])

st.divider()
with st.expander("🧠 Entraîner le modèle sur votre historique"):
    st.write("Importez un historique ligne-par-course/cheval. La variable cible est `finish_position`; le modèle calcule séparément la probabilité de victoire (1er) et de placement (1er–3e).")
    hist = st.file_uploader("Historique CSV pour entraînement", type=["csv"], key="training_csv")
    if hist:
        try:
            hist_df = normalize_history(read_uploaded_csv(hist))
            missing = [c for c in REQUIRED_HISTORY_COLUMNS if c not in hist_df.columns]
            if missing:
                st.error(f"Colonnes manquantes : {', '.join(missing)}")
            else:
                st.write(f"Lignes disponibles : **{len(hist_df):,}**")
                if st.button("🧪 Entraîner / valider", type="secondary"):
                    with st.spinner("Entraînement et validation temporelle…"):
                        metrics = model.fit(hist_df)
                    st.success("Modèle entraîné pour cette session.")
                    st.json(metrics)
        except Exception as exc:
            st.error(f"Impossible d'entraîner le modèle : {exc}")
    else:
        st.caption("Aucun historique n'est fourni dans le dépôt : c'est volontaire, afin de ne pas embarquer de données sous licence inconnue.")

with st.expander("📐 Méthodologie et limites"):
    st.markdown("""
### Variables utilisées
- **Cote** : transformée en probabilité implicite `1 / cote`, puis log-transformée.
- **Forme récente** : score décroissant dans le temps, avec plus de poids aux courses récentes.
- **Taux de victoire / placé** : statistiques carrière si disponibles.
- **Corde** : position normalisée dans la course lorsqu'elle est disponible.
- **Poids, distance, taille du peloton** : variables contextuelles normalisées.

### Modèle
Une **régression logistique régularisée** est entraînée en chronologie sur l'historique, puis évaluée sur une partie temporellement postérieure. Elle fournit une `predict_proba` interprétable ; les probabilités de victoire sont ensuite normalisées dans chaque course afin d'obtenir une distribution qui somme à 100 %.

### Limites
Le modèle ne connaît pas tous les événements latents (incident de course, rythme réel, météo locale instantanée, état physique, tactique, biais de parcours, etc.). Une bonne probabilité statistique peut donc perdre. Les résultats ne constituent pas une garantie de gain.
""")

st.caption("Sources : intégration PMU publique non documentée utilisée avec gestion d'erreurs + import CSV comme voie principale pour l'historique. France Galop et LeTrot sont recommandés pour les enrichissements manuels/contractuels de statistiques spécialisées.")
