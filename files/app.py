"""Application Streamlit — Outil de pronostic hippique (galop plat).

Deux scores calculés par course, à partir de 6 paramètres saisis par cheval :
- Score intrinsèque : estime la valeur du cheval indépendamment de la cote
  (forme, aptitude distance, aptitude terrain, poids porté, jockey/entraîneur).
- Score de valeur : compare le score intrinsèque à la cote du marché pour
  repérer les chevaux potentiellement sous-cotés ou surcotés.
"""

import streamlit as st
import pandas as pd

from models import (
    COL_NOM,
    COL_FORME,
    COL_POIDS,
    COL_DISTANCE,
    COL_TERRAIN,
    COL_JOCKEY,
    COL_COTE,
    OPTIONS_APTITUDE,
    OPTIONS_JOCKEY,
    dataframe_vide,
)
from scoring import calculer_score_intrinseque, calculer_score_valeur, POIDS_DEFAUT
from data_logger import preparer_export, dataframe_vers_csv_bytes

st.set_page_config(page_title="Pronostic Hippique", page_icon="🐎", layout="wide")

st.title("🐎 Outil de pronostic hippique — Galop plat")
st.caption(
    "Score intrinsèque (indépendant de la cote) + score de valeur "
    "(comparaison à la cote du marché)."
)

# --- 1. Informations générales de la course ---
st.subheader("1. Informations générales de la course")
col1, col2, col3 = st.columns(3)
with col1:
    nb_chevaux = st.number_input("Nombre de partants", min_value=2, max_value=25, value=8)
with col2:
    distance = st.number_input(
        "Distance de la course (m)", min_value=800, max_value=5000, value=2000, step=100
    )
with col3:
    terrain_jour = st.selectbox("Terrain du jour", ["Bon", "Souple", "Lourd", "Léger"])

# --- Pondération ajustable (barre latérale) ---
with st.sidebar:
    st.header("Pondération du score intrinsèque")
    st.caption("Les 5 poids doivent idéalement sommer à 100%.")
    poids_forme = st.slider("Forme récente", 0, 100, int(POIDS_DEFAUT["forme"] * 100))
    poids_distance = st.slider("Aptitude distance", 0, 100, int(POIDS_DEFAUT["distance"] * 100))
    poids_terrain = st.slider("Aptitude terrain", 0, 100, int(POIDS_DEFAUT["terrain"] * 100))
    poids_poids = st.slider("Poids porté", 0, 100, int(POIDS_DEFAUT["poids"] * 100))
    poids_jockey = st.slider("Jockey / Entraîneur", 0, 100, int(POIDS_DEFAUT["jockey"] * 100))

    total_poids = poids_forme + poids_distance + poids_terrain + poids_poids + poids_jockey
    if total_poids != 100:
        st.warning(f"Somme actuelle des poids : {total_poids}% (idéalement 100%)")

    poids_utilisateur = {
        "forme": poids_forme / 100,
        "distance": poids_distance / 100,
        "terrain": poids_terrain / 100,
        "poids": poids_poids / 100,
        "jockey": poids_jockey / 100,
    }

    st.divider()
    st.caption(
        "Cet outil produit une estimation basée sur les paramètres saisis, "
        "pas une garantie de résultat. À utiliser comme aide à la réflexion."
    )

# --- 2. Saisie des chevaux ---
st.subheader("2. Données des partants")
st.caption("Renseigne les 6 paramètres pour chaque cheval du peloton.")

if "df_chevaux" not in st.session_state or len(st.session_state.df_chevaux) != nb_chevaux:
    st.session_state.df_chevaux = dataframe_vide(nb_chevaux)

df_edite = st.data_editor(
    st.session_state.df_chevaux,
    num_rows="fixed",
    use_container_width=True,
    column_config={
        COL_FORME: st.column_config.NumberColumn(
            "Forme (podiums /5)", min_value=0, max_value=5, step=1
        ),
        COL_POIDS: st.column_config.NumberColumn(
            "Poids (kg)", min_value=45.0, max_value=70.0, step=0.5
        ),
        COL_DISTANCE: st.column_config.SelectboxColumn(
            "Aptitude distance", options=OPTIONS_APTITUDE
        ),
        COL_TERRAIN: st.column_config.SelectboxColumn(
            "Aptitude terrain", options=OPTIONS_APTITUDE
        ),
        COL_JOCKEY: st.column_config.SelectboxColumn(
            "Jockey/Entraîneur", options=OPTIONS_JOCKEY
        ),
        COL_COTE: st.column_config.NumberColumn(
            "Cote probable", min_value=1.01, step=0.1
        ),
    },
    key="editeur_chevaux",
)

# --- 3. Calcul et affichage des résultats ---
if st.button("Calculer les pronostics", type="primary"):
    resultats = calculer_score_intrinseque(df_edite, poids_utilisateur)
    resultats = calculer_score_valeur(resultats)

    st.subheader("3. Résultats")

    tab_classement, tab_valeur, tab_graph = st.tabs(
        ["Classement (score intrinsèque)", "Score de valeur", "Graphique comparatif"]
    )

    with tab_classement:
        classement = resultats.sort_values("score_intrinseque", ascending=False)
        st.dataframe(
            classement[
                [
                    COL_NOM,
                    "score_intrinseque",
                    "note_forme",
                    "note_distance",
                    "note_terrain",
                    "note_poids",
                    "note_jockey",
                ]
            ].round(2),
            use_container_width=True,
            hide_index=True,
        )

    with tab_valeur:
        valeur = resultats.sort_values("ecart_valeur", ascending=False)
        affichage_valeur = valeur[
            [COL_NOM, COL_COTE, "proba_modele", "proba_marche", "ecart_valeur"]
        ].copy()
        affichage_valeur["proba_modele"] = (
            (affichage_valeur["proba_modele"] * 100).round(1).astype(str) + " %"
        )
        affichage_valeur["proba_marche"] = (
            (affichage_valeur["proba_marche"] * 100).round(1).astype(str) + " %"
        )
        affichage_valeur["ecart_valeur"] = (valeur["ecart_valeur"] * 100).round(1)
        st.dataframe(affichage_valeur, use_container_width=True, hide_index=True)
        st.caption(
            "Écart positif = cheval potentiellement sous-coté par le marché "
            "selon le modèle. Écart négatif = potentiellement surcoté."
        )

    with tab_graph:
        graph_data = resultats.set_index(COL_NOM)[["proba_modele", "proba_marche"]]
        graph_data.columns = ["Modèle", "Marché (cote)"]
        st.bar_chart(graph_data)

    st.subheader("4. Export")
    export_df = preparer_export(resultats, {"distance": distance, "terrain": terrain_jour})
    csv_bytes = dataframe_vers_csv_bytes(export_df)
    st.download_button(
        "Télécharger les résultats en CSV",
        data=csv_bytes,
        file_name=f"pronostic_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )
    st.caption(
        "Astuce : complète la colonne « Résultat réel » après la course. "
        "En accumulant ces fichiers au fil du temps, tu te constitues une "
        "base historique utile pour calibrer les poids plus tard."
    )
