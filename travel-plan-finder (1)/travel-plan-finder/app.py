"""Travel Plan Finder — point d'entrée Streamlit.

Lancer en local :  streamlit run app.py
Déploiement      :  voir README.md (GitHub → Streamlit Community Cloud)
"""

import streamlit as st

from travel_planner.ui import (
    inject_css, render_search_form, render_sidebar, run_and_render,
)

st.set_page_config(
    page_title="Travel Plan Finder",
    page_icon="🧭",
    layout="wide",
)

inject_css()
render_sidebar()

st.title("🧭 Travel Plan Finder")
st.caption(
    "Le meilleur plan de voyage, pas juste le moins cher : "
    "transport + hébergement comparés et notés selon votre budget."
)

criteria = render_search_form()

if criteria is not None:
    run_and_render(criteria)
else:
    st.info(
        "👆 Renseignez départ, destination et budget, puis lancez la recherche. "
        "Exemple pour démarrer : **Paris → Lyon, budget 600 € ± 100 €, 3 nuits, 2 voyageurs**."
    )
