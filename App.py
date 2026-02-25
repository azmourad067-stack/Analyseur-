import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import re
import plotly.graph_objects as go
import plotly.express as px
from itertools import combinations

# Configuration Streamlit
st.set_page_config(
    page_title="🐴 PronoHippo",
    page_icon="🐴",
    layout="wide"
)

# ===================== SCORING ENGINE =====================

class ScoringEngine:
    """Moteur de scoring intelligent"""
    
    def __init__(self):
        self.weights = {
            'reussite_driver': 0.20,
            'reussite_entraineur': 0.20,
            'victoires': 0.20,
            'musique': 0.20,
            'autres': 0.20
        }
    
    def parse_musique_score(self, musique_str):
        """Convertit musique en score"""
        if not musique_str:
            return 50
        score = sum(100 if (c.isdigit() and c != '0') else 50 for c in str(musique_str)[-10:])
        count = min(len(str(musique_str)[-10:]), 10)
        return score / count if count > 0 else 50
    
    def calculate_score(self, horse_data):
        """Calcule le score global"""
        score = 0
        
        driver_pct = float(horse_data.get('reussite_driver', 10))
        score += driver_pct * self.weights['reussite_driver'] / 5
        
        trainer_pct = float(horse_data.get('reussite_entraineur', 10))
        score += trainer_pct * self.weights['reussite_entraineur'] / 5
        
        victoires = float(horse_data.get('victoires', 0))
        vict_score = min(100, (victoires / 50) * 100) if victoires > 0 else 20
        score += vict_score * self.weights['victoires'] / 100
        
        musique = horse_data.get('musique', '')
        musique_score = self.parse_musique_score(musique)
        score += musique_score * self.weights['musique'] / 100
        
        score += 50 * self.weights['autres'] / 100
        
        return min(100, max(0, score))

# ===================== INTERFACE =====================

def main():
    st.title("🐴 PronoHippo")
    st.markdown("### Pronostics Hippiques Intelligents")
    st.markdown("---")
    
    # Mode d'entrée
    tab1, tab2 = st.tabs(["📝 Entrée Manuelle", "📋 Résumé"])
    
    with tab1:
        st.header("Entrez les données des chevaux")
        
        num_horses = st.slider("Nombre de chevaux:", 2, 20, 5)
        
        horses_data = []
        
        with st.form("entry_form"):
            cols = st.columns(5)
            
            for i in range(num_horses):
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        nom = st.text_input(f"Cheval {i+1}", key=f"nom_{i}", value=f"Cheval {i+1}")
                    
                    with col2:
                        driver = st.number_input(f"Driver %", 0, 100, 15, key=f"driver_{i}")
                    
                    with col3:
                        trainer = st.number_input(f"Trainer %", 0, 100, 12, key=f"trainer_{i}")
                    
                    with col4:
                        vict = st.number_input(f"Victoires", 0, 500, 25, key=f"vict_{i}")
                    
                    with col5:
                        musique = st.text_input(f"Musique", "3a0aDa", key=f"mus_{i}")
                    
                    horses_data.append({
                        'numero': i + 1,
                        'nom': nom,
                        'reussite_driver': int(driver),
                        'reussite_entraineur': int(trainer),
                        'victoires': int(vict),
                        'musique': musique
                    })
            
            submitted = st.form_submit_button("🎯 Analyser", use_container_width=True, type="primary")
        
        if submitted:
            st.session_state.horses_df = pd.DataFrame(horses_data)
            st.session_state.analyze = True
    
    with tab2:
        if 'analyze' in st.session_state and st.session_state.analyze:
            display_results(st.session_state.horses_df)

def display_results(horses_df):
    """Affiche les résultats"""
    st.header("📊 Résultats de l'Analyse")
    
    # Scoring
    engine = ScoringEngine()
    rankings = {}
    
    for idx, row in horses_df.iterrows():
        score = engine.calculate_score(row.to_dict())
        rankings[idx] = {
            'numero': row['numero'],
            'nom': row['nom'],
            'score': score,
            'driver': row['reussite_driver'],
            'trainer': row['reussite_entraineur'],
            'victoires': row['victoires']
        }
    
    # Trier
    sorted_rankings = dict(sorted(rankings.items(), key=lambda x: x[1]['score'], reverse=True))
    
    # Onglets résultats
    res_tab1, res_tab2, res_tab3, res_tab4 = st.tabs([
        "🏆 Classement",
        "🎯 Trios",
        "🎰 Quintés",
        "📈 Graphiques"
    ])
    
    with res_tab1:
        st.subheader("Classement Final")
        ranking_list = []
        for pos, (idx, horse) in enumerate(sorted_rankings.items(), 1):
            ranking_list.append({
                '🏅': pos,
                'Nom': horse['nom'],
                'Score': f"{horse['score']:.1f}/100",
                'Driver': f"{horse['driver']}%",
                'Trainer': f"{horse['trainer']}%"
            })
        st.dataframe(pd.DataFrame(ranking_list), use_container_width=True, hide_index=True)
        
        # Top 3
        st.subheader("🎯 Top 3 Favoris")
        top3_indices = list(sorted_rankings.keys())[:3]
        for i, idx in enumerate(top3_indices, 1):
            horse = sorted_rankings[idx]
            st.write(f"{i}. **{horse['nom']}** - Score: {horse['score']:.1f}/100")
    
    with res_tab2:
        st.subheader("🎯 Top 10 Combinaisons Trio")
        
        trio_combos = []
        indices = list(sorted_rankings.keys())[:10]
        
        if len(indices) >= 3:
            for combo in list(combinations(indices, 3))[:10]:
                scores = [sorted_rankings[idx]['score'] for idx in combo]
                avg_score = sum(scores) / len(scores)
                names = " - ".join([sorted_rankings[idx]['nom'] for idx in combo])
                trio_combos.append({
                    'Combinaison': names,
                    'Score': f"{avg_score:.1f}"
                })
        
        if trio_combos:
            st.dataframe(pd.DataFrame(trio_combos), use_container_width=True, hide_index=True)
    
    with res_tab3:
        st.subheader("🎰 Top 10 Combinaisons Quinté+")
        
        quinte_combos = []
        indices = list(sorted_rankings.keys())[:12]
        
        if len(indices) >= 5:
            for combo in list(combinations(indices, 5))[:10]:
                scores = [sorted_rankings[idx]['score'] for idx in combo]
                avg_score = sum(scores) / len(scores)
                names = " - ".join([sorted_rankings[idx]['nom'] for idx in combo])
                quinte_combos.append({
                    'Combinaison': names,
                    'Score': f"{avg_score:.1f}"
                })
        
        if quinte_combos:
            st.dataframe(pd.DataFrame(quinte_combos), use_container_width=True, hide_index=True)
    
    with res_tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            scores_data = {h['nom']: h['score'] for h in list(sorted_rankings.values())[:10]}
            fig = go.Figure(data=[
                go.Bar(x=list(scores_data.keys()), y=list(scores_data.values()),
                       marker=dict(color=list(scores_data.values()), colorscale='Viridis'))
            ])
            fig.update_layout(title="Scores (Top 10)", height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            all_scores = [h['score'] for h in sorted_rankings.values()]
            fig = go.Figure(data=[go.Histogram(x=all_scores, nbinsx=10)])
            fig.update_layout(title="Distribution des Scores", height=400)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
