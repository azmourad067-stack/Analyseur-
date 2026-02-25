import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import re
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from itertools import combinations
import json

# Configuration Streamlit
st.set_page_config(
    page_title="🐴 PronoHippo - Pronostics Hippiques",
    page_icon="🐴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== INITIALISATION OCR FLEXIBLE =====================

def load_ocr_engine():
    """Charge le meilleur moteur OCR disponible"""
    ocr_method = None
    
    # Essayer EasyOCR en premier
    try:
        import easyocr
        st.session_state.ocr_reader = easyocr.Reader(['fr', 'en'], gpu=False)
        ocr_method = "EasyOCR"
    except:
        # Fallback sur Tesseract
        try:
            import pytesseract
            ocr_method = "Tesseract"
        except:
            ocr_method = "Manual"
    
    return ocr_method

# ===================== FONCTIONS OCR ET EXTRACTION =====================

@st.cache_resource
def get_ocr_reader():
    """Charge le lecteur OCR avec gestion d'erreurs"""
    try:
        import easyocr
        return easyocr.Reader(['fr', 'en'], gpu=False)
    except Exception as e:
        st.warning(f"⚠️ EasyOCR non disponible: {str(e)}")
        return None

def extract_text_from_image(image, method="easyocr"):
    """Extrait le texte d'une image avec OCR flexible"""
    try:
        if method == "easyocr":
            try:
                import easyocr
                reader = get_ocr_reader()
                if reader:
                    img_array = np.array(image)
                    results = reader.readtext(img_array, detail=1)
                    text = "\n".join([detection[1] for detection in results])
                    return text, results
            except:
                pass
        
        # Fallback: Tesseract
        if method in ["tesseract", "fallback"]:
            try:
                import pytesseract
                text = pytesseract.image_to_string(image, lang='fra+eng')
                return text, []
            except:
                pass
        
        # Dernier recours: extraction manuelle
        st.warning("⚠️ OCR indisponible. Veuillez entrer les données manuellement.")
        return "", []
    
    except Exception as e:
        st.error(f"❌ Erreur OCR: {str(e)}")
        return "", []

def clean_ocr_text(text):
    """Nettoie le texte extrait par OCR"""
    text = text.replace('|', 'l')
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    return text

# ===================== PARSING ET STRUCTURATION =====================

def create_horses_dataframe_from_text(text):
    """Crée un DataFrame à partir du texte OCR"""
    lines = text.split('\n')
    horses = []
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 3:
            continue
        
        # Parser une ligne de données
        parts = line.split()
        
        if parts and parts[0].isdigit():
            try:
                numero = int(parts[0])
                horse_entry = {'numero': numero}
                
                # Chercher distance
                for part in parts:
                    if part.isdigit() and 1500 <= int(part) <= 3000:
                        horse_entry['distance'] = int(part)
                        break
                
                # Chercher pourcentages
                pct_matches = re.findall(r'(\d+)%', line)
                if pct_matches:
                    for i, pct in enumerate(pct_matches[:2]):
                        if i == 0:
                            horse_entry['reussite_driver'] = int(pct)
                        elif i == 1:
                            horse_entry['reussite_entraineur'] = int(pct)
                
                # Chercher le nom
                for part in parts[1:]:
                    if not part.isdigit() and '%' not in part and len(part) > 2:
                        horse_entry['nom'] = part
                        break
                
                # Ajouter si on a au moins le nom
                if 'nom' in horse_entry:
                    horse_entry.setdefault('distance', 2100)
                    horse_entry.setdefault('reussite_driver', 10)
                    horse_entry.setdefault('reussite_entraineur', 10)
                    horse_entry.setdefault('victoires', 0)
                    horse_entry.setdefault('musique', '')
                    horses.append(horse_entry)
            
            except (ValueError, IndexError):
                continue
    
    if horses:
        return pd.DataFrame(horses)
    else:
        # Données par défaut pour démo
        return pd.DataFrame([
            {'numero': 1, 'nom': 'Cheval 1', 'distance': 2100, 'reussite_driver': 15, 
             'reussite_entraineur': 12, 'victoires': 25, 'musique': '3a0aDa8a5a'},
            {'numero': 2, 'nom': 'Cheval 2', 'distance': 2100, 'reussite_driver': 12,
             'reussite_entraineur': 14, 'victoires': 30, 'musique': '0a6m4a0a1a'},
        ])

# ===================== ALGORITHME DE SCORING =====================

class ScoringEngine:
    """Moteur de scoring intelligent"""
    
    def __init__(self, weights=None):
        self.weights = weights or {
            'record': 0.20,
            'reussite_driver': 0.15,
            'reussite_entraineur': 0.15,
            'victoires': 0.15,
            'musique': 0.20,
            'regularite': 0.10,
            'ecart': 0.05
        }
    
    def parse_musique_score(self, musique_str):
        """Convertit la musique en score"""
        if not musique_str or not isinstance(musique_str, str):
            return 50
        
        score = 0
        count = 0
        
        for char in musique_str[-10:]:
            if char.isdigit():
                score += 100 if char != '0' else 50
            elif char.isalpha():
                score += 50
            count += 1
        
        return score / count if count > 0 else 50
    
    def calculate_score(self, horse_data):
        """Calcule le score global d'un cheval"""
        score = 0
        
        # Score base
        reussite_driver = float(horse_data.get('reussite_driver', 10))
        score += reussite_driver * self.weights['reussite_driver']
        
        reussite_entraineur = float(horse_data.get('reussite_entraineur', 10))
        score += reussite_entraineur * self.weights['reussite_entraineur']
        
        victoires = float(horse_data.get('victoires', 0))
        victoires_score = min(100, (victoires / 50) * 100) if victoires > 0 else 20
        score += victoires_score * self.weights['victoires']
        
        musique = horse_data.get('musique', '')
        musique_score = self.parse_musique_score(musique)
        score += musique_score * self.weights['musique']
        
        # Scores fixes
        score += 60 * self.weights['record']
        score += 70 * self.weights['regularite']
        score += 50 * self.weights['ecart']
        
        return min(100, max(0, score))

# ===================== GÉNÉRATION DE PRONOSTICS =====================

class PronosticsGenerator:
    """Génère les pronostics et combinaisons"""
    
    @staticmethod
    def generate_trio_combinations(rankings, num_combos=10):
        """Génère les meilleures combinaisons Trio"""
        combos = []
        sorted_horses = sorted(rankings.items(), key=lambda x: x[1]['score'], reverse=True)[:10]
        indices = [h[0] for h in sorted_horses]
        
        if len(indices) >= 3:
            trio_combos = list(combinations(indices, 3))[:num_combos]
            for combo in trio_combos:
                combo_score = sum(rankings[idx]['score'] for idx in combo) / 3
                combos.append({'chevaux': combo, 'score_moyen': combo_score})
        
        return combos
    
    @staticmethod
    def generate_quinte_combinations(rankings, num_combos=10):
        """Génère les meilleures combinaisons Quinté+"""
        combos = []
        sorted_horses = sorted(rankings.items(), key=lambda x: x[1]['score'], reverse=True)[:12]
        indices = [h[0] for h in sorted_horses]
        
        if len(indices) >= 5:
            quinte_combos = list(combinations(indices, 5))[:num_combos]
            for combo in quinte_combos:
                combo_score = sum(rankings[idx]['score'] for idx in combo) / 5
                combos.append({'chevaux': combo, 'score_moyen': combo_score})
        
        return combos

# ===================== INTERFACE STREAMLIT =====================

def main():
    # En-tête
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🐴 PronoHippo")
        st.markdown("### Système Intelligent de Pronostics Hippiques v1.1")
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("📊 Poids du Scoring")
        
        col1_w, col2_w = st.columns(2)
        
        with col1_w:
            poids_driver = st.slider("Driver %", 0.0, 1.0, 0.15, 0.05)
            poids_entraineur = st.slider("Entraîneur %", 0.0, 1.0, 0.15, 0.05)
            poids_victoires = st.slider("Victoires", 0.0, 1.0, 0.15, 0.05)
        
        with col2_w:
            poids_musique = st.slider("Musique", 0.0, 1.0, 0.20, 0.05)
            poids_record = st.slider("Record", 0.0, 1.0, 0.20, 0.05)
            poids_autres = st.slider("Autres", 0.0, 1.0, 0.15, 0.05)
        
        # Normaliser les poids
        total = poids_driver + poids_entraineur + poids_victoires + poids_musique + poids_record + poids_autres
        weights = {
            'record': poids_record / total,
            'reussite_driver': poids_driver / total,
            'reussite_entraineur': poids_entraineur / total,
            'victoires': poids_victoires / total,
            'musique': poids_musique / total,
            'regularite': 0,
            'ecart': 0
        }
        
        st.markdown("---")
        st.info("💡 Poids ajustables en temps réel")
    
    # Section Upload
    st.header("📤 Télécharger les Statistiques")
    
    # Option: Upload ou Entrée Manuelle
    input_method = st.radio(
        "Choisir la méthode d'entrée:",
        ["📸 Uploader des images", "📝 Entrer manuellement"]
    )
    
    horses_df = None
    
    if input_method == "📸 Uploader des images":
        uploaded_files = st.file_uploader(
            "Téléchargez les photos (PNG, JPG, JPEG)",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Téléchargez une ou plusieurs captures d'écran"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} image(s) téléchargée(s)")
            
            # Aperçu
            with st.expander("👀 Aperçu des images"):
                cols = st.columns(min(3, len(uploaded_files)))
                for idx, file in enumerate(uploaded_files):
                    with cols[idx % 3]:
                        image = Image.open(file)
                        st.image(image, use_column_width=True, caption=file.name)
            
            # Analyser
            if st.button("🔍 Analyser les Images", use_container_width=True, type="primary"):
                progress_bar = st.progress(0)
                status = st.status("Traitement en cours...", expanded=True)
                
                with status:
                    # Extraction OCR
                    st.write("📖 Extraction du texte...")
                    all_text = ""
                    for file in uploaded_files:
                        image = Image.open(file)
                        text, _ = extract_text_from_image(image)
                        all_text += text + "\n"
                    
                    progress_bar.progress(40)
                    
                    # Nettoyage
                    st.write("🧹 Nettoyage des données...")
                    clean_text = clean_ocr_text(all_text)
                    
                    progress_bar.progress(60)
                    
                    # Structuration
                    st.write("📊 Structuration...")
                    horses_df = create_horses_dataframe_from_text(clean_text)
                    
                    progress_bar.progress(80)
                    
                    # Scoring
                    st.write("🧠 Calcul des scores...")
                    scoring_engine = ScoringEngine(weights)
                    rankings = {}
                    for idx, row in horses_df.iterrows():
                        score = scoring_engine.calculate_score(row.to_dict())
                        rankings[idx] = {
                            'numero': row.get('numero', idx + 1),
                            'nom': row.get('nom', f'Cheval {idx + 1}'),
                            'score': score,
                            'reussite_driver': row.get('reussite_driver', 0),
                        }
                    
                    progress_bar.progress(90)
                    
                    st.write("🎯 Génération des pronostics...")
                    progress_bar.progress(100)
                
                # Afficher résultats
                display_results(horses_df, rankings)
        else:
            st.info("👆 Téléchargez des images pour commencer")
    
    else:  # Entrée Manuelle
        st.subheader("Entrez les données manuellement:")
        
        num_horses = st.number_input("Nombre de chevaux:", 1, 20, 5)
        
        horse_data = []
        cols = st.columns(5)
        
        with st.form("manual_entry"):
            for i in range(num_horses):
                with st.container():
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        nom = st.text_input(f"Nom {i+1}", key=f"nom_{i}")
                    with col2:
                        driver_pct = st.number_input(f"% Driver {i+1}", 0, 100, 15, key=f"driver_{i}")
                    with col3:
                        trainer_pct = st.number_input(f"% Trainer {i+1}", 0, 100, 12, key=f"trainer_{i}")
                    with col4:
                        victoires = st.number_input(f"Victoires {i+1}", 0, 500, 25, key=f"vict_{i}")
                    
                    if nom:
                        horse_data.append({
                            'numero': i + 1,
                            'nom': nom,
                            'distance': 2100,
                            'reussite_driver': driver_pct,
                            'reussite_entraineur': trainer_pct,
                            'victoires': victoires,
                            'musique': ''
                        })
            
            submitted = st.form_submit_button("📊 Analyser", use_container_width=True, type="primary")
        
        if submitted and horse_data:
            horses_df = pd.DataFrame(horse_data)
            
            # Scoring
            scoring_engine = ScoringEngine(weights)
            rankings = {}
            for idx, row in horses_df.iterrows():
                score = scoring_engine.calculate_score(row.to_dict())
                rankings[idx] = {
                    'numero': row.get('numero', idx + 1),
                    'nom': row.get('nom', f'Cheval {idx + 1}'),
                    'score': score,
                    'reussite_driver': row.get('reussite_driver', 0),
                }
            
            display_results(horses_df, rankings)

def display_results(horses_df, rankings):
    """Affiche les résultats"""
    st.markdown("---")
    st.header("📊 Résultats")
    
    # Onglets
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Données",
        "🏆 Classement",
        "🎯 Trios",
        "🎰 Quintés",
        "📈 Graphiques"
    ])
    
    with tab1:
        st.dataframe(horses_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("🏆 Classement Final")
        ranking_data = []
        for pos, (idx, horse) in enumerate(sorted(rankings.items(), key=lambda x: x[1]['score'], reverse=True), 1):
            ranking_data.append({
                '🏅': pos,
                'Nom': horse['nom'],
                'Score': f"{horse['score']:.1f}",
                '% Driver': f"{horse['reussite_driver']}%"
            })
        st.dataframe(pd.DataFrame(ranking_data), use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("🎯 Top 10 Trios")
        trio_combos = PronosticsGenerator.generate_trio_combinations(rankings)
        trio_data = []
        for pos, combo in enumerate(trio_combos, 1):
            chevaux = " - ".join([rankings[idx]['nom'] for idx in combo['chevaux']])
            trio_data.append({
                '🏅': pos,
                'Combinaison': chevaux,
                'Score': f"{combo['score_moyen']:.1f}"
            })
        st.dataframe(pd.DataFrame(trio_data), use_container_width=True, hide_index=True)
    
    with tab4:
        st.subheader("🎰 Top 10 Quintés+")
        quinte_combos = PronosticsGenerator.generate_quinte_combinations(rankings)
        quinte_data = []
        for pos, combo in enumerate(quinte_combos, 1):
            chevaux = " - ".join([rankings[idx]['nom'] for idx in combo['chevaux']])
            quinte_data.append({
                '🏅': pos,
                'Combinaison': chevaux,
                'Score': f"{combo['score_moyen']:.1f}"
            })
        st.dataframe(pd.DataFrame(quinte_data), use_container_width=True, hide_index=True)
    
    with tab5:
        st.subheader("📈 Visualisations")
        col1, col2 = st.columns(2)
        
        with col1:
            scores_data = {
                horse['nom']: horse['score']
                for _, horse in list(sorted(rankings.items(), key=lambda x: x[1]['score'], reverse=True))[:10]
            }
            
            fig = go.Figure(data=[
                go.Bar(x=list(scores_data.keys()), y=list(scores_data.values()),
                       marker=dict(color=list(scores_data.values()), colorscale='Viridis'))
            ])
            fig.update_layout(title="Scores (Top 10)", height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            all_scores = [h['score'] for h in rankings.values()]
            fig = go.Figure(data=[go.Histogram(x=all_scores, nbinsx=10)])
            fig.update_layout(title="Distribution des Scores", height=400)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
