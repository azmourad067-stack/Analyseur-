# app.py
# Application Streamlit de prédiction du Loto français
# Basée sur l'analyse statistique des tirages passés

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import io

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Loto",
    page_icon="🎲",
    layout="wide"
)

st.title("🎲 Prédiction du Loto français")
st.markdown("""
Cette application analyse l'historique des tirages du Loto pour générer une combinaison prédite.
L'hypothèse sous-jacente est que les tirages, bien qu'aléatoires en apparence, présentent
des structures statistiques exploitables (fréquences, écarts, tendances, corrélations).
""")

# ------------------------------------------------------------------------------
# 1. Fonctions de chargement et de parsing
# ------------------------------------------------------------------------------

@st.cache_data
def load_data(uploaded_file) -> pd.DataFrame:
    """
    Charge et parse le fichier CSV d'historique des tirages.
    Attend les colonnes : date_de_tirage, boule_1 à boule_5, numero_chance
    Retourne un DataFrame avec les colonnes : date, boules (liste), chance
    """
    try:
        # Lecture du CSV avec séparateur point-virgule
        df_raw = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
        
        # Vérification des colonnes nécessaires
        required_cols = ['date_de_tirage', 'boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'numero_chance']
        missing = [col for col in required_cols if col not in df_raw.columns]
        if missing:
            st.error(f"Colonnes manquantes dans le fichier : {missing}")
            return None
        
        # Conversion de la date
        df_raw['date_de_tirage'] = pd.to_datetime(df_raw['date_de_tirage'], format='%d/%m/%Y', errors='coerce')
        if df_raw['date_de_tirage'].isna().any():
            st.warning("Certaines dates n'ont pas pu être parsées (format attendu : JJ/MM/AAAA).")
            # On supprime les lignes avec date invalide
            df_raw = df_raw.dropna(subset=['date_de_tirage'])
        
        # Construction du DataFrame simplifié
        df = pd.DataFrame()
        df['date'] = df_raw['date_de_tirage']
        df['boules'] = df_raw[['boule_1','boule_2','boule_3','boule_4','boule_5']].values.tolist()
        df['chance'] = df_raw['numero_chance'].astype(int)
        
        # Trier par date croissante
        df = df.sort_values('date').reset_index(drop=True)
        
        # Vérification des plages de valeurs
        for boules in df['boules']:
            if not all(1 <= b <= 49 for b in boules):
                st.warning("Certaines boules sont hors de la plage 1-49. Elles seront ignorées.")
                # On pourrait filtrer, mais on laisse tel quel pour l'instant
        
        if not (1 <= df['chance'].min() <= 10 and df['chance'].max() <= 10):
            st.warning("Certains numéros chance sont hors de la plage 1-10.")
        
        return df
    
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        return None

# ------------------------------------------------------------------------------
# 2. Fonctions d'analyse statistique
# ------------------------------------------------------------------------------

def compute_frequencies(df, max_num=49):
    """
    Calcule la fréquence d'apparition de chaque numéro (1..max_num) dans les boules.
    Retourne un array de longueur max_num+1 (index 0 inutilisé).
    """
    freq = np.zeros(max_num+1, dtype=int)
    for boules in df['boules']:
        for b in boules:
            if 1 <= b <= max_num:
                freq[b] += 1
    return freq

def compute_chance_frequencies(df, max_chance=10):
    """
    Calcule la fréquence d'apparition de chaque numéro chance (1..max_chance).
    """
    freq = np.zeros(max_chance+1, dtype=int)
    for c in df['chance']:
        if 1 <= c <= max_chance:
            freq[c] += 1
    return freq

def compute_delays(df, max_num=49):
    """
    Calcule l'écart (nombre de tirages depuis la dernière apparition) pour chaque numéro.
    Retourne un array de longueur max_num+1.
    """
    last_seen = np.full(max_num+1, -1, dtype=int)  # -1 = jamais vu
    for i, boules in enumerate(df['boules']):
        for b in boules:
            if 1 <= b <= max_num:
                last_seen[b] = i
    n = len(df)
    delays = np.full(max_num+1, n, dtype=int)  # si jamais vu, on met n (max possible)
    for b in range(1, max_num+1):
        if last_seen[b] >= 0:
            delays[b] = n - 1 - last_seen[b]
    return delays

def compute_trend(df, window=50, max_num=49):
    """
    Calcule une tendance récente : fréquence sur les derniers 'window' tirages.
    Retourne un array de longueur max_num+1.
    """
    recent_df = df.tail(window)
    return compute_frequencies(recent_df, max_num)

def compute_cooccurrence(df, max_num=49):
    """
    Calcule une matrice de co-occurrence : nombre de tirages où deux numéros apparaissent ensemble.
    Retourne une matrice (max_num+1 x max_num+1).
    """
    cooc = np.zeros((max_num+1, max_num+1), dtype=int)
    for boules in df['boules']:
        for i in range(5):
            for j in range(i+1, 5):
                a, b = boules[i], boules[j]
                if 1 <= a <= max_num and 1 <= b <= max_num:
                    cooc[a, b] += 1
                    cooc[b, a] += 1
    return cooc

# ------------------------------------------------------------------------------
# 3. Génération de la prédiction
# ------------------------------------------------------------------------------

def generate_prediction(df, freq, delays, trend, cooc,
                        weight_freq=0.4, weight_delay=0.3, weight_trend=0.3,
                        random_factor=0.1):
    """
    Génère une combinaison prédite en combinant les scores de fréquence, écart et tendance.
    """
    max_num = 49
    n = len(df)
    
    # Normalisation des scores
    freq_norm = freq[1:] / freq[1:].max() if freq[1:].max() > 0 else np.zeros(max_num)
    delay_norm = delays[1:] / delays[1:].max() if delays[1:].max() > 0 else np.zeros(max_num)
    trend_norm = trend[1:] / trend[1:].max() if trend[1:].max() > 0 else np.zeros(max_num)
    
    # Score global pour chaque numéro
    scores = (weight_freq * freq_norm +
              weight_delay * delay_norm +
              weight_trend * trend_norm)
    
    # Ajout d'un facteur aléatoire pour éviter de toujours choisir les mêmes
    noise = np.random.uniform(0, random_factor, max_num)
    scores = scores + noise
    
    # Sélection des 5 numéros avec les scores les plus élevés (sans remise)
    # On trie les indices par score décroissant
    indices = np.argsort(scores)[::-1]  # indices 0-based pour les numéros 1..49
    selected_nums = [idx+1 for idx in indices[:5]]
    
    # Pour le numéro chance : on utilise les fréquences des chances
    chance_freq = compute_chance_frequencies(df)
    chance_probs = chance_freq[1:] / chance_freq[1:].sum() if chance_freq[1:].sum() > 0 else np.ones(10)/10
    # On peut aussi intégrer un écart pour le chance (moins pertinent)
    # On tire selon les probabilités
    chance = np.random.choice(np.arange(1, 11), p=chance_probs)
    
    # Trier les boules pour l'affichage
    selected_nums.sort()
    
    return selected_nums, chance, scores

# ------------------------------------------------------------------------------
# 4. Interface Streamlit
# ------------------------------------------------------------------------------

def main():
    st.sidebar.header("📂 Chargement des données")
    uploaded_file = st.sidebar.file_uploader("Choisissez un fichier CSV (historique des tirages)", type="csv")
    
    if uploaded_file is None:
        st.info("Veuillez charger un fichier CSV d'historique pour commencer.")
        st.markdown("""
        **Format attendu :**
        - Colonnes : `date_de_tirage`, `boule_1` à `boule_5`, `numero_chance`
        - Séparateur : point-virgule (;)
        - Dates au format `JJ/MM/AAAA`
        - Boules : entiers de 1 à 49
        - Numéro chance : entier de 1 à 10
        """)
        return
    
    # Chargement
    df = load_data(uploaded_file)
    if df is None or len(df) == 0:
        st.error("Aucune donnée valide n'a pu être chargée.")
        return
    
    st.sidebar.success(f"✅ {len(df)} tirages chargés.")
    
    # Aperçu des données
    with st.expander("Aperçu des données"):
        st.dataframe(df.head(10))
    
    # Paramètres de prédiction
    st.sidebar.header("⚙️ Paramètres de prédiction")
    weight_freq = st.sidebar.slider("Poids de la fréquence", 0.0, 1.0, 0.4, 0.05)
    weight_delay = st.sidebar.slider("Poids de l'écart (retard)", 0.0, 1.0, 0.3, 0.05)
    weight_trend = st.sidebar.slider("Poids de la tendance récente", 0.0, 1.0, 0.3, 0.05)
    random_factor = st.sidebar.slider("Facteur aléatoire", 0.0, 0.5, 0.1, 0.01)
    
    # Calcul des statistiques (avec cache)
    @st.cache_data
    def compute_stats(df):
        freq = compute_frequencies(df)
        delays = compute_delays(df)
        trend = compute_trend(df)
        cooc = compute_cooccurrence(df)
        return freq, delays, trend, cooc
    
    freq, delays, trend, cooc = compute_stats(df)
    
    # Affichage des statistiques
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nombre de tirages", len(df))
        # Fréquence max
        max_freq = freq.max()
        max_num = np.argmax(freq)
        st.metric("Numéro le plus fréquent", f"{max_num} ({max_freq} fois)")
    with col2:
        # Écart max
        max_delay = delays.max()
        max_delay_num = np.argmax(delays)
        st.metric("Numéro le plus en retard", f"{max_delay_num} ({max_delay} tirages)")
    with col3:
        # Tendances
        recent_freq = compute_frequencies(df.tail(50))
        top_trend = np.argmax(recent_freq[1:])+1
        st.metric("Numéro en tendance (derniers 50)", top_trend)
    
    # Graphique des fréquences
    st.subheader("📊 Fréquences des numéros")
    freq_df = pd.DataFrame({
        'Numéro': np.arange(1, 50),
        'Fréquence': freq[1:]
    })
    st.bar_chart(freq_df.set_index('Numéro'))
    
    # Bouton de génération
    if st.button("🎯 Générer une prédiction"):
        with st.spinner("Analyse en cours..."):
            # On génère plusieurs combinaisons et on prend la meilleure (ou on en affiche une)
            # Pour plus de robustesse, on peut faire une petite simulation
            best_score = -np.inf
            best_comb = None
            best_chance = None
            best_scores = None
            for _ in range(50):  # on simule 50 tirages pondérés
                nums, chance, scores = generate_prediction(
                    df, freq, delays, trend, cooc,
                    weight_freq, weight_delay, weight_trend, random_factor
                )
                # Score total = somme des scores des numéros sélectionnés
                total_score = sum(scores[n-1] for n in nums)  # scores est un array 0-based pour numéro-1
                if total_score > best_score:
                    best_score = total_score
                    best_comb = nums
                    best_chance = chance
                    best_scores = scores
            
            # Affichage des résultats
            st.subheader("🔮 Combinaison prédite")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### **{best_comb[0]} - {best_comb[1]} - {best_comb[2]} - {best_comb[3]} - {best_comb[4]}**")
            with col2:
                st.markdown(f"### Numéro chance : **{best_chance}**")
            
            # Explications
            st.subheader("📝 Justification statistique")
            # Récupérer les scores pour chaque numéro sélectionné
            explanations = []
            for num in best_comb:
                idx = num - 1
                f = freq[num]
                d = delays[num]
                t = trend[num]
                score = best_scores[idx]
                expl = f"**{num}** : fréquence={f}, écart={d}, tendance={t}, score global={score:.3f}"
                explanations.append(expl)
            st.markdown("\n".join(explanations))
            
            st.info("""
            **Méthodologie :**
            - La fréquence donne un poids aux numéros qui sortent souvent.
            - L'écart favorise les numéros qui n'ont pas été tirés depuis longtemps (loi des séries).
            - La tendance récente capture les numéros "chauds" sur les derniers tirages.
            - Un facteur aléatoire est ajouté pour éviter de toujours proposer les mêmes combinaisons.
            - La combinaison finale est celle qui maximise le score global parmi plusieurs simulations.
            """)
            
            # Option : afficher les scores de tous les numéros (classement)
            with st.expander("Voir le classement complet des numéros"):
                # On normalise les scores pour un affichage lisible
                score_df = pd.DataFrame({
                    'Numéro': np.arange(1, 50),
                    'Score': best_scores
                }).sort_values('Score', ascending=False)
                st.dataframe(score_df)

if __name__ == "__main__":
    main()
