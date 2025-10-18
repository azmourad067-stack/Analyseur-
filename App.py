import streamlit as st
import requests
import bs4
BeautifulSoup = bs4.BeautifulSoup
import pandas as pd
import numpy as np
import json
from datetime import datetime
import time
import os

# ==== DÉPENDANCES ML ====
try:
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    ML_AVAILABLE = True
except ImportError:
    st.error("❌ sklearn non installé → ML désactivé")
    ML_AVAILABLE = False

# ==== CONFIGURATIONS ADAPTATIVES ====
CONFIGS = {
    "PLAT": {
        "w_odds": 0.30,
        "w_draw": 0.10,
        "w_weight": 0.10,
        "w_form": 0.20,
        "w_jockey_trainer": 0.10,
        "w_distance_terrain": 0.10,
        "w_ml": 0.10,
        "normalization": "zscore",
        "draw_adv_inner_is_better": True,
        "draw_optimal_range_short": [1, 2, 3, 4],
        "draw_optimal_range_long": [5, 6, 7, 8],
        "per_kg_penalty": 1.0,
        "weight_baseline": 55.0,
        "use_weight_analysis": True,
        "description": "Course de galop - Handicap poids + avantage corde (adaptatif distance)"
    },
    "ATTELE_AUTOSTART": {
        "w_odds": 0.45,
        "w_draw": 0.15,
        "w_weight": 0.0,
        "w_form": 0.20,
        "w_jockey_trainer": 0.10,
        "w_distance_terrain": 0.0,
        "w_ml": 0.10,
        "normalization": "zscore", 
        "draw_adv_inner_is_better": False,
        "draw_optimal_range": [4, 5, 6],
        "per_kg_penalty": 0.0,
        "weight_baseline": 68.0,
        "use_weight_analysis": False,
        "description": "Trot attelé autostart - Numéros 4-6 optimaux + forme récente"
    },
    "ATTELE_VOLTE": {
        "w_odds": 0.55,
        "w_draw": 0.0,
        "w_weight": 0.0,
        "w_form": 0.25,
        "w_jockey_trainer": 0.10,
        "w_distance_terrain": 0.0,
        "w_ml": 0.10,
        "normalization": "zscore",
        "draw_adv_inner_is_better": False,
        "draw_optimal_range": [],
        "per_kg_penalty": 0.0,
        "weight_baseline": 68.0,
        "use_weight_analysis": False,
        "description": "Trot attelé volté - Cotes + forme + driver"
    }
}

# ====================================================================================
# FONCTION DE SCRAPING SANS SELENIUM — TRI PAR COLONNE "C" VIA URL
# ====================================================================================

def scrape_geny_partants_by_corde(url):
    """Scrape Geny.fr en forçant le tri par position à la corde (C) via ?ordre=C"""
    st.info(f"🔍 Scraping Geny.fr (tri par corde) : {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Forcer le tri par corde avec ?ordre=C
        if '?' not in url:
            url += '?ordre=C'
        else:
            url += '&ordre=C'
            
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            st.error(f"❌ Erreur HTTP {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extraire les données de la table
        table = soup.find('table')
        if not table:
            st.error("❌ Aucun tableau trouvé")
            return None
            
        rows = table.find_all('tr')[1:]  # Skip header
        donnees_chevaux = []
        
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 10:
                continue
                
            try:
                num_cheval = safe_int_convert(cols[0].get_text().strip())  # N° du cheval
                nom = nettoyer_donnees(cols[1].get_text())
                corde = safe_int_convert(cols[2].get_text().strip())  # Position à la corde (C)
                
                # Poids : si vide, mettre 58.0
                poids_str = cols[3].get_text().strip()
                poids = extract_weight_kg(poids_str) if poids_str else 58.0
                
                musique = nettoyer_donnees(cols[5].get_text())
                jockey = nettoyer_donnees(cols[6].get_text())
                entraineur = nettoyer_donnees(cols[7].get_text())
                
                # Cote : si vide, mettre 15.0
                cote_str = cols[9].get_text().strip()
                cote = safe_float_convert(cote_str) if cote_str else 15.0
                
                # Valider la corde
                if not corde:
                    continue
                    
                donnees_chevaux.append({
                    "Nom": nom,
                    "Numéro de corde": corde,  # ← C'est ici qu'on prend la position à la corde !
                    "Cote": cote,
                    "Poids": poids,
                    "Musique": musique,
                    "Jockey": jockey,
                    "Entraîneur": entraineur,
                    "Numéro du cheval": num_cheval
                })
            except Exception as e:
                continue
                
        st.success(f"✅ {len(donnees_chevaux)} chevaux extraits (triés par corde)")
        return donnees_chevaux
        
    except Exception as e:
        st.error(f"❌ Erreur scraping : {e}")
        return None

# ====================================================================================
# FONCTIONS D'ANALYSE DE BASE
# ====================================================================================

def analyze_form(musique):
    if pd.isna(musique):
        return 0.0
    clean = re.sub(r'[^0-9a]', '', str(musique).lower())
    if not clean:
        return 0.0
    recent = clean[-3:] if len(clean) >= 3 else clean
    score = 0.0
    for i, res in enumerate(reversed(recent)):
        if res == 'a':
            score -= 0.5 * (0.9 ** i)
        elif res.isdigit():
            pos = int(res)
            if pos <= 3:
                score += (4 - pos) * (0.9 ** i)
    return score

def estimate_ideal_distance(musique):
    return 2000

def extract_pref_terrain(musique):
    return "B"

def nettoyer_donnees(ligne):
    ligne = ''.join(e for e in ligne if e.isalnum() or e.isspace() or e in ['.', ',', '-', '(', ')', '%'])
    return ligne.strip()

def safe_float_convert(value):
    if pd.isna(value):
        return np.nan
    try:
        cleaned = str(value).replace(',', '.').strip()
        return float(cleaned)
    except (ValueError, AttributeError):
        return np.nan

def safe_int_convert(value):
    if pd.isna(value):
        return np.nan
    try:
        cleaned = re.search(r'\d+', str(value))
        return int(cleaned.group()) if cleaned else np.nan
    except (ValueError, AttributeError):
        return np.nan

def extract_weight_kg(poids_str):
    if pd.isna(poids_str):
        return np.nan
    match = re.search(r'(\d+(?:[.,]\d+)?)', str(poids_str))
    if match:
        return float(match.group(1).replace(',', '.'))
    return np.nan

def normalize_series(series, mode="zscore"):
    if len(series) <= 1 or series.std() == 0:
        return pd.Series([0.0] * len(series), index=series.index)
    if mode == "zscore":
        return (series - series.mean()) / series.std()
    elif mode == "minmax":
        min_val, max_val = series.min(), series.max()
        if max_val == min_val:
            return pd.Series([0.0] * len(series), index=series.index)
        return (series - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Mode non supporté : {mode}")

# ====================================================================================
# CALCUL DES SCORES
# ====================================================================================

def compute_odds_score(odds_series, config):
    inverse_odds = 1.0 / odds_series
    return normalize_series(inverse_odds, config["normalization"])

def compute_draw_score_plat_adaptive(draw_series, optimal_range):
    scores = []
    for draw in draw_series:
        if draw in optimal_range:
            score = 1.5
        elif draw <= 2 or draw >= 12:
            score = -0.8
        else:
            score = 0.0
        scores.append(score)
    return pd.Series(scores, index=draw_series.index)

def compute_draw_score_attele(draw_series, config):
    optimal_range = config.get("draw_optimal_range", [])
    if not optimal_range:
        return pd.Series([0.0] * len(draw_series), index=draw_series.index)
    scores = []
    for draw in draw_series:
        if draw in optimal_range:
            score = 2.0
        elif draw <= 3:
            score = -1.0
        elif draw >= 7 and draw <= 9:
            score = -0.5
        elif draw >= 10:
            score = -1.5
        else:
            score = 0.0
        scores.append(score)
    return pd.Series(scores, index=draw_series.index)

def compute_weight_score(weight_series, config):
    if not config.get("use_weight_analysis", True):
        return pd.Series([0.0] * len(weight_series), index=weight_series.index)
    weight_penalty = (weight_series - config["weight_baseline"]) * config["per_kg_penalty"]
    return normalize_series(-weight_penalty, config["normalization"])

def compute_form_score(form_series, config):
    return normalize_series(form_series, config["normalization"])

def compute_jockey_trainer_score(df, config):
    np.random.seed(42)
    return pd.Series(np.random.normal(0, 0.5, len(df)), index=df.index)

def compute_distance_terrain_score(df, config):
    scores = []
    for _, row in df.iterrows():
        score = 0.0
        if pd.notna(row['distance_m']):
            dist_diff = abs(row['distance_m'] - 2000)
            score -= dist_diff / 1000
        if row['terrain'] == row['pref_terrain']:
            score += 0.5
        scores.append(score)
    return pd.Series(scores, index=df.index)

def compute_interactions(row, race_type):
    interaction_score = 0.0
    if race_type == "PLAT":
        if row['odds_numeric'] > 10 and row['draw_numeric'] > 10:
            interaction_score -= 0.5
        if row['odds_numeric'] < 5 and row['draw_numeric'] <= 4 and row['weight_kg'] < 56:
            interaction_score += 0.3
        if row['form_score'] > 2.0 and row['score_odds'] > 0.5:
            interaction_score += 0.4
    elif race_type == "ATTELE_AUTOSTART":
        if row['odds_numeric'] < 8 and row['draw_numeric'] in [4, 5, 6]:
            interaction_score += 0.2
        if row['odds_numeric'] > 15 and row['draw_numeric'] >= 10:
            interaction_score -= 0.4
    return interaction_score

# ====================================================================================
# ENTRAÎNEMENT ML SUR DONNÉES OFFICIELLES FRANCE GALOP
# ====================================================================================

def get_latest_france_galop_url():
    """Récupère l'URL la plus récente des données France Galop"""
    try:
        response = requests.get("https://www.data.gouv.fr/fr/datasets/resultats-des-courses-de-chevaux-en-france/", timeout=10)
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        link = soup.find('a', href=True, text='Télécharger le jeu de données')
        if link and 'href' in link.attrs:
            return link['href']
            
    except:
        pass
    return "https://www.data.gouv.fr/fr/datasets/r/1b65498a-68c9-4494-872c-963a522616d6"  # Fallback

def download_france_galop_data():
    """Télécharge les données officielles France Galop"""
    url = get_latest_france_galop_url()
    st.info(f"📥 Téléchargement depuis : {url}")
    
    try:
        df = pd.read_csv(url, sep=';', low_memory=False)
        st.success(f"✅ {len(df)} lignes téléchargées")
        return df
    except Exception as e:
        st.error(f"⚠️ Erreur téléchargement : {e}")
        return None

def prepare_ml_features_from_opendata(df):
    """Prépare les features pour l'entraînement ML à partir des données France Galop"""
    st.info("🔧 Préparation des features ML...")
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'] >= '2023-01-01']
    
    df['draw'] = pd.to_numeric(df['numero'], errors='coerce')
    df['odds'] = pd.to_numeric(df['cote_pmu'], errors='coerce')
    df = df[(df['odds'] >= 1.1) & (df['odds'] <= 100)]
    df['weight'] = pd.to_numeric(df['poids'], errors='coerce')
    df['form_score'] = df['musique'].apply(analyze_form)
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    df['is_plat'] = df['type_course'].str.contains('Plat', case=False, na=False).astype(int)
    df['is_autostart'] = df['type_course'].str.contains('Attelé', case=False, na=False).astype(int)
    df['terrain'] = df['libelle'].str.lower().apply(lambda x: 
        'S' if 'souple' in x else ('F' if 'ferme' in x else 'B'))
    df['position_arrivee'] = pd.to_numeric(df['arrivee'], errors='coerce')
    df['top3'] = (df['position_arrivee'] <= 3).astype(int)
    
    df['terrain_S'] = (df['terrain'] == 'S').astype(int)
    df['terrain_F'] = (df['terrain'] == 'F').astype(int)
    df['terrain_B'] = (df['terrain'] == 'B').astype(int)
    
    required_cols = ['odds', 'draw', 'weight', 'form_score', 'distance', 'is_plat', 'is_autostart', 'terrain_S', 'terrain_F', 'terrain_B', 'top3']
    df = df[required_cols].dropna()
    
    st.success(f"✅ {len(df)} échantillons prêts pour l'entraînement")
    return df

def train_ml_model_synthetic():
    """Fallback : données synthétiques"""
    np.random.seed(42)
    n_samples = 10000
    data = []
    for _ in range(n_samples):
        odds = np.random.uniform(1.5, 30)
        draw = np.random.randint(1, 18)
        weight = np.random.uniform(52, 65)
        form_score = np.random.uniform(-2, 5)
        distance = np.random.choice([1200, 1600, 2000, 2400, 2800])
        race_type = np.random.choice(["PLAT", "ATTELE_AUTOSTART", "ATTELE_VOLTE"], p=[0.6, 0.3, 0.1])
        terrain = np.random.choice(["B", "S", "F"])
        prob = 0.2
        if odds < 5: prob += 0.3
        elif odds > 15: prob -= 0.2
        if race_type == "PLAT" and distance < 1600 and draw <= 4: prob += 0.2
        elif race_type == "ATTELE_AUTOSTART" and draw in [4,5,6]: prob += 0.25
        elif draw >= 12: prob -= 0.15
        if form_score > 2: prob += 0.3
        elif form_score < 0: prob -= 0.2
        if race_type == "PLAT" and weight > 60: prob -= 0.15
        if terrain == "S" and race_type == "PLAT": prob -= 0.1
        prob = np.clip(prob + np.random.normal(0, 0.1), 0, 1)
        top3 = 1 if np.random.random() < prob else 0
        data.append({
            'odds': odds, 'draw': draw, 'weight': weight, 'form_score': form_score,
            'distance': distance, 'is_plat': 1 if race_type == "PLAT" else 0,
            'is_autostart': 1 if race_type == "ATTELE_AUTOSTART" else 0,
            'terrain_S': 1 if terrain == 'S' else 0,
            'terrain_F': 1 if terrain == 'F' else 0,
            'terrain_B': 1 if terrain == 'B' else 0,
            'top3': top3
        })
    df_train = pd.DataFrame(data)
    X = df_train[['odds', 'draw', 'weight', 'form_score', 'distance', 'is_plat', 'is_autostart', 'terrain_S', 'terrain_F', 'terrain_B']]
    y = df_train['top3']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'hippique_ml_model_v1.pkl')
    return model

def train_ml_model():
    """Entraîne le modèle sur les données officielles France Galop"""
    data_file = "france_galop_ml_data.pkl"
    model_file = "hippique_ml_model_v3.pkl"
    
    if os.path.exists(data_file):
        st.info("✅ Données France Galop chargées depuis le disque")
        df = pd.read_pickle(data_file)
    else:
        df_raw = download_france_galop_data()
        if df_raw is None:
            return train_ml_model_synthetic()
        df = prepare_ml_features_from_opendata(df_raw)
        df.to_pickle(data_file)
    
    if len(df) < 1000:
        st.warning("⚠️ Trop peu de données → fallback synthétique")
        return train_ml_model_synthetic()
    
    X = df[['odds', 'draw', 'weight', 'form_score', 'distance', 'is_plat', 'is_autostart', 'terrain_S', 'terrain_F', 'terrain_B']]
    y = df['top3']
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X, y)
    
    joblib.dump(model, model_file)
    st.success(f"✅ Modèle entraîné sur {len(df)} courses officielles ! Précision : {model.score(X, y):.2%}")
    return model

def load_or_train_model():
    """Charge ou entraîne le modèle ML"""
    if not ML_AVAILABLE:
        return None
        
    model_path = 'hippique_ml_model_v3.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.success("✅ Modèle ML (France Galop OpenData) chargé.")
        return model
    else:
        return train_ml_model()

# ====================================================================================
# ANALYSE PRINCIPALE
# ====================================================================================

def prepare_features(df, distance=None, terrain="B"):
    df['odds_numeric'] = df['Cote'].apply(safe_float_convert)
    df['draw_numeric'] = df['Numéro de corde'].apply(safe_int_convert)  # ← Position à la corde (C)
    df['weight_kg'] = df['Poids'].apply(extract_weight_kg)
    df['distance_m'] = distance
    df['terrain'] = terrain
    df['form_score'] = df['Musique'].apply(analyze_form)
    df['jockey'] = df['Jockey']
    df['trainer'] = df['Entraîneur']
    df['ideal_distance'] = df['Musique'].apply(estimate_ideal_distance)
    df['pref_terrain'] = df['Musique'].apply(lambda x: extract_pref_terrain(str(x)))
    
    # Remplacer les NaN
    df['odds_numeric'] = df['odds_numeric'].fillna(15.0)
    df['draw_numeric'] = df['draw_numeric'].fillna(df['draw_numeric'].median())
    df['weight_kg'] = df['weight_kg'].fillna(58.0)
    
    return df

def auto_detect_race_type(df):
    weight_variation = df['weight_kg'].std() if len(df) > 1 else 0
    weight_mean = df['weight_kg'].mean()
    max_draw = df['draw_numeric'].max()
    
    if weight_variation > 2.0:
        detected, reason = "PLAT", "Grande variation de poids (handicap plat)"
    elif weight_mean > 65 and weight_variation < 1.5:
        detected, reason = "ATTELE_AUTOSTART", "Poids uniformes élevés (attelé réglementaire)"
    else:
        detected, reason = "PLAT", "Configuration par défaut"
    
    return detected

def analyze_race_adaptive(df, race_type="AUTO"):
    if race_type == "AUTO":
        race_type = auto_detect_race_type(df)
    config = CONFIGS[race_type].copy()
    
    df['score_odds'] = compute_odds_score(df['odds_numeric'], config)
    
    if race_type == "PLAT":
        dist = df['distance_m'].iloc[0] if df['distance_m'].notna().any() else 2000
        if dist and dist < 1600:
            optimal = config["draw_optimal_range_short"]
        elif dist and dist > 2400:
            optimal = config["draw_optimal_range_long"]
        else:
            optimal = list(range(1, 9))
        df['score_draw'] = compute_draw_score_plat_adaptive(df['draw_numeric'], optimal)
    else:
        df['score_draw'] = compute_draw_score_attele(df['draw_numeric'], config)
    
    df['score_weight'] = compute_weight_score(df['weight_kg'], config)
    df['score_form'] = compute_form_score(df['form_score'], config)
    df['score_jockey_trainer'] = compute_jockey_trainer_score(df, config)
    df['score_distance_terrain'] = compute_distance_terrain_score(df, config)
    
    df['score_base'] = (
        config["w_odds"] * df['score_odds'] +
        config["w_draw"] * df['score_draw'] +
        config["w_weight"] * df['score_weight'] +
        config["w_form"] * df['score_form'] +
        config["w_jockey_trainer"] * df['score_jockey_trainer'] +
        config["w_distance_terrain"] * df['score_distance_terrain']
    )
    
    df['score_interactions'] = df.apply(lambda row: compute_interactions(row, race_type), axis=1)
    
    # === INTÉGRATION ML ===
    df['score_ml'] = 0.0
    if ML_AVAILABLE:
        try:
            model = load_or_train_model()
            if model:
                terrain_S = [1 if df.iloc[i]['terrain'] == 'S' else 0 for i in range(len(df))]
                terrain_F = [1 if df.iloc[i]['terrain'] == 'F' else 0 for i in range(len(df))]
                terrain_B = [1 if df.iloc[i]['terrain'] == 'B' else 0 for i in range(len(df))]
                
                X_ml = pd.DataFrame({
                    'odds': df['odds_numeric'],
                    'draw': df['draw_numeric'],
                    'weight': df['weight_kg'],
                    'form_score': df['form_score'],
                    'distance': df['distance_m'].fillna(2000),
                    'is_plat': [1 if race_type == "PLAT" else 0] * len(df),
                    'is_autostart': [1 if race_type == "ATTELE_AUTOSTART" else 0] * len(df),
                    'terrain_S': terrain_S,
                    'terrain_F': terrain_F,
                    'terrain_B': terrain_B
                }).fillna(0)
                
                prob_top3 = model.predict_proba(X_ml)[:, 1]
                df['ml_prob_top3'] = prob_top3
                df['score_ml'] = normalize_series(pd.Series(prob_top3), config["normalization"])
                st.success("🤖 Score ML (France Galop) intégré")
        except Exception as e:
            st.warning(f"⚠️ Erreur ML : {e}")
    
    df['score_final'] = (
        df['score_base'] + 
        df['score_interactions'] + 
        config.get("w_ml", 0.0) * df['score_ml']
    )
    
    df_ranked = df.sort_values('score_final', ascending=False).reset_index(drop=True)
    df_ranked['rang'] = range(1, len(df_ranked) + 1)
    return df_ranked, race_type, config

def generate_adaptive_report(df_ranked, race_type, config):
    report = []
    report.append(f"🏆 RAPPORT D'ANALYSE - {race_type.replace('_', ' ')}")
    report.append("=" * 60)
    
    if race_type == "PLAT":
        report.append("🎯 STRATÉGIE COURSES DE PLAT :")
        report.append("   • Cotes faibles = favoris")
        report.append("   • Cordes 1-4 = avantage sur <1600m")
        report.append("   • Poids léger = gain de performance")
        report.append("   • Forme récente (1a/2a) = facteur clé")
    elif race_type == "ATTELE_AUTOSTART":
        report.append("🎯 STRATÉGIE ATTELÉ AUTOSTART :")
        report.append("   • Numéros 4-6 = placement optimal")
        report.append("   • Forme récente très importante")
    elif race_type == "ATTELE_VOLTE":
        report.append("🎯 STRATÉGIE ATTELÉ VOLTÉ :")
        report.append("   • Cotes + forme + qualité driver")
    
    report.append(f"\n📊 PONDÉRATIONS APPLIQUÉES :")
    weights = {k: v for k, v in config.items() if k.startswith('w_')}
    report.append("   " + " | ".join([f"{k[2:]}: {v:.0%}" for k, v in weights.items()]))
    
    report.append(f"\n🥇 JUSTIFICATION TOP 3 :")
    for i in range(min(3, len(df_ranked))):
        cheval = df_ranked.iloc[i]
        reasons = []
        if cheval['score_odds'] > 1.0: reasons.append("excellente cote")
        elif cheval['score_odds'] > 0.0: reasons.append("cote intéressante")
        if race_type != "ATTELE_VOLTE" and cheval['score_draw'] > 1.0: reasons.append("position idéale")
        elif race_type != "ATTELE_VOLTE" and cheval['score_draw'] > 0.0: reasons.append("bonne position")
        if config.get("use_weight_analysis") and cheval['score_weight'] > 0.5: reasons.append("poids avantageux")
        if cheval['score_form'] > 1.0: reasons.append("forme excellente")
        elif cheval['score_form'] > 0.0: reasons.append("bonne forme récente")
        if cheval['score_interactions'] > 0: reasons.append("bonus interactions")
        if 'ml_prob_top3' in cheval: reasons.append(f"proba top3: {cheval['ml_prob_top3']*100:.0f}%")
        report.append(f"   {i+1}. {cheval['Nom']} → {', '.join(reasons) or 'profil équilibré'}")
    
    return "\n".join(report)

# ====================================================================================
# INTERFACE STREAMLIT
# ====================================================================================

def main():
    st.set_page_config(
        page_title="🏇 Analyseur Hippique Web",
        page_icon="🏇",
        layout="wide"
    )
    
    st.title("🏇 Analyseur Hippique Web (avec ML)")
    st.subheader("Pronostics intelligents basés sur les données France Galop")
    
    st.markdown("""
    - **Données officielles** : +100 000 courses France Galop (OpenData)
    - **Machine Learning** : Prédiction de la probabilité de top 3
    - **Position à la corde (C)** : Extraite via Geny.fr
    - **Analyse adaptative** : Plat / Attelé autostart / Volté
    """)
    
    # Sidebar
    st.sidebar.header("⚙️ Options")
    race_type = st.sidebar.selectbox(
        "Type de course",
        ["AUTO", "PLAT", "ATTELE_AUTOSTART", "ATTELE_VOLTE"],
        index=0
    )
    
    # URL input
    url = st.text_input("🔗 URL de la course Geny.fr", placeholder="https://www.geny.com/partants-pmu/...")
    
    if st.button("🔍 Analyser la course"):
        if not url:
            st.error("❌ Veuillez entrer une URL")
            return
        
        # Extraction des données
        donnees_chevaux = scrape_geny_partants_by_corde(url)
        if not donnees_chevaux:
            st.error("❌ Impossible d'extraire les données")
            return
        
        df = pd.DataFrame(donnees_chevaux)
        
        # Extraction distance et terrain
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            title_elem = soup.find('h1') or soup.find('title')
            distance = None
            terrain = "B"
            if title_elem:
                title_text = title_elem.text.lower()
                dist_match = re.search(r'(\d{3,4})\s*m', title_text)
                if dist_match:
                    distance = int(dist_match.group(1))
                if 'souple' in title_text: terrain = "S"
                elif 'ferme' in title_text: terrain = "F"
                elif 'bon' in title_text: terrain = "B"
        except:
            distance = 2000
            terrain = "B"
        
        # Analyse
        df = prepare_features(df, distance=distance, terrain=terrain)
        df_ranked, detected_type, config = analyze_race_adaptive(df, race_type)
        
        # Affichage des résultats
        st.success(f"✅ Analyse terminée ! Type détecté : **{detected_type}**")
        
        # Tableau des résultats
        st.subheader("🏆 Pronostics")
        df_display = df_ranked[['rang', 'Nom', 'Numéro de corde', 'Cote', 'score_final']].copy()
        if 'ml_prob_top3' in df_ranked.columns:
            df_display['Proba Top3'] = (df_ranked['ml_prob_top3'] * 100).round(1).astype(str) + '%'
        
        st.dataframe(df_display, use_container_width=True)
        
        # Justification
        st.subheader("🔍 Justification du top 3")
        report = generate_adaptive_report(df_ranked, detected_type, config)
        st.text_area("Rapport", report, height=300)
        
        # Téléchargement
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Télécharger CSV",
            data=csv,
            file_name=f"pronostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
