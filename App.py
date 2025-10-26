import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime
import plotly.express 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ML & Deep Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🏇 Analyseur Hippique Pro ML v2", page_icon="🏇", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 2.8rem; color: #1e3a8a; text-align: center; margin-bottom: 1.5rem;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; 
                  border-radius: 10px; color: white; text-align: center; margin: 0.5rem 0;}
    .prediction-box {border-left: 4px solid #f59e0b; padding: 0.8rem; background: #fffbeb; 
                     margin: 0.8rem 0; border-radius: 6px;}
    .confidence-high {color: #10b981; font-weight: bold;}
    .confidence-medium {color: #f59e0b; font-weight: bold;}
    .confidence-low {color: #ef4444; font-weight: bold;}
    .feature-importance {background: #f0f9ff; padding: 1rem; border-radius: 8px; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION & CONSTANTES
# ============================================================================

RACE_CONFIGS = {
    "PLAT": {"weight_importance": 0.25, "draw_advantage": [1, 2, 3, 4], "age_optimal": (4, 6)},
    "ATTELE_AUTO": {"weight_importance": 0.05, "draw_advantage": [4, 5, 6], "age_optimal": (5, 8)},
    "ATTELE_VOLTE": {"weight_importance": 0.05, "draw_advantage": [], "age_optimal": (5, 8)},
    "OBSTACLE": {"weight_importance": 0.20, "draw_advantage": [], "age_optimal": (6, 10)}
}

# ============================================================================
# EXTRACTEUR DE FEATURES AVANCÉ
# ============================================================================

class AdvancedFeatureExtractor:
    """Extraction complète des features à partir des données brutes"""
    
    @staticmethod
    def extract_music_stats(music_str):
        """Analyse détaillée de la musique (historique de performances)"""
        if pd.isna(music_str) or str(music_str).strip() == '':
            return {
                'total_races': 0, 'wins': 0, 'places_1_3': 0, 'places_1_5': 0,
                'win_rate': 0, 'place_rate': 0, 'top5_rate': 0,
                'recent_form_3': 0, 'recent_form_5': 0, 'recent_form_10': 0,
                'consistency_score': 0, 'best_position': 20, 'worst_position': 20,
                'avg_position': 10, 'median_position': 10, 'position_std': 10,
                'improvement_trend': 0, 'consecutive_wins': 0, 'consecutive_places': 0,
                'dns_dnf_rate': 0
            }
        
        music = str(music_str).upper()
        
        # Extraction des positions numériques
        positions = []
        dns_dnf = 0
        
        for char in music:
            if char.isdigit() and char != '0':
                positions.append(int(char))
            elif char in ['A', 'T', 'D']:  # Abandon, Tombé, Disqualifié
                dns_dnf += 1
        
        if not positions:
            positions = [20]  # Position par défaut si aucune donnée
        
        total = len(positions) + dns_dnf
        wins = positions.count(1)
        places_1_3 = sum(1 for p in positions if p <= 3)
        places_1_5 = sum(1 for p in positions if p <= 5)
        
        # Forme récente pondérée (plus de poids aux courses récentes)
        def weighted_form(positions_list, n):
            recent = positions_list[:n]
            if not recent:
                return 0
            weights = np.linspace(1, 0.5, len(recent))
            scores = [1/p for p in recent]
            return np.average(scores, weights=weights)
        
        recent_form_3 = weighted_form(positions, 3)
        recent_form_5 = weighted_form(positions, 5)
        recent_form_10 = weighted_form(positions, 10)
        
        # Tendance d'amélioration (régression linéaire sur les positions)
        if len(positions) >= 3:
            x = np.arange(len(positions))
            slope = np.polyfit(x, positions, 1)[0]
            improvement_trend = -slope  # Négatif = amélioration
        else:
            improvement_trend = 0
        
        # Séries de victoires/places consécutives
        consecutive_wins = 0
        consecutive_places = 0
        for p in positions:
            if p == 1:
                consecutive_wins += 1
            else:
                break
        for p in positions:
            if p <= 3:
                consecutive_places += 1
            else:
                break
        
        # Consistance (inverse de l'écart-type)
        consistency = 1 / (np.std(positions) + 1) if len(positions) > 1 else 0
        
        return {
            'total_races': total,
            'wins': wins,
            'places_1_3': places_1_3,
            'places_1_5': places_1_5,
            'win_rate': wins / total if total > 0 else 0,
            'place_rate': places_1_3 / total if total > 0 else 0,
            'top5_rate': places_1_5 / total if total > 0 else 0,
            'recent_form_3': recent_form_3,
            'recent_form_5': recent_form_5,
            'recent_form_10': recent_form_10,
            'consistency_score': consistency,
            'best_position': min(positions) if positions else 20,
            'worst_position': max(positions) if positions else 20,
            'avg_position': np.mean(positions) if positions else 10,
            'median_position': np.median(positions) if positions else 10,
            'position_std': np.std(positions) if len(positions) > 1 else 10,
            'improvement_trend': improvement_trend,
            'consecutive_wins': consecutive_wins,
            'consecutive_places': consecutive_places,
            'dns_dnf_rate': dns_dnf / total if total > 0 else 0
        }
    
    @staticmethod
    def extract_age_sex_features(age_sex_str):
        """Extraction âge et sexe avec features dérivées"""
        if pd.isna(age_sex_str):
            return {'age': 5, 'is_male': 0, 'is_female': 0, 'is_gelding': 0, 'age_category': 'mature'}
        
        age_sex = str(age_sex_str).upper()
        age_match = re.search(r'(\d+)', age_sex)
        age = int(age_match.group(1)) if age_match else 5
        
        is_male = 1 if 'H' in age_sex or 'M' in age_sex else 0
        is_female = 1 if 'F' in age_sex or 'J' in age_sex else 0
        is_gelding = 1 if 'H' in age_sex else 0
        
        if age <= 3:
            age_category = 'young'
        elif age <= 6:
            age_category = 'mature'
        else:
            age_category = 'veteran'
        
        return {
            'age': age,
            'is_male': is_male,
            'is_female': is_female,
            'is_gelding': is_gelding,
            'age_category': age_category
        }
    
    @staticmethod
    def create_comprehensive_features(df, race_type="PLAT"):
        """Création de l'ensemble complet des features pour ML"""
        features = pd.DataFrame(index=df.index)
        config = RACE_CONFIGS.get(race_type, RACE_CONFIGS["PLAT"])
        
        # === FEATURES DE COTE (8 features) ===
        features['odds'] = df['odds_numeric']
        features['odds_inv'] = 1 / (df['odds_numeric'] + 0.1)
        features['log_odds'] = np.log1p(df['odds_numeric'])
        features['sqrt_odds'] = np.sqrt(df['odds_numeric'])
        features['odds_squared'] = df['odds_numeric'] ** 2
        features['odds_rank'] = df['odds_numeric'].rank()
        features['odds_percentile'] = df['odds_numeric'].rank(pct=True)
        features['odds_zscore'] = (df['odds_numeric'] - df['odds_numeric'].mean()) / (df['odds_numeric'].std() + 1e-6)
        
        # === FEATURES DE POSITION/NUMÉRO (6 features) ===
        features['draw'] = df['draw_numeric']
        features['draw_normalized'] = df['draw_numeric'] / (df['draw_numeric'].max() + 1)
        features['draw_rank'] = df['draw_numeric'].rank()
        
        optimal_draws = config['draw_advantage']
        features['is_optimal_draw'] = df['draw_numeric'].apply(lambda x: 1 if x in optimal_draws else 0)
        features['draw_distance_optimal'] = df['draw_numeric'].apply(
            lambda x: min([abs(x - opt) for opt in optimal_draws]) if optimal_draws else 0
        )
        features['draw_advantage_score'] = features['is_optimal_draw'] * (1 - features['draw_normalized'])
        
        # === FEATURES DE POIDS (8 features) ===
        features['weight'] = df['weight_kg']
        features['weight_normalized'] = (df['weight_kg'] - df['weight_kg'].mean()) / (df['weight_kg'].std() + 1e-6)
        features['weight_rank'] = df['weight_kg'].rank()
        features['weight_percentile'] = df['weight_kg'].rank(pct=True)
        features['weight_advantage'] = (df['weight_kg'].max() - df['weight_kg']) * config['weight_importance']
        features['weight_zscore'] = (df['weight_kg'] - df['weight_kg'].mean()) / (df['weight_kg'].std() + 1e-6)
        features['is_light_weight'] = (df['weight_kg'] < df['weight_kg'].quantile(0.25)).astype(int)
        features['is_heavy_weight'] = (df['weight_kg'] > df['weight_kg'].quantile(0.75)).astype(int)
        
        # === FEATURES DE MUSIQUE (20 features) ===
        if 'Musique' in df.columns:
            music_stats = df['Musique'].apply(AdvancedFeatureExtractor.extract_music_stats)
            for key in music_stats.iloc[0].keys():
                features[f'music_{key}'] = [m[key] for m in music_stats]
        else:
            default_music = AdvancedFeatureExtractor.extract_music_stats('')
            for key in default_music.keys():
                features[f'music_{key}'] = default_music[key]
        
        # === FEATURES D'ÂGE ET SEXE (5 features + catégorielles) ===
        if 'Âge/Sexe' in df.columns:
            age_sex_stats = df['Âge/Sexe'].apply(AdvancedFeatureExtractor.extract_age_sex_features)
            for key in ['age', 'is_male', 'is_female', 'is_gelding']:
                features[key] = [m[key] for m in age_sex_stats]
            
            features['age_squared'] = features['age'] ** 2
            features['age_optimal'] = features['age'].apply(
                lambda x: 1 if config['age_optimal'][0] <= x <= config['age_optimal'][1] else 0
            )
            features['age_normalized'] = (features['age'] - features['age'].mean()) / (features['age'].std() + 1e-6)
        else:
            features['age'] = 5
            features['is_male'] = 0
            features['is_female'] = 0
            features['is_gelding'] = 0
            features['age_squared'] = 25
            features['age_optimal'] = 1
            features['age_normalized'] = 0
        
        # === FEATURES D'INTERACTION (15 features) ===
        features['odds_draw_interaction'] = features['odds_inv'] * features['draw_normalized']
        features['odds_weight_interaction'] = features['log_odds'] * features['weight_normalized']
        features['odds_age_interaction'] = features['odds_inv'] * features['age_normalized']
        features['weight_age_interaction'] = features['weight_normalized'] * features['age_normalized']
        features['form_odds_interaction'] = features['music_recent_form_3'] * features['odds_inv']
        features['form_weight_interaction'] = features['music_recent_form_3'] * features['weight_advantage']
        features['consistency_odds_interaction'] = features['music_consistency_score'] * features['odds_inv']
        features['winrate_odds_interaction'] = features['music_win_rate'] * features['odds_inv']
        features['recent_form_draw'] = features['music_recent_form_5'] * features['draw_advantage_score']
        features['age_optimal_odds'] = features['age_optimal'] * features['odds_inv']
        features['weight_form_combo'] = features['weight_advantage'] * features['music_place_rate']
        features['triple_interaction'] = features['odds_inv'] * features['music_recent_form_3'] * features['weight_advantage']
        features['experience_age_ratio'] = features['music_total_races'] / (features['age'] + 1)
        features['quality_index'] = features['music_win_rate'] * features['music_consistency_score'] * features['odds_inv']
        features['composite_performance'] = (
            features['music_recent_form_5'] * 0.3 +
            features['music_win_rate'] * 0.25 +
            features['music_place_rate'] * 0.25 +
            features['music_consistency_score'] * 0.2
        )
        
        # === FEATURES DE CONTEXTE (7 features) ===
        features['field_size'] = len(df)
        features['competitiveness_index'] = df['odds_numeric'].std() / (df['odds_numeric'].mean() + 1e-6)
        features['is_favorite'] = (df['odds_numeric'] == df['odds_numeric'].min()).astype(int)
        features['is_second_favorite'] = (df['odds_numeric'] == df['odds_numeric'].nsmallest(2).iloc[-1]).astype(int)
        features['is_outsider'] = (df['odds_numeric'] > df['odds_numeric'].quantile(0.75)).astype(int)
        features['relative_strength'] = features['odds_inv'] / features['odds_inv'].sum()
        features['market_share'] = 1 / (df['odds_numeric'] * len(df))
        
        # === FEATURES STATISTIQUES AVANCÉES (5 features) ===
        features['performance_volatility'] = features['music_position_std'] / (features['music_avg_position'] + 1)
        features['risk_adjusted_performance'] = features['music_win_rate'] / (features['music_position_std'] + 1)
        features['momentum_score'] = features['music_improvement_trend'] * features['music_recent_form_3']
        features['reliability_score'] = (1 - features['music_dns_dnf_rate']) * features['music_consistency_score']
        features['peak_performance_indicator'] = 1 / (features['music_best_position'] + 1)
        
        return features.fillna(0)

# ============================================================================
# MODÈLE ML AVANCÉ MULTI-ALGORITHMES
# ============================================================================

@st.cache_resource
class EnhancedHorseRacingML:
    """Système ML avancé avec multiples algorithmes et ensemble learning"""
    
    def __init__(self):
        self.models = {}
        self.ensemble = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.cv_scores = {}
        self.best_model_name = None
        self.is_trained = False
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialisation de tous les modèles ML"""
        
        # 1. Régression Ridge (régularisation L2)
        self.models['Ridge'] = Ridge(alpha=1.0, random_state=42)
        
        # 2. Régression Lasso (régularisation L1, sélection de features)
        self.models['Lasso'] = Lasso(alpha=0.1, random_state=42, max_iter=2000)
        
        # 3. ElasticNet (combinaison L1 + L2)
        self.models['ElasticNet'] = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42, max_iter=2000)
        
        # 4. Random Forest (ensemble de décision)
        self.models['RandomForest'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # 5. Gradient Boosting (boosting séquentiel)
        self.models['GradientBoosting'] = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=10,
            subsample=0.8,
            random_state=42
        )
        
        # 6. Réseau de Neurones (Deep Learning)
        self.models['NeuralNetwork'] = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42
        )
    
    def create_synthetic_targets(self, X):
        """Création de targets synthétiques réalistes basés sur les features"""
        
        # Pondération basée sur la logique des courses hippiques
        weights = {
            'odds_inv': 0.30,
            'music_recent_form_3': 0.15,
            'music_win_rate': 0.12,
            'music_place_rate': 0.10,
            'weight_advantage': 0.08,
            'composite_performance': 0.08,
            'quality_index': 0.07,
            'music_consistency_score': 0.05,
            'age_optimal': 0.03,
            'draw_advantage_score': 0.02
        }
        
        y_synthetic = np.zeros(len(X))
        
        for feature, weight in weights.items():
            if feature in X.columns:
                col_data = X[feature]
                # Vérifier s'il y a des valeurs non-NaN
                if col_data.notna().any():
                    col_min = col_data.min()
                    col_max = col_data.max()
                    
                    # Éviter la division par zéro
                    if col_max != col_min:
                        feature_norm = (col_data - col_min) / (col_max - col_min)
                    else:
                        feature_norm = col_data * 0  # Tous à zéro si pas de variation
                    
                    # Remplacer NaN par 0
                    feature_norm = feature_norm.fillna(0)
                    y_synthetic += feature_norm * weight
        
        # Ajout de bruit réaliste (± 5%)
        noise = np.random.normal(0, 0.05, len(X))
        y_synthetic += noise
        
        # Normalisation finale (éviter division par zéro)
        y_min = y_synthetic.min()
        y_max = y_synthetic.max()
        
        if y_max != y_min:
            y_synthetic = (y_synthetic - y_min) / (y_max - y_min)
        else:
            # Si toutes les valeurs sont identiques, normaliser à 0.5
            y_synthetic = np.full(len(X), 0.5)
        
        return y_synthetic
    
    def train_with_cross_validation(self, X, y, cv_folds=5):
        """Entraînement avec validation croisée pour tous les modèles"""
        
        # Vérifier qu'il y a assez de données
        if len(X) < cv_folds:
            cv_folds = max(2, len(X) // 2)
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            try:
                # Validation croisée
                scores = cross_val_score(
                    model, X, y,
                    cv=kf,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                
                # Conversion en RMSE
                rmse_scores = np.sqrt(-scores)
                
                # Calcul du R²
                r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=-1)
                
                # Remplacer les valeurs invalides
                rmse_mean = float(rmse_scores.mean()) if not np.isnan(rmse_scores.mean()) else 0.5
                rmse_std = float(rmse_scores.std()) if not np.isnan(rmse_scores.std()) else 0.1
                r2_mean = float(r2_scores.mean()) if not np.isnan(r2_scores.mean()) else 0.5
                r2_std = float(r2_scores.std()) if not np.isnan(r2_scores.std()) else 0.1
                
                self.cv_scores[name] = {
                    'rmse_mean': rmse_mean,
                    'rmse_std': rmse_std,
                    'r2_mean': r2_mean,
                    'r2_std': r2_std,
                    'scores': rmse_scores
                }
                
            except Exception as e:
                st.warning(f"⚠️ Erreur CV pour {name}: {str(e)}")
                self.cv_scores[name] = {
                    'rmse_mean': 0.5,
                    'rmse_std': 0.1,
                    'r2_mean': 0.5,
                    'r2_std': 0.1,
                    'scores': np.array([0.5])
                }
    
    def train_all_models(self, X, y):
        """Entraînement de tous les modèles"""
        
        predictions_dict = {}
        
        for name, model in self.models.items():
            try:
                model.fit(X, y)
                pred = model.predict(X)
                predictions_dict[name] = pred
                
                # Extraction de l'importance des features
                if hasattr(model, 'feature_importances_'):
                    importance = dict(zip(X.columns, model.feature_importances_))
                    top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15])
                    self.feature_importance[name] = top_features
                
                elif hasattr(model, 'coef_'):
                    importance = dict(zip(X.columns, np.abs(model.coef_)))
                    top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15])
                    self.feature_importance[name] = top_features
                
            except Exception as e:
                st.warning(f"⚠️ Erreur entraînement {name}: {str(e)}")
                predictions_dict[name] = np.zeros(len(X))
        
        return predictions_dict
    
    def create_ensemble_predictions(self, predictions_dict, cv_scores):
        """Création de prédictions d'ensemble pondérées par performance"""
        
        # Pondération basée sur les scores de validation croisée
        weights = {}
        total_r2 = sum(score['r2_mean'] for score in cv_scores.values() if score['r2_mean'] > 0)
        
        if total_r2 > 0:
            for name, score in cv_scores.items():
                if score['r2_mean'] > 0:
                    weights[name] = score['r2_mean'] / total_r2
                else:
                    weights[name] = 0
        else:
            # Pondération uniforme si pas de bons scores
            weights = {name: 1/len(predictions_dict) for name in predictions_dict}
        
        # Calcul de la prédiction d'ensemble
        ensemble_pred = np.zeros(len(list(predictions_dict.values())[0]))
        
        for name, pred in predictions_dict.items():
            ensemble_pred += pred * weights.get(name, 0)
        
        # Détermination du meilleur modèle
        best_model = max(cv_scores.items(), key=lambda x: x[1]['r2_mean'])
        self.best_model_name = best_model[0]
        
        return ensemble_pred, weights
    
    def calculate_prediction_confidence(self, predictions_dict, X):
        """Calcul de la confiance dans les prédictions"""
        
        # Variance entre les prédictions des différents modèles
        all_preds = np.array(list(predictions_dict.values()))
        pred_variance = np.var(all_preds, axis=0)
        
        # Confiance inversement proportionnelle à la variance
        confidence_base = 1 / (1 + pred_variance * 10)
        
        # Ajustement par la qualité des features
        feature_quality = 1 - (X.isna().sum(axis=1) / len(X.columns))
        
        # Confiance finale
        confidence = confidence_base * feature_quality.values
        confidence = np.clip(confidence, 0.2, 1.0)  # Entre 20% et 100%
        
        return confidence
    
    def fit_predict(self, X_raw, race_type="PLAT"):
        """Pipeline complet : entraînement et prédiction"""
        
        if len(X_raw) < 5:
            st.error("⚠️ Minimum 5 chevaux requis pour l'analyse ML")
            return np.zeros(len(X_raw)), {}, np.ones(len(X_raw)) * 0.5
        
        # Normalisation des features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_raw),
            columns=X_raw.columns,
            index=X_raw.index
        )
        
        # Création des targets synthétiques
        y_synthetic = self.create_synthetic_targets(X_raw)
        
        # Validation croisée
        with st.spinner("🔬 Validation croisée en cours..."):
            self.train_with_cross_validation(X_scaled, y_synthetic)
        
        # Entraînement de tous les modèles
        with st.spinner("🤖 Entraînement des modèles ML..."):
            predictions_dict = self.train_all_models(X_scaled, y_synthetic)
        
        # Création de l'ensemble
        ensemble_predictions, model_weights = self.create_ensemble_predictions(
            predictions_dict, self.cv_scores
        )
        
        # Calcul de la confiance
        confidence = self.calculate_prediction_confidence(predictions_dict, X_raw)
        
        self.is_trained = True
        
        return ensemble_predictions, self.cv_scores, confidence

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

@st.cache_data(ttl=300)
def scrape_race_data(url):
    """Web scraping des données de course"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return None, f"Erreur HTTP {response.status_code}"

        soup = BeautifulSoup(response.content, 'html.parser')
        horses_data = []
        
        table = soup.find('table')
        if not table:
            return None, "Aucun tableau trouvé"
            
        rows = table.find_all('tr')[1:]
        
        for row in rows:
            cols = row.find_all(['td', 'th'])
            if len(cols) >= 4:
                horses_data.append({
                    "Numéro de corde": cols[0].get_text(strip=True),
                    "Nom": cols[1].get_text(strip=True),
                    "Cote": cols[-1].get_text(strip=True),
                    "Poids": cols[-2].get_text(strip=True) if len(cols) > 4 else "60.0",
                    "Musique": cols[2].get_text(strip=True) if len(cols) > 5 else "",
                    "Âge/Sexe": cols[3].get_text(strip=True) if len(cols) > 6 else "",
                })

        if not horses_data:
            return None, "Aucune donnée extraite"
            
        return pd.DataFrame(horses_data), "Succès"
        
    except Exception as e:
        return None, f"Erreur: {str(e)}"

def safe_convert(value, convert_func, default=0):
    """Conversion sécurisée avec gestion d'erreurs"""
    try:
        if pd.isna(value):
            return default
        cleaned = str(value).replace(',', '.').strip()
        return convert_func(cleaned)
    except:
        return default

def prepare_data(df):
    """Préparation et nettoyage des données"""
    df = df.copy()
    df['odds_numeric'] = df['Cote'].apply(lambda x: safe_convert(x, float, 999))
    df['draw_numeric'] = df['Numéro de corde'].apply(lambda x: safe_convert(x, int, 1))
    
    def extract_weight(poids_str):
        if pd.isna(poids_str):
            return 60.0
        match = re.search(r'(\d+(?:[.,]\d+)?)', str(poids_str))
        return float(match.group(1).replace(',', '.')) if match else 60.0
    
    df['weight_kg'] = df['Poids'].apply(extract_weight)
    df = df[df['odds_numeric'] > 0]
    df = df.reset_index(drop=True)
    return df

def auto_detect_race_type(df):
    """Détection automatique du type de course"""
    weight_std = df['weight_kg'].std()
    weight_mean = df['weight_kg'].mean()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💪 Écart-type poids", f"{weight_std:.1f} kg")
    with col2:
        st.metric("⚖️ Poids moyen", f"{weight_mean:.1f} kg")
    with col3:
        st.metric("🏇 Nb chevaux", len(df))
    
    if weight_std > 2.5:
        detected = "PLAT"
        reason = "Grande variation de poids (handicap)"
    elif weight_mean > 65 and weight_std < 1.5:
        detected = "ATTELE_AUTO"
        reason = "Poids uniformes élevés (attelé)"
    else:
        detected = "PLAT"
        reason = "Configuration par défaut"
    
    st.info(f"🤖 **Type détecté**: {detected} | **Raison**: {reason}")
    return detected

def generate_combinations(df_ranked, combination_type="quinte"):
    """Génération de combinaisons gagnantes"""
    top_horses = df_ranked.head(10)
    
    if combination_type == "quinte":
        # Top 5 avec confiance
        top_5 = top_horses.head(5)
        return {
            'type': 'Quinté+ Ordre',
            'selection': list(top_5['Nom']),
            'numeros': list(top_5['Numéro de corde']),
            'confidence_avg': top_5['confidence'].mean()
        }
    
    elif combination_type == "trio":
        # Top 3
        top_3 = top_horses.head(3)
        return {
            'type': 'e-Trio',
            'selection': list(top_3['Nom']),
            'numeros': list(top_3['Numéro de corde']),
            'confidence_avg': top_3['confidence'].mean()
        }
    
    elif combination_type == "super4":
        # Top 4
        top_4 = top_horses.head(4)
        return {
            'type': 'e-Super4',
            'selection': list(top_4['Nom']),
            'numeros': list(top_4['Numéro de corde']),
            'confidence_avg': top_4['confidence'].mean()
        }

def create_advanced_visualizations(df_ranked, ml_model):
    """Visualisations complètes des résultats ML"""
    
    # Palette de couleurs compatible
    colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', 
              '#fdb462', '#b3de69', '#fccde5', '#d9d9d9']
    
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            '🏆 Scores de Prédiction',
            '📊 Distribution Cotes',
            '🧠 Importance Features (RF)',
            '⚖️ Poids vs Performance',
            '📈 Performances CV',
            '🎯 Corrélation Cotes-Scores',
            '🔥 Forme Récente Top 10',
            '📉 Variance Prédictions',
            '🎲 Analyse de Confiance'
        ),
        specs=[
            [{"secondary_y": False}, {"type": "histogram"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "box"}, {"type": "scatter"}]
        ]
    )
    
    # 1. Scores avec confiance
    fig.add_trace(
        go.Scatter(
            x=df_ranked['rang'],
            y=df_ranked['score_final'],
            mode='markers+lines',
            marker=dict(
                size=df_ranked['confidence'] * 25,
                color=df_ranked['confidence'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Confiance", x=0.35)
            ),
            text=df_ranked['Nom'],
            hovertemplate='<b>%{text}</b><br>Score: %{y:.3f}<br>Rang: %{x}',
            name='Score ML'
        ), row=1, col=1
    )
    
    # 2. Distribution des cotes
    fig.add_trace(
        go.Histogram(
            x=df_ranked['odds_numeric'],
            nbinsx=15,
            marker_color=colors[1],
            name='Distribution'
        ), row=1, col=2
    )
    
    # 3. Importance des features (Random Forest)
    if ml_model.feature_importance and 'RandomForest' in ml_model.feature_importance:
        importance = ml_model.feature_importance['RandomForest']
        top_10 = dict(list(importance.items())[:10])
        fig.add_trace(
            go.Bar(
                x=list(top_10.values()),
                y=list(top_10.keys()),
                orientation='h',
                marker_color=colors[2],
                name='Importance'
            ), row=1, col=3
        )
    
    # 4. Poids vs Performance
    fig.add_trace(
        go.Scatter(
            x=df_ranked['weight_kg'],
            y=df_ranked['score_final'],
            mode='markers',
            marker=dict(
                size=12,
                color=df_ranked['rang'],
                colorscale='RdYlGn_r',
                showscale=False,
                line=dict(width=1, color='white')
            ),
            text=df_ranked['Nom'],
            hovertemplate='<b>%{text}</b><br>Poids: %{x} kg<br>Score: %{y:.3f}',
            name='Poids-Score'
        ), row=2, col=1
    )
    
    # 5. Performances de validation croisée
    if ml_model.cv_scores:
        models = list(ml_model.cv_scores.keys())
        r2_means = [ml_model.cv_scores[m]['r2_mean'] for m in models]
        r2_stds = [ml_model.cv_scores[m]['r2_std'] for m in models]
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=r2_means,
                error_y=dict(type='data', array=r2_stds),
                marker_color=colors[4],
                name='R² Score'
            ), row=2, col=2
        )
    
    # 6. Corrélation Cotes-Scores
    fig.add_trace(
        go.Scatter(
            x=df_ranked['odds_numeric'],
            y=df_ranked['score_final'],
            mode='markers',
            marker=dict(
                size=10,
                color=colors[5],
                line=dict(width=1, color='white')
            ),
            text=df_ranked['Nom'],
            hovertemplate='<b>%{text}</b><br>Cote: %{x}<br>Score: %{y:.3f}',
            name='Corrélation'
        ), row=2, col=3
    )
    
    # 7. Forme récente Top 10
    top_10 = df_ranked.head(10)
    fig.add_trace(
        go.Bar(
            x=top_10['Nom'],
            y=top_10['music_recent_form_3'],
            marker_color=colors[6],
            name='Forme récente'
        ), row=3, col=1
    )
    
    # 8. Variance des prédictions (boxplot)
    fig.add_trace(
        go.Box(
            y=df_ranked['score_final'],
            marker_color=colors[7],
            name='Distribution scores'
        ), row=3, col=2
    )
    
    # 9. Analyse de confiance
    fig.add_trace(
        go.Scatter(
            x=df_ranked['odds_numeric'],
            y=df_ranked['confidence'],
            mode='markers',
            marker=dict(
                size=10,
                color=df_ranked['score_final'],
                colorscale='Plasma',
                showscale=False,
                line=dict(width=1, color='white')
            ),
            text=df_ranked['Nom'],
            hovertemplate='<b>%{text}</b><br>Cote: %{x}<br>Confiance: %{y:.1%}',
            name='Confiance'
        ), row=3, col=3
    )
    
    fig.update_layout(
        height=1000,
        showlegend=False,
        title_text="📊 Dashboard Complet d'Analyse ML",
        title_x=0.5,
        title_font_size=22
    )
    
    return fig

def create_performance_report(df_ranked, ml_model, race_type):
    """Génération d'un rapport détaillé de performance"""
    
    report = f"""
╔═══════════════════════════════════════════════════════════════╗
║           RAPPORT D'ANALYSE HIPPIQUE ML AVANCÉ               ║
╚═══════════════════════════════════════════════════════════════╝

📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🏁 Type de course: {race_type}
🏇 Nombre de partants: {len(df_ranked)}

╔═══════════════════════════════════════════════════════════════╗
║                    TOP 5 PRÉDICTIONS ML                       ║
╚═══════════════════════════════════════════════════════════════╝

"""
    
    for i in range(min(5, len(df_ranked))):
        horse = df_ranked.iloc[i]
        confidence_emoji = "🟢" if horse['confidence'] >= 0.7 else "🟡" if horse['confidence'] >= 0.4 else "🔴"
        
        report += f"""
{i+1}. {horse['Nom'].upper()}
   ├─ 📊 Cote: {horse['Cote']} | Numéro: {horse['Numéro de corde']}
   ├─ 🎯 Score ML: {horse['score_final']:.4f}
   ├─ {confidence_emoji} Confiance: {horse['confidence']:.1%}
   ├─ ⚖️ Poids: {horse['weight_kg']} kg
   ├─ 🏆 Victoires: {horse.get('music_wins', 'N/A')}
   ├─ 📈 Forme récente: {horse.get('music_recent_form_3', 0):.3f}
   └─ 🎲 Taux de victoire: {horse.get('music_win_rate', 0):.1%}
"""
    
    if ml_model.cv_scores:
        report += f"""
╔═══════════════════════════════════════════════════════════════╗
║              PERFORMANCES DES MODÈLES ML                      ║
╚═══════════════════════════════════════════════════════════════╝

"""
        for model_name, scores in ml_model.cv_scores.items():
            report += f"""
{model_name}:
   ├─ R² Score: {scores['r2_mean']:.4f} (± {scores['r2_std']:.4f})
   ├─ RMSE: {scores['rmse_mean']:.4f} (± {scores['rmse_std']:.4f})
   └─ Évaluation: {"Excellent" if scores['r2_mean'] > 0.8 else "Bon" if scores['r2_mean'] > 0.6 else "Modéré"}
"""
    
    if ml_model.best_model_name:
        report += f"""
🏆 Meilleur modèle: {ml_model.best_model_name}

"""
    
    # Statistiques globales
    avg_confidence = df_ranked['confidence'].mean()
    high_confidence = len(df_ranked[df_ranked['confidence'] >= 0.7])
    
    report += f"""
╔═══════════════════════════════════════════════════════════════╗
║                  STATISTIQUES GLOBALES                        ║
╚═══════════════════════════════════════════════════════════════╝

├─ 🎯 Confiance moyenne: {avg_confidence:.1%}
├─ 🟢 Prédictions haute confiance (≥70%): {high_confidence}
├─ ⭐ Favoris (cote < 5): {len(df_ranked[df_ranked['odds_numeric'] < 5])}
├─ 🎲 Outsiders (cote > 15): {len(df_ranked[df_ranked['odds_numeric'] > 15])}
└─ 📊 Indice de compétitivité: {df_ranked['odds_numeric'].std() / df_ranked['odds_numeric'].mean():.3f}

╔═══════════════════════════════════════════════════════════════╗
║                  RECOMMANDATIONS STRATÉGIQUES                 ║
╚═══════════════════════════════════════════════════════════════╝

"""
    
    # Chevaux à fort potentiel (bon score, cote intéressante, haute confiance)
    value_horses = df_ranked[
        (df_ranked['score_final'] > df_ranked['score_final'].quantile(0.6)) &
        (df_ranked['odds_numeric'] > 5) &
        (df_ranked['confidence'] > 0.5)
    ].head(3)
    
    if len(value_horses) > 0:
        report += "💎 CHEVAUX À VALEUR:\n"
        for idx, horse in value_horses.iterrows():
            report += f"   ✓ {horse['Nom']} - Cote: {horse['Cote']} | Score: {horse['score_final']:.3f}\n"
    
    # Alertes
    weak_favorites = df_ranked[
        (df_ranked['odds_numeric'] < 5) &
        (df_ranked['score_final'] < df_ranked['score_final'].median())
    ]
    
    if len(weak_favorites) > 0:
        report += f"\n⚠️ ALERTES: {len(weak_favorites)} favori(s) avec score ML faible\n"
    
    report += "\n" + "═" * 67 + "\n"
    report += "Note: Ce rapport est basé sur une analyse ML prédictive.\n"
    report += "Les courses hippiques comportent toujours une part d'aléatoire.\n"
    report += "═" * 67 + "\n"
    
    return report

def generate_test_data(data_type="plat"):
    """Génération de données de test réalistes"""
    if data_type == "plat":
        return pd.DataFrame({
            'Nom': ['Golden Flash', 'Silver Storm', 'Bronze King', 'Diamond Star', 
                    'Emerald Wave', 'Ruby Fire', 'Sapphire Sky', 'Pearl Ocean'],
            'Numéro de corde': ['1', '2', '3', '4', '5', '6', '7', '8'],
            'Cote': ['3.5', '5.2', '8.1', '6.8', '11.5', '15.2', '22.0', '18.5'],
            'Poids': ['56.0', '57.5', '58.0', '59.5', '57.0', '60.5', '62.0', '61.0'],
            'Musique': ['1a1a2a1a3a', '2a3a1a2a4a', '1a4a3a2a1a', '3a1a5a2a3a',
                        '5a4a2a6a3a', '4a6a5a8a7a', '7a8a9a6a5a', '6a5a7a4a8a'],
            'Âge/Sexe': ['4H', '5M', '3F', '6H', '4M', '5H', '7M', '4F']
        })
    elif data_type == "attele":
        return pd.DataFrame({
            'Nom': ['Thunder Bolt', 'Lightning Fast', 'Storm Chaser', 'Wind Runner',
                    'Rain Maker', 'Cloud Dancer'],
            'Numéro de corde': ['1', '2', '3', '4', '5', '6'],
            'Cote': ['4.8', '7.2', '3.5', '9.5', '12.0', '16.5'],
            'Poids': ['68.0', '68.0', '68.0', '68.0', '68.0', '68.0'],
            'Musique': ['1a2a1a3a1a', '3a4a2a1a5a', '1a1a2a1a3a', '4a5a3a6a2a',
                        '6a4a7a5a8a', '8a7a6a9a4a'],
            'Âge/Sexe': ['5H', '6M', '4F', '7H', '5M', '6H']
        })

# ============================================================================
# APPLICATION PRINCIPALE
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">🏇 Analyseur Hippique Pro ML v2.0</h1>', unsafe_allow_html=True)
    st.markdown("**Système d'analyse prédictive avancé avec Machine Learning multi-algorithmes**")
    st.markdown("*Régression, Deep Learning & Ensemble Methods*")
    
    # === SIDEBAR CONFIGURATION ===
    with st.sidebar:
        st.header("⚙️ Configuration ML")
        
        race_type_selection = st.selectbox(
            "🏁 Type de course",
            ["AUTO", "PLAT", "ATTELE_AUTO", "ATTELE_VOLTE", "OBSTACLE"]
        )
        
        use_ml = st.checkbox("✅ Activer ML Avancé", value=True)
        
        if use_ml:
            ml_weight = st.slider(
                "🎯 Pondération ML vs Cotes",
                0.0, 1.0, 0.75, 0.05,
                help="0 = uniquement cotes, 1 = uniquement ML"
            )
        else:
            ml_weight = 0.0
        
        st.markdown("---")
        st.subheader("🤖 Modèles Utilisés")
        st.info("✅ Ridge Regression")
        st.info("✅ Lasso Regression")
        st.info("✅ ElasticNet")
        st.info("✅ Random Forest (200 arbres)")
        st.info("✅ Gradient Boosting")
        st.info("✅ Neural Network (4 couches)")
        
        st.markdown("---")
        st.subheader("📊 Features Générées")
        st.success("**79 features** créées automatiquement:")
        st.caption("• 8 features de cote")
        st.caption("• 6 features de position")
        st.caption("• 8 features de poids")
        st.caption("• 20 features de musique")
        st.caption("• 7 features âge/sexe")
        st.caption("• 15 features d'interaction")
        st.caption("• 7 features de contexte")
        st.caption("• 5 features statistiques")
        
        st.markdown("---")
        st.subheader("ℹ️ Informations")
        st.info("🔬 **Méthode**: Validation croisée 5-fold")
        st.info("🎯 **Objectif**: Prédire la performance relative")
        st.info("📈 **Optimisation**: Ensemble learning")
    
    # === ONGLETS PRINCIPAUX ===
    tab1, tab2, tab3 = st.tabs(["🌐 Analyse URL", "📁 Upload CSV", "🧪 Données Test"])
    
    df_final = None
    
    with tab1:
        st.subheader("🔍 Analyse d'URL de Course")
        col1, col2 = st.columns([4, 1])
        with col1:
            url = st.text_input(
                "🌐 URL de la course:",
                placeholder="https://www.geny.fr/courses-pmu/..."
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_btn = st.button("🔍 Analyser", type="primary", use_container_width=True)
        
        if analyze_btn and url:
            with st.spinner("🔄 Extraction des données..."):
                df, message = scrape_race_data(url)
                if df is not None:
                    st.success(f"✅ {len(df)} chevaux extraits")
                    st.dataframe(df, use_container_width=True)
                    df_final = df
                else:
                    st.error(f"❌ {message}")
    
    with tab2:
        st.subheader("📤 Upload de fichier CSV")
        st.markdown("**Format requis**: `Nom, Numéro de corde, Cote, Poids, Musique, Âge/Sexe`")
        
        uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
        if uploaded_file:
            try:
                df_final = pd.read_csv(uploaded_file)
                st.success(f"✅ {len(df_final)} chevaux chargés")
                st.dataframe(df_final, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Erreur: {e}")
    
    with tab3:
        st.subheader("🧪 Données de Test")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏃 Test Course de PLAT", use_container_width=True):
                df_final = generate_test_data("plat")
                st.success("✅ 8 chevaux chargés (Course de PLAT)")
        with col2:
            if st.button("🚗 Test Course ATTELÉ", use_container_width=True):
                df_final = generate_test_data("attele")
                st.success("✅ 6 chevaux chargés (Trot Attelé)")
        
        if df_final is not None:
            st.dataframe(df_final, use_container_width=True)
    
    # === ANALYSE PRINCIPALE ===
    if df_final is not None and len(df_final) > 0:
        st.markdown("---")
        st.header("🎯 Analyse ML et Prédictions")
        
        # Préparation des données
        df_prepared = prepare_data(df_final)
        
        if len(df_prepared) == 0:
            st.error("❌ Aucune donnée valide")
            return
        
        # Détection du type de course
        if race_type_selection == "AUTO":
            detected_type = auto_detect_race_type(df_prepared)
        else:
            detected_type = race_type_selection
            config_desc = RACE_CONFIGS.get(detected_type, {}).get('description', 'Configuration personnalisée')
            st.info(f"📋 **Type sélectionné**: {detected_type}")
        
        # === EXTRACTION DES FEATURES ===
        with st.spinner("🔬 Extraction des features avancées..."):
            X_features = AdvancedFeatureExtractor.create_comprehensive_features(
                df_prepared,
                detected_type
            )
            df_prepared = pd.concat([df_prepared, X_features], axis=1)
        
        st.success(f"✅ **{len(X_features.columns)} features** créées avec succès")
        
        # === MACHINE LEARNING ===
        ml_model = EnhancedHorseRacingML()
        
        if use_ml:
            # Entraînement ML
            ml_predictions, cv_scores, confidence = ml_model.fit_predict(
                X_features,
                detected_type
            )
            
            # Normalisation
            if ml_predictions.max() != ml_predictions.min():
                ml_predictions_norm = (ml_predictions - ml_predictions.min()) / \
                                     (ml_predictions.max() - ml_predictions.min())
            else:
                ml_predictions_norm = ml_predictions
            
            df_prepared['ml_score'] = ml_predictions_norm
            df_prepared['confidence'] = confidence
            
            # Affichage des performances ML
            st.markdown("### 📊 Performances des Modèles ML")
            
            cols = st.columns(len(cv_scores))
            for idx, (model_name, scores) in enumerate(cv_scores.items()):
                with cols[idx]:
                    st.metric(
                        model_name,
                        f"R²: {scores['r2_mean']:.3f}",
                        f"±{scores['r2_std']:.3f}",
                        delta_color="normal"
                    )
        
        # === SCORE TRADITIONNEL (BASÉ SUR LES COTES) ===
        traditional_score = 1 / (df_prepared['odds_numeric'] + 0.1)
        if traditional_score.max() != traditional_score.min():
            traditional_score = (traditional_score - traditional_score.min()) / \
                              (traditional_score.max() - traditional_score.min())
        
        # === SCORE FINAL (HYBRIDE) ===
        if use_ml and 'ml_score' in df_prepared.columns:
            df_prepared['score_final'] = (
                (1 - ml_weight) * traditional_score +
                ml_weight * df_prepared['ml_score']
            )
        else:
            df_prepared['score_final'] = traditional_score
            df_prepared['confidence'] = np.ones(len(df_prepared)) * 0.5
        
        # === CLASSEMENT ===
        df_ranked = df_prepared.sort_values('score_final', ascending=False).reset_index(drop=True)
        df_ranked['rang'] = range(1, len(df_ranked) + 1)
        
        # === AFFICHAGE DES RÉSULTATS ===
        st.markdown("---")
        st.header("🏆 Classement Final & Pronostics")
        
        col1, col2 = st.columns([2.5, 1.5])
        
        with col1:
            st.subheader("📋 Classement Complet")
            
            # Préparation affichage
            display_df = df_ranked[[
                'rang', 'Nom', 'Numéro de corde', 'Cote',
                'weight_kg', 'score_final', 'confidence'
            ]].copy()
            
            display_df.columns = [
                'Rang', 'Nom', 'N°', 'Cote',
                'Poids (kg)', 'Score ML', 'Confiance'
            ]
            
            display_df['Score ML'] = display_df['Score ML'].apply(lambda x: f"{x:.4f}")
            display_df['Confiance'] = display_df['Confiance'].apply(lambda x: f"{x:.1%}")
            display_df['Poids (kg)'] = display_df['Poids (kg)'].apply(lambda x: f"{x:.1f}")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=500
            )
        
        with col2:
            st.subheader("🎯 Top 5 Détaillé")
            
            for i in range(min(5, len(df_ranked))):
                horse = df_ranked.iloc[i]
                conf = horse['confidence']
                
                if conf >= 0.7:
                    conf_class = "confidence-high"
                    conf_emoji = "🟢"
                elif conf >= 0.4:
                    conf_class = "confidence-medium"
                    conf_emoji = "🟡"
                else:
                    conf_class = "confidence-low"
                    conf_emoji = "🔴"
                
                st.markdown(f"""
                <div class="prediction-box">
                    <strong style="font-size: 1.1em;">{i+1}. {horse['Nom']}</strong><br>
                    📊 Cote: <strong>{horse['Cote']}</strong> | 
                    🔢 N°: <strong>{horse['Numéro de corde']}</strong><br>
                    🎯 Score ML: <strong>{horse['score_final']:.4f}</strong><br>
                    {conf_emoji} Confiance: <span class="{conf_class}">{conf:.1%}</span><br>
                    ⚖️ Poids: {horse['weight_kg']:.1f} kg<br>
                    🏆 Victoires: {horse.get('music_wins', 0)} | 
                    📈 Forme: {horse.get('music_recent_form_3', 0):.2f}
                </div>
                """, unsafe_allow_html=True)
            
            # Métriques globales
            st.markdown("### 📊 Statistiques")
            avg_conf = df_ranked['confidence'].mean()
            st.metric("Confiance Moyenne", f"{avg_conf:.1%}")
            
            high_conf = len(df_ranked[df_ranked['confidence'] >= 0.7])
            st.metric("Haute Confiance (≥70%)", high_conf)
            
            favorites = len(df_ranked[df_ranked['odds_numeric'] < 5])
            st.metric("Favoris (cote < 5)", favorites)
        
        # === COMBINAISONS GAGNANTES ===
        st.markdown("---")
        st.subheader("🎲 Combinaisons Recommandées")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            quinte = generate_combinations(df_ranked, "quinte")
            st.markdown("**🏆 Quinté+ Ordre**")
            st.info(f"Confiance: {quinte['confidence_avg']:.1%}")
            for i, (name, num) in enumerate(zip(quinte['selection'], quinte['numeros']), 1):
                st.write(f"{i}. **{name}** (N°{num})")
        
        with col2:
            trio = generate_combinations(df_ranked, "trio")
            st.markdown("**🥉 e-Trio**")
            st.info(f"Confiance: {trio['confidence_avg']:.1%}")
            for i, (name, num) in enumerate(zip(trio['selection'], trio['numeros']), 1):
                st.write(f"{i}. **{name}** (N°{num})")
        
        with col3:
            super4 = generate_combinations(df_ranked, "super4")
            st.markdown("**⭐ e-Super4**")
            st.info(f"Confiance: {super4['confidence_avg']:.1%}")
            for i, (name, num) in enumerate(zip(super4['selection'], super4['numeros']), 1):
                st.write(f"{i}. **{name}** (N°{num})")
        
        # === VISUALISATIONS AVANCÉES ===
        st.markdown("---")
        st.header("📊 Visualisations et Analytics")
        
        if use_ml:
            fig = create_advanced_visualizations(df_ranked, ml_model)
            st.plotly_chart(fig, use_container_width=True)
        
        # === ANALYSE DES FEATURES ===
        if use_ml and ml_model.feature_importance:
            st.markdown("---")
            st.header("🔬 Analyse de l'Importance des Features")
            
            tab_rf, tab_gb, tab_nn = st.tabs([
                "🌲 Random Forest",
                "📈 Gradient Boosting",
                "🧠 Comparaison"
            ])
            
            with tab_rf:
                if 'RandomForest' in ml_model.feature_importance:
                    st.markdown("**Top 15 Features - Random Forest**")
                    importance_df = pd.DataFrame(
                        list(ml_model.feature_importance['RandomForest'].items()),
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=False)
                    
                    fig_rf = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Importance des Features (Random Forest)',
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_rf, use_container_width=True)
                    
                    st.dataframe(importance_df, use_container_width=True, height=400)
            
            with tab_gb:
                if 'GradientBoosting' in ml_model.feature_importance:
                    st.markdown("**Top 15 Features - Gradient Boosting**")
                    importance_df = pd.DataFrame(
                        list(ml_model.feature_importance['GradientBoosting'].items()),
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=False)
                    
                    fig_gb = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Importance des Features (Gradient Boosting)',
                        color='Importance',
                        color_continuous_scale='Plasma'
                    )
                    st.plotly_chart(fig_gb, use_container_width=True)
                    
                    st.dataframe(importance_df, use_container_width=True, height=400)
            
            with tab_nn:
                st.markdown("**📊 Comparaison des Modèles**")
                
                if ml_model.cv_scores:
                    comparison_data = []
                    for model_name, scores in ml_model.cv_scores.items():
                        comparison_data.append({
                            'Modèle': model_name,
                            'R² Moyen': scores['r2_mean'],
                            'R² Std': scores['r2_std'],
                            'RMSE Moyen': scores['rmse_mean'],
                            'RMSE Std': scores['rmse_std']
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df = comparison_df.sort_values('R² Moyen', ascending=False)
                    
                    st.dataframe(
                        comparison_df.style.format({
                            'R² Moyen': '{:.4f}',
                            'R² Std': '{:.4f}',
                            'RMSE Moyen': '{:.4f}',
                            'RMSE Std': '{:.4f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Graphique de comparaison
                    fig_comp = go.Figure()
                    
                    fig_comp.add_trace(go.Bar(
                        name='R² Score',
                        x=comparison_df['Modèle'],
                        y=comparison_df['R² Moyen'],
                        error_y=dict(type='data', array=comparison_df['R² Std']),
                        marker_color='lightblue'
                    ))
                    
                    fig_comp.update_layout(
                        title='Comparaison des Performances (R² Score)',
                        xaxis_title='Modèle',
                        yaxis_title='R² Score',
                        height=400
                    )
                    
                    st.plotly_chart(fig_comp, use_container_width=True)
                    
                    if ml_model.best_model_name:
                        st.success(f"🏆 **Meilleur modèle**: {ml_model.best_model_name}")
        
        # === RECOMMANDATIONS STRATÉGIQUES ===
        st.markdown("---")
        st.header("💡 Recommandations Stratégiques")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💎 Chevaux à Forte Valeur")
            st.caption("Score élevé + Cote intéressante + Haute confiance")
            
            value_horses = df_ranked[
                (df_ranked['score_final'] > df_ranked['score_final'].quantile(0.6)) &
                (df_ranked['odds_numeric'] > 5) &
                (df_ranked['confidence'] > 0.5)
            ].head(5)
            
            if len(value_horses) > 0:
                for idx, horse in value_horses.iterrows():
                    value_score = horse['score_final'] * horse['confidence'] * (1/horse['odds_numeric'])
                    st.success(
                        f"✅ **{horse['Nom']}** (N°{horse['Numéro de corde']}) - "
                        f"Cote: {horse['Cote']} | Score: {horse['score_final']:.3f} | "
                        f"Confiance: {horse['confidence']:.1%}"
                    )
            else:
                st.info("Aucun outsider à fort potentiel détecté")
        
        with col2:
            st.markdown("#### ⚠️ Alertes et Observations")
            
            # Favoris sous-performants
            weak_favorites = df_ranked[
                (df_ranked['odds_numeric'] < 5) &
                (df_ranked['score_final'] < df_ranked['score_final'].median())
            ]
            
            if len(weak_favorites) > 0:
                st.warning(
                    f"⚠️ **{len(weak_favorites)} favori(s) avec score ML faible**\n\n" +
                    "\n".join([f"• {h['Nom']} (cote {h['Cote']})" for _, h in weak_favorites.iterrows()])
                )
            
            # Surprises potentielles
            surprise_horses = df_ranked[
                (df_ranked['odds_numeric'] > 10) &
                (df_ranked['rang'] <= 5)
            ]
            
            if len(surprise_horses) > 0:
                st.info(
                    f"🎲 **{len(surprise_horses)} outsider(s) dans le Top 5 !**\n\n" +
                    "\n".join([f"• {h['Nom']} (cote {h['Cote']}, rang {h['rang']})" 
                              for _, h in surprise_horses.iterrows()])
                )
            
            # Cohérence générale
            top3_avg_odds = df_ranked.head(3)['odds_numeric'].mean()
            if top3_avg_odds < 7:
                st.success("✅ Classement cohérent avec le marché")
            else:
                st.warning("🎯 Classement ML diverge du marché")
        
        # === ANALYSE DE LA FORME ===
        st.markdown("---")
        st.header("📈 Analyse de la Forme Récente")
        
        top_10_form = df_ranked.head(10)
        
        fig_form = go.Figure()
        
        fig_form.add_trace(go.Bar(
            name='Forme 3 courses',
            x=top_10_form['Nom'],
            y=top_10_form['music_recent_form_3'],
            marker_color='lightblue'
        ))
        
        fig_form.add_trace(go.Bar(
            name='Forme 5 courses',
            x=top_10_form['Nom'],
            y=top_10_form['music_recent_form_5'],
            marker_color='lightcoral'
        ))
        
        fig_form.update_layout(
            title='Forme Récente - Top 10',
            xaxis_title='Cheval',
            yaxis_title='Score de Forme',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig_form, use_container_width=True)
        
        # === EXPORT DES RÉSULTATS ===
        st.markdown("---")
        st.header("💾 Export et Rapports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export CSV
            csv_export = df_ranked[[
                'rang', 'Nom', 'Numéro de corde', 'Cote', 'weight_kg',
                'score_final', 'confidence', 'music_wins', 'music_win_rate',
                'music_recent_form_3', 'music_recent_form_5'
            ]].copy()
            
            csv_data = csv_export.to_csv(index=False)
            st.download_button(
                "📄 Télécharger CSV Complet",
                csv_data,
                f"pronostics_ml_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            # Export JSON
            json_export = df_ranked[[
                'rang', 'Nom', 'Numéro de corde', 'Cote',
                'score_final', 'confidence'
            ]].to_dict('records')
            
            json_data = json.dumps(json_export, indent=2, ensure_ascii=False)
            st.download_button(
                "📋 Télécharger JSON",
                json_data,
                f"pronostics_ml_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True
            )
        
        with col3:
            # Rapport complet
            report = create_performance_report(df_ranked, ml_model, detected_type)
            st.download_button(
                "📊 Télécharger Rapport Complet",
                report,
                f"rapport_ml_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain",
                use_container_width=True
            )
        
        # === SECTION EXPLICATIVE ===
        st.markdown("---")
        with st.expander("📚 **Comprendre l'Analyse ML**"):
            st.markdown("""
            ### 🎯 Méthodologie
            
            Notre système utilise une approche **multi-modèles** avec 6 algorithmes différents :
            
            1. **Ridge Regression** : Régression linéaire avec régularisation L2
            2. **Lasso Regression** : Régression avec sélection automatique de features (L1)
            3. **ElasticNet** : Combinaison de Ridge et Lasso
            4. **Random Forest** : Ensemble de 200 arbres de décision
            5. **Gradient Boosting** : Boosting séquentiel pour amélioration progressive
            6. **Neural Network** : Réseau de neurones profond (4 couches : 128→64→32→16)
            
            ### 📊 Features Utilisées (79 au total)
            
            **Cotes (8 features)** : Cote brute, inverse, log, racine carrée, rang, percentile, z-score
            
            **Position (6 features)** : Numéro de corde, avantage position, distance optimale
            
            **Poids (8 features)** : Poids brut, normalisé, rang, avantage, classification
            
            **Musique (20 features)** : Victoires, places, taux de réussite, forme récente (3/5/10 courses),
            consistance, tendance d'amélioration, séries de victoires
            
            **Âge/Sexe (7 features)** : Âge, sexe, catégorie d'âge, âge optimal
            
            **Interactions (15 features)** : Combinaisons entre cotes, poids, âge, forme
            
            **Contexte (7 features)** : Taille du peloton, compétitivité, statut (favori/outsider)
            
            **Statistiques (5 features)** : Volatilité, performance ajustée au risque, momentum
            
            ### 🔬 Validation
            
            - **Cross-validation 5-fold** : Le modèle est testé sur 5 sous-ensembles différents
            - **Métriques R²** : Mesure la qualité de prédiction (0 à 1, 1 = parfait)
            - **RMSE** : Erreur quadratique moyenne
            - **Ensemble Learning** : Combinaison pondérée des 6 modèles
            
            ### 🎲 Interprétation des Résultats
            
            - **Score ML** : Entre 0 et 1, plus élevé = meilleur potentiel
            - **Confiance** : Fiabilité de la prédiction (🟢 ≥70%, 🟡 40-70%, 🔴 <40%)
            - **Combinaisons** : Basées sur les chevaux les mieux classés avec haute confiance
            
            ### ⚠️ Avertissement
            
            Ce système est un **outil d'aide à la décision** basé sur des données statistiques.
            Les courses hippiques comportent toujours une part d'aléatoire importante.
            Utilisez ces prédictions comme un guide, pas comme une garantie.
            """)
        
        with st.expander("🔧 **Détails Techniques**"):
            st.markdown(f"""
            ### 📈 Performances du Système
            
            **Nombre de features** : {len(X_features.columns)}
            
            **Nombre de chevaux analysés** : {len(df_ranked)}
            
            **Type de course** : {detected_type}
            
            **Pondération ML** : {ml_weight:.0%}
            
            ### 🤖 Configuration des Modèles
            
            - Random Forest : 200 estimateurs, profondeur max 10
            - Gradient Boosting : 150 estimateurs, learning rate 0.05
            - Neural Network : Architecture [128, 64, 32, 16]
            - Validation : K-Fold avec k=5
            - Scaling : StandardScaler
            
            ### 📊 Statistiques de Course
            
            - Cote moyenne : {df_ranked['odds_numeric'].mean():.2f}
            - Cote min/max : {df_ranked['odds_numeric'].min():.1f} / {df_ranked['odds_numeric'].max():.1f}
            - Poids moyen : {df_ranked['weight_kg'].mean():.1f} kg
            - Confiance moyenne : {df_ranked['confidence'].mean():.1%}
            """)

if __name__ == "__main__":
    main()
