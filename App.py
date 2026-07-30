# -*- coding: utf-8 -*-
"""
Application Streamlit de pronostics hippiques
=============================================
Scrape une course depuis Geny.fr, calcule des probabilités de victoire
selon un modèle pondéré, et affiche les résultats.

Dépendances : streamlit, requests, beautifulsoup4, pandas, numpy
Installation : pip install -r requirements.txt
Lancement : streamlit run app.py
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import logging

# ---------- Module SCRAPER (intégré) ----------
logger = logging.getLogger(__name__)

def scrape_race(url):
    """
    Extrait les données d'une course depuis une URL geny.fr.
    Retourne un dictionnaire avec les métadonnées et la liste des participants.
    Chaque participant contient : numero, cheval, jockey, cote, poids, age, performances.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Erreur réseau : {e}")

    soup = BeautifulSoup(response.text, 'html.parser')

    # --- Recherche des participants ---
    participants = []
    # Sélecteurs possibles (à ajuster si la structure change)
    selectors = [
        'div.participant', 'div.partant', 'div.horse-item',
        'tr.participant', 'tr.line', 'div.item-participant'
    ]

    found = False
    for selector in selectors:
        items = soup.select(selector)
        if items:
            found = True
            for item in items:
                participant = extract_participant(item)
                if participant:
                    participants.append(participant)
            break  # on garde le premier sélecteur qui fonctionne

    if not found:
        # Fallback : chercher toutes les lignes de tableau avec assez de cellules
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 4:
                    # On tente d'extraire un nom de cheval (présence de lettres)
                    text = ' '.join([c.get_text(strip=True) for c in cells])
                    if re.search(r'[A-Za-z]', text):
                        participant = extract_participant_from_row(row)
                        if participant:
                            participants.append(participant)
            if participants:
                break

    if not participants:
        raise Exception("Aucun participant trouvé. La structure HTML a peut-être changé.")

    # --- Métadonnées de la course ---
    metadata = {}
    info_elem = soup.find('div', class_=re.compile(r'info-course|race-info|header-course'))
    if info_elem:
        text = info_elem.get_text(separator=' ', strip=True)
        distance_match = re.search(r'(\d+)\s*m', text)
        if distance_match:
            metadata['distance'] = distance_match.group(1) + ' m'
        terrain_match = re.search(r'Terrain\s*[:.]?\s*(\w+)', text, re.IGNORECASE)
        if terrain_match:
            metadata['terrain'] = terrain_match.group(1)
    else:
        metadata = {'distance': 'Non disponible', 'terrain': 'Non disponible'}

    return {
        'metadata': metadata,
        'participants': participants
    }


def extract_participant(item):
    """Extrait les données d'un participant à partir d'un élément BeautifulSoup."""
    def get_text(elem, selector):
        el = elem.select_one(selector)
        return el.get_text(strip=True) if el else ''

    def get_float(elem, selector):
        text = get_text(elem, selector)
        try:
            return float(text.replace(',', '.'))
        except:
            return None

    num = get_text(item, '.number, .num, .dossard')
    horse = get_text(item, '.horse, .cheval, .name')
    jockey = get_text(item, '.jockey, .driver')
    odds = get_float(item, '.odds, .cote, .ratio')
    weight = get_text(item, '.weight, .poids')
    age = get_text(item, '.age, .sex')
    perf = get_text(item, '.performance, .recent, .forme')

    # Si on n'a pas de nom de cheval, on ignore
    if not horse:
        return None

    return {
        'numero': num,
        'cheval': horse,
        'jockey': jockey,
        'cote': odds,
        'poids': weight,
        'age': age,
        'performances': perf
    }


def extract_participant_from_row(row):
    """Extrait les données d'une ligne de tableau (fallback)."""
    cells = row.find_all('td')
    if len(cells) < 4:
        return None
    texts = [c.get_text(strip=True) for c in cells]
    # On suppose que le cheval est une cellule contenant des lettres
    horse_idx = None
    for i, t in enumerate(texts):
        if re.search(r'[A-Za-z]', t) and not re.search(r'\d', t):
            horse_idx = i
            break
    if horse_idx is None:
        return None

    horse = texts[horse_idx]
    num = texts[0] if len(texts) > 0 else ''
    jockey = texts[horse_idx+1] if horse_idx+1 < len(texts) else ''
    # Essayer de trouver la cote (un nombre flottant)
    odds = None
    for t in texts:
        try:
            odds = float(t.replace(',', '.'))
            break
        except:
            pass
    weight = ''
    age = ''
    perf = ''
    for t in texts:
        if re.search(r'\d+[pP]?\s*\d+', t):
            perf = t
            break
    return {
        'numero': num,
        'cheval': horse,
        'jockey': jockey,
        'cote': odds,
        'poids': weight,
        'age': age,
        'performances': perf
    }


# ---------- Module ANALYSIS (intégré) ----------
def compute_probabilities(race_data):
    """
    Calcule les probabilités de victoire pour chaque cheval selon une
    pondération multi‑facteurs transparente :
      - Cote du marché (probabilité implicite) → poids 0.5
      - Forme récente (moyenne des places) → poids 0.3
      - Poids (plus il est bas, mieux c'est) → poids 0.2
    Les scores sont normalisés pour que la somme des probabilités = 1.
    """
    participants = race_data['participants']
    if not participants:
        return pd.DataFrame()

    df = pd.DataFrame(participants)

    # ---------- 1. Probabilité implicite par la cote ----------
    cotes = df['cote'].astype(float)
    cotes.fillna(cotes.mean(), inplace=True)
    cotes = cotes.clip(lower=0.1)          # éviter division par zéro
    prob_cote = 1.0 / cotes
    prob_cote = prob_cote / prob_cote.sum()  # normalisation

    # ---------- 2. Score de forme récente ----------
    def forme_score(perf_str):
        if pd.isna(perf_str) or perf_str == '':
            return np.nan
        numbers = re.findall(r'\d+', perf_str)
        if not numbers:
            return np.nan
        places = [int(n) for n in numbers]
        mean_place = np.mean(places)
        max_place = 20
        score = 1 - (mean_place - 1) / (max_place - 1)
        return np.clip(score, 0, 1)

    df['forme_score'] = df['performances'].apply(forme_score)
    df['forme_score'].fillna(df['forme_score'].mean(), inplace=True)

    # ---------- 3. Score lié au poids ----------
    def poids_score(weight_str):
        if pd.isna(weight_str) or weight_str == '':
            return np.nan
        numbers = re.findall(r'\d+', weight_str)
        if not numbers:
            return np.nan
        weight = float(numbers[0])
        min_w, max_w = 50, 70
        score = (max_w - weight) / (max_w - min_w)
        return np.clip(score, 0, 1)

    df['poids_score'] = df['poids'].apply(poids_score)
    df['poids_score'].fillna(df['poids_score'].mean(), inplace=True)

    # ---------- 4. Agrégation pondérée ----------
    forme_norm = df['forme_score'] / df['forme_score'].sum()
    poids_norm = df['poids_score'] / df['poids_score'].sum()

    weights = {'cote': 0.5, 'forme': 0.3, 'poids': 0.2}
    df['score'] = (weights['cote'] * prob_cote +
                   weights['forme'] * forme_norm +
                   weights['poids'] * poids_norm)

    df['probabilite'] = df['score'] / df['score'].sum()

    result = df[['cheval', 'probabilite', 'cote', 'jockey', 'performances']].copy()
    result['probabilite'] = result['probabilite'].round(4)
    return result


# ---------- Application Streamlit ----------
st.set_page_config(page_title="Pronostics Hippiques", layout="wide")
st.title("🏇 Analyse de course hippique - Pronostics probabilistes")

# Cache pour éviter de rescraper la même course
@st.cache_data(ttl=3600)
def load_race_data(url):
    return scrape_race(url)

# Interface
url = st.text_input("Entrez l'URL de la course Geny.fr :",
                    placeholder="https://www.geny.fr/...")

if st.button("🔍 Analyser la course"):
    if not url:
        st.error("Veuillez entrer une URL.")
    else:
        with st.spinner("Scraping et analyse en cours..."):
            try:
                race_data = load_race_data(url)
                participants = race_data.get('participants', [])
                if not participants:
                    st.error("Aucun participant trouvé. Vérifiez l'URL ou la structure de la page.")
                else:
                    prob_df = compute_probabilities(race_data)
                    if prob_df.empty:
                        st.error("Impossible de calculer les probabilités.")
                    else:
                        st.success("✅ Analyse terminée !")

                        meta = race_data.get('metadata', {})
                        if meta:
                            col1, col2 = st.columns(2)
                            col1.metric("Distance", meta.get('distance', 'N/A'))
                            col2.metric("Terrain", meta.get('terrain', 'N/A'))

                        st.subheader("📊 Classement des chevaux par probabilité de victoire")
                        display_df = prob_df.sort_values('probabilite', ascending=False)
                        display_df['probabilite'] = display_df['probabilite'].apply(lambda x: f"{x*100:.1f}%")
                        st.dataframe(display_df, use_container_width=True)

                        st.subheader("📈 Distribution des probabilités")
                        chart_data = display_df.set_index('cheval')['probabilite']
                        chart_data = chart_data.str.rstrip('%').astype(float)
                        st.bar_chart(chart_data)

                        with st.expander("🔎 Voir les données brutes extraites"):
                            st.json(race_data)

            except Exception as e:
                st.error(f"❌ Une erreur est survenue : {str(e)}")
