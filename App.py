import streamlit as st
import pandas as pd
import numpy as np
import easyocr
import cv2
from PIL import Image
import re
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ==============================
# CONFIG
# ==============================

st.set_page_config(
    page_title="IA Pronostics Hippiques",
    page_icon="🏇",
    layout="wide"
)

# ==============================
# OCR MODULE
# ==============================

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['fr'])

def extract_text_from_image(image):
    reader = load_ocr()
    result = reader.readtext(np.array(image), detail=0)
    return " ".join(result)

# ==============================
# DATA CLEANING
# ==============================

def extract_horse_data(raw_text):
    """
    Extraction intelligente basée sur regex adaptative
    """
    horses = []

    lines = raw_text.split("\n")

    for line in lines:
        pattern = r"(\d+)\s+([A-Za-z\-]+).*?(\d{1,2},?\d?)%.*?(\d{1,2},?\d?)%.*?(\d+:\d+\.\d+)"
        match = re.search(pattern, line)

        if match:
            number = int(match.group(1))
            name = match.group(2)
            driver_pct = float(match.group(3).replace(",", "."))
            trainer_pct = float(match.group(4).replace(",", "."))
            record = match.group(5)

            horses.append({
                "Numéro": number,
                "Nom": name,
                "Driver %": driver_pct,
                "Entraineur %": trainer_pct,
                "Record": record
            })

    return pd.DataFrame(horses)

def convert_record_to_seconds(record):
    try:
        minute, sec = record.split(":")
        return int(minute)*60 + float(sec)
    except:
        return np.nan

def musique_score(musique):
    """
    Scoring intelligent de la musique
    1 = 10 pts, 2 = 7 pts, 3 = 5 pts etc.
    """
    score = 0
    for char in musique:
        if char.isdigit():
            pos = int(char)
            if pos == 1:
                score += 10
            elif pos == 2:
                score += 7
            elif pos == 3:
                score += 5
            elif pos <= 5:
                score += 3
    return score

# ==============================
# SCORING ENGINE
# ==============================

def compute_scores(df):

    df["Record_sec"] = df["Record"].apply(convert_record_to_seconds)

    scaler = MinMaxScaler()

    df["Record_score"] = 1 - scaler.fit_transform(df[["Record_sec"]])
    df["Driver_score"] = scaler.fit_transform(df[["Driver %"]])
    df["Trainer_score"] = scaler.fit_transform(df[["Entraineur %"]])

    df["Score Global"] = (
        df["Record_score"] * 0.35 +
        df["Driver_score"] * 0.25 +
        df["Trainer_score"] * 0.20
    )

    df = df.sort_values("Score Global", ascending=False).reset_index(drop=True)

    return df

# ==============================
# PRONOSTIC ENGINE
# ==============================

def generate_trio(top_numbers):
    return list(itertools.combinations(top_numbers[:6], 3))[:10]

def generate_quinte(top_numbers):
    return list(itertools.combinations(top_numbers[:8], 5))[:10]

# ==============================
# UI
# ==============================

st.title("🏇 IA Pronostics Hippiques")
st.subheader("Analyse intelligente à partir de photos statistiques")

st.markdown("## 📤 Télécharger les photos")

uploaded_files = st.file_uploader(
    "📷 Télécharger les photos",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:

    images = []
    for file in uploaded_files:
        image = Image.open(file)
        images.append(image)
        st.image(image, width=250)

    if st.button("🔎 Analyser la course"):

        progress = st.progress(0)
        full_text = ""

        for i, img in enumerate(images):
            text = extract_text_from_image(img)
            full_text += text + "\n"
            progress.progress((i+1)/len(images))

        df = extract_horse_data(full_text)

        if df.empty:
            st.error("Aucune donnée exploitable détectée.")
        else:

            df = compute_scores(df)

            st.success("Analyse terminée !")

            st.markdown("## 📊 Données Structurées")
            st.dataframe(df)

            st.markdown("## 🏆 Classement Probable")
            st.write(df[["Numéro", "Nom", "Score Global"]])

            top3 = df.head(3)
            bases = df.head(2)
            outsiders = df.tail(4)

            st.markdown("## 🎯 Recommandations")

            st.write("### 🥇 Top 3 conseillé")
            st.write(top3[["Numéro", "Nom"]])

            st.write("### 🔒 2 Bases solides")
            st.write(bases[["Numéro", "Nom"]])

            st.write("### 💎 Outsiders intéressants")
            st.write(outsiders[["Numéro", "Nom"]])

            st.markdown("## 🔢 10 Combinaisons Trio")
            trio = generate_trio(df["Numéro"].tolist())
            for t in trio:
                st.write(t)

            st.markdown("## 🔢 10 Combinaisons Quinté")
            quinte = generate_quinte(df["Numéro"].tolist())
            for q in quinte:
                st.write(q)

            st.markdown("## 📊 Graphique des Scores")

            fig, ax = plt.subplots()
            ax.bar(df["Numéro"].astype(str), df["Score Global"])
            ax.set_xlabel("Chevaux")
            ax.set_ylabel("Score")
            st.pyplot(fig)
