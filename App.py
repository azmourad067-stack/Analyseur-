import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageEnhance
import io
import re
import json
import base64
import time
import hashlib
import cv2
from datetime import datetime
from typing import List, Dict, Optional

# OCR & LLM
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

# Configuration Streamlit
st.set_page_config(page_title="🏇 PronoHippique Ultimate", layout="wide")
APP_VERSION = "3.0"

# ============================================
# 1. PRÉTRAITEMENT IMAGE (agressif)
# ============================================
def preprocess_image_for_ocr(pil_img: Image.Image, target_width: int = 1600) -> Image.Image:
    """Redimensionne, améliore le contraste, binarise et redresse."""
    # Redimensionnement si trop petit
    w, h = pil_img.size
    if w < target_width:
        ratio = target_width / w
        new_size = (int(w * ratio), int(h * ratio))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)

    # Conversion OpenCV
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE (contraste local)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Binarisation adaptative
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 10)

    # Nettoyage (médian)
    denoised = cv2.medianBlur(binary, 3)

    # Correction d'inclinaison (deskew)
    coords = np.column_stack(np.where(denoised > 0))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[2]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > 0.5:
            h_img, w_img = denoised.shape
            M = cv2.getRotationMatrix2D((w_img // 2, h_img // 2), angle, 1.0)
            denoised = cv2.warpAffine(denoised, M, (w_img, h_img), borderMode=cv2.BORDER_CONSTANT)

    # Retour en PIL
    return Image.fromarray(denoised)

# ============================================
# 2. PROMPT GEMINI (ULTRA-DÉTAILLÉ)
# ============================================
def gemini_prompt() -> str:
    return """
Tu es un expert en extraction de données hippiques françaises (PMU, Paris-Turf).

L'image contient un tableau de partants avec les colonnes suivantes (l'ordre peut varier légèrement) :
- N° (numéro)
- Cheval (nom)
- Un symbole (corde, ex: ♉, ♌, ♍, ...) – à ignorer
- SA (sexe + âge, ex: F5, H4, M6)
- Poids (nombre, peut être absent)
- Déch. (décharge, souvent "-")
- Jockey (nom du driver)
- Entraîneur
- Musique (ex: 1p1p5p(25)0p)
- Valeur (nombre, parfois absent)
- Rapports probables PMU (cote)
- Genybet (cote)

**Instructions** :
- Retourne UNIQUEMENT un JSON valide, sans aucun texte autour, sans markdown.
- Le JSON doit contenir un champ "chevaux" : liste d'objets avec les clés : "numero", "cheval", "sa", "driver", "entraineur", "musique", "cote_pmu", "cote_genybet".
- Si une valeur est absente, mets null (pas de chaîne vide).
- Ne crée pas de clés supplémentaires.
- La musique doit être préservée telle quelle (ex: "0p1p(25)0p").
- Exemple de sortie :
{
  "chevaux": [
    {"numero": 1, "cheval": "Divide And Rule", "sa": "F5", "driver": "A. Lemaitre", "entraineur": "J.-Pi. Gauvin", "musique": "0p1p(25)0p", "cote_pmu": 16, "cote_genybet": 13.8},
    ...
  ]
}

Si tu ne vois aucun tableau ou aucune donnée exploitable, retourne {"chevaux": []}.
"""

# ============================================
# 3. EXTRACTION AVEC GEMINI (CACHÉE)
# ============================================
@st.cache_data(show_spinner=False, ttl=3600)
def extract_with_gemini(image_bytes: bytes, api_key: str) -> Dict:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        img = Image.open(io.BytesIO(image_bytes))
        img_proc = preprocess_image_for_ocr(img)
        response = model.generate_content(
            [gemini_prompt(), img_proc],
            generation_config={"temperature": 0.0, "max_output_tokens": 8192}
        )
        text = response.text.strip()
        # Nettoyer le markdown
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        data = json.loads(text)
        if "chevaux" in data:
            return {"success": True, "data": data, "raw_text": text}
        else:
            return {"success": False, "error": "Format JSON invalide", "raw_text": text}
    except Exception as e:
        return {"success": False, "error": str(e), "raw_text": ""}

# ============================================
# 4. EXTRACTION AVEC EASYOCR + REGEX (FALLBACK)
# ============================================
def extract_with_easyocr(image_bytes: bytes) -> Dict:
    if not HAS_EASYOCR:
        return {"success": False, "error": "EasyOCR non installé"}
    try:
        reader = easyocr.Reader(['fr', 'en'], gpu=False, verbose=False)
        img = Image.open(io.BytesIO(image_bytes))
        img_proc = preprocess_image_for_ocr(img)
        img_np = np.array(img_proc)
        result = reader.readtext(img_np, detail=0, paragraph=False)
        full_text = " ".join(result)
        # Obtenir aussi la mise en page (lignes)
        result_bbox = reader.readtext(img_np, detail=1, paragraph=False)
        lines = {}
        for (bbox, text, conf) in result_bbox:
            if conf < 0.25:
                continue
            y_center = (bbox[0][1] + bbox[2][1]) // 2
            bucket = y_center // 20
            lines.setdefault(bucket, []).append((bbox[0][0], text))
        ordered_lines = []
        for y in sorted(lines.keys()):
            line = " ".join(t for _, t in sorted(lines[y], key=lambda x: x[0]))
            ordered_lines.append(line)
        layout_text = "\n".join(ordered_lines)

        # Parsing avec regex spécifiques au tableau PMU
        horses = parse_pmu_table(layout_text)
        return {"success": True, "data": {"chevaux": horses}, "raw_text": layout_text}
    except Exception as e:
        return {"success": False, "error": str(e), "raw_text": ""}

def parse_pmu_table(text: str) -> List[Dict]:
    """Extrait les chevaux à partir du texte brut (lignes) avec des regex adaptées."""
    horses = []
    lines = text.split('\n')
    # Pattern principal pour une ligne typique
    # Exemple : "1 Divide And Rule ♉ 8 F5 60 - A. Lemaitre J.-Pi. Gauvin 0p1p(25)0p 42 16 13,8"
    pattern = re.compile(
        r'^(\d+)\s+'                     # numéro
        r'([A-Za-z][A-Za-z\s\-\.]+?)\s+' # nom
        r'[♉♌♍♎♏♐♑♒♓♔♕♖♗♘♙♚]?\s*'       # symbole corde (optionnel)
        r'(?:\d+\s+)?'                   # un nombre (corde?)
        r'([FMH]\d+)?\s*'                # SA (optionnel)
        r'(?:\d+(?:\.\d+)?\s+)?'         # poids (optionnel)
        r'(?:[-–]\s+)?'                  # décharge (optionnel)
        r'([A-Z][a-z]\.?\s+[A-Z][a-z]+)?\s*'  # jockey (optionnel)
        r'([A-Z][a-z\.\-]+\s+[A-Z][a-z\.\-]+)?\s*' # entraineur (optionnel)
        r'([0-9pP\(\)a-zA-Z]+)?\s*'      # musique (optionnel)
        r'(?:(\d+(?:[.,]\d+)?)\s+)?'     # cote PMU (optionnel)
        r'(\d+(?:[.,]\d+)?)\s*$'         # cote Genybet (souvent présente)
    )
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = pattern.search(line)
        if m:
            num = int(m.group(1))
            name = m.group(2).strip()
            sa = m.group(3) if m.group(3) else ""
            driver = m.group(4) if m.group(4) else ""
            entraineur = m.group(5) if m.group(5) else ""
            musique = m.group(6) if m.group(6) else ""
            cote_pmu_str = m.group(7) if m.group(7) else "0"
            cote_geny_str = m.group(8) if m.group(8) else "0"
            try:
                cote_pmu = float(cote_pmu_str.replace(',', '.'))
            except:
                cote_pmu = 0.0
            try:
                cote_geny = float(cote_geny_str.replace(',', '.'))
            except:
                cote_geny = 0.0
            horses.append({
                "numero": num,
                "cheval": name,
                "sa": sa,
                "driver": driver,
                "entraineur": entraineur,
                "musique": musique,
                "cote_pmu": cote_pmu,
                "cote_genybet": cote_geny
            })
    # Si aucun cheval trouvé, essayer pattern plus simple (numéro, nom, deux cotes)
    if not horses:
        simple = re.compile(r'^(\d+)\s+([A-Za-z][A-Za-z\s\-]+?)\s+.*?(\d+(?:[.,]\d+)?)\s+(\d+(?:[.,]\d+)?)$')
        for line in lines:
            m = simple.search(line)
            if m:
                num = int(m.group(1))
                name = m.group(2).strip()
                c1 = m.group(3)
                c2 = m.group(4)
                try:
                    cote_pmu = float(c1.replace(',', '.'))
                except:
                    cote_pmu = 0.0
                try:
                    cote_geny = float(c2.replace(',', '.'))
                except:
                    cote_geny = 0.0
                horses.append({
                    "numero": num,
                    "cheval": name,
                    "sa": "",
                    "driver": "",
                    "entraineur": "",
                    "musique": "",
                    "cote_pmu": cote_pmu,
                    "cote_genybet": cote_geny
                })
    # Déduplication par numéro
    unique = {}
    for h in horses:
        if h["numero"] not in unique:
            unique[h["numero"]] = h
    return list(unique.values())

# ============================================
# 5. FUSION DES EXTRACTIONS (MULTIPLES IMAGES)
# ============================================
def merge_horses(horses_list: List[List[Dict]]) -> List[Dict]:
    merged = {}
    for horses in horses_list:
        for h in horses:
            num = h["numero"]
            if num not in merged:
                merged[num] = h.copy()
            else:
                # Fusionne les champs manquants
                for k, v in h.items():
                    if v and not merged[num].get(k):
                        merged[num][k] = v
    return list(merged.values())

# ============================================
# 6. SCORING (simple mais efficace)
# ============================================
def decode_musique(musique: str) -> List[int]:
    if not musique:
        return []
    musique = re.sub(r'\([^)]+\)', '', musique)  # enlève (25)
    tokens = re.findall(r'[0-9]+|[A-Za-z]', musique)
    scores = []
    for t in tokens:
        if t.isdigit():
            p = int(t)
            if p == 1: scores.append(10)
            elif p == 2: scores.append(7)
            elif p == 3: scores.append(5)
            elif p <= 5: scores.append(3)
            else: scores.append(1)
        else:
            scores.append(0)
    return scores

def musique_score(musique: str) -> float:
    scores = decode_musique(musique)
    if not scores:
        return 5.0
    recent = scores[-5:]
    return sum(recent) / len(recent) if recent else 5.0

def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["score_musique"] = df["musique"].apply(musique_score)
    # Inverse de la cote PMU (plus petite cote = meilleur score)
    df["score_cote"] = df["cote_pmu"].apply(lambda x: max(0, 10 - (x / 4)) if x > 0 else 5)
    df["score_global"] = (df["score_musique"] * 0.6 + df["score_cote"] * 0.4).round(2)
    df["rang"] = df["score_global"].rank(ascending=False, method='min').astype(int)
    total = df["score_global"].sum()
    df["proba"] = (df["score_global"] / total * 100).round(1) if total > 0 else 0
    return df.sort_values("score_global", ascending=False).reset_index(drop=True)

# ============================================
# 7. INTERFACE STREAMLIT
# ============================================
st.markdown("""
<style>
.big-title {
    background: linear-gradient(135deg, #0d3320 0%, #1a6b3c 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 2rem;
}
.stButton > button {
    background: #1a6b3c;
    color: white;
    border-radius: 10px;
    width: 100%;
    font-weight: bold;
}
</style>
<div class="big-title">
    <h1>🏇 PronoHippique Ultimate v{APP_VERSION}</h1>
    <p>Gemini Vision + EasyOCR fallback • Extraction robuste des tableaux PMU</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    engine_choice = st.radio(
        "Moteur d'extraction",
        ["🤖 Gemini (recommandé)", "📷 EasyOCR (local)"],
        help="Gemini nécessite une clé API, EasyOCR fonctionne hors ligne."
    )
    gemini_key = ""
    if "Gemini" in engine_choice:
        gemini_key = st.text_input("Clé API Google Gemini", type="password", help="Obtenez une clé sur https://aistudio.google.com/")
        if not gemini_key:
            st.warning("⚠️ Clé Gemini manquante, bascule automatique sur EasyOCR.")
    debug = st.checkbox("🔍 Afficher les textes bruts (debug)", value=True)
    st.divider()
    st.caption("Les images sont prétraitées (CLAHE, binarisation, redressement).")

# Upload
uploaded_files = st.file_uploader(
    "📸 Téléchargez une ou plusieurs captures du tableau",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    # Aperçu
    st.subheader("🖼️ Images chargées")
    cols = st.columns(min(len(uploaded_files), 4))
    for i, f in enumerate(uploaded_files):
        with cols[i % 4]:
            st.image(Image.open(f), caption=f.name, use_container_width=True)
    
    if st.button("🚀 Lancer l'analyse ultime", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        all_horses = []
        raw_texts = []
        
        for idx, file in enumerate(uploaded_files):
            status_text.text(f"Traitement de {file.name}... ({idx+1}/{len(uploaded_files)})")
            progress_bar.progress((idx) / len(uploaded_files))
            img_bytes = file.getvalue()
            
            # Choix du moteur
            if "Gemini" in engine_choice and gemini_key:
                result = extract_with_gemini(img_bytes, gemini_key)
            else:
                result = extract_with_easyocr(img_bytes)
            
            if result["success"]:
                horses = result["data"].get("chevaux", [])
                all_horses.extend(horses)
                raw_texts.append(f"--- {file.name} ---\n{result.get('raw_text', '')}\n")
            else:
                raw_texts.append(f"--- {file.name} (ERREUR) ---\n{result.get('error', '')}\n")
        
        progress_bar.progress(0.9)
        status_text.text("Fusion des données et calcul des scores...")
        
        # Fusion
        merged_horses = merge_horses([all_horses])   # all_horses est déjà une liste de dicts
        df_raw = pd.DataFrame(merged_horses)
        
        # Debug
        if debug:
            st.subheader("📄 Texte OCR / Réponse Gemini (brut)")
            st.text_area("", "\n".join(raw_texts), height=300)
        
        if df_raw.empty:
            st.error("❌ Aucun cheval extrait. Vérifiez la qualité des images ou le moteur utilisé.")
            if "Gemini" in engine_choice and not gemini_key:
                st.info("💡 Pas de clé Gemini fournie. Utilisez EasyOCR ou ajoutez une clé.")
        else:
            df_scored = compute_scores(df_raw)
            st.success(f"✅ {len(df_scored)} chevaux extraits avec succès !")
            
            # Affichage du classement
            st.subheader("🏆 Classement pronostiqué")
            display_cols = ["rang", "numero", "cheval", "score_global", "proba", "cote_pmu", "musique"]
            st.dataframe(df_scored[display_cols], use_container_width=True)
            
            # Top 3
            st.subheader("🥇 Podium IA")
            top3 = df_scored.head(3)
            cols = st.columns(3)
            for i, (_, row) in enumerate(top3.iterrows()):
                with cols[i]:
                    st.metric(
                        f"#{row['numero']} {row['cheval']}",
                        f"{row['score_global']}/10",
                        f"Proba {row['proba']}%"
                    )
            
            # Graphique
            fig = px.bar(
                df_scored, x="cheval", y="score_global", color="score_global",
                color_continuous_scale="Viridis", title="Scores globaux par cheval"
            )
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Export CSV
            csv = df_scored.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Télécharger le pronostic (CSV)",
                csv,
                f"pronostic_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )
        
        progress_bar.progress(1.0)
        status_text.text("Analyse terminée.")
else:
    st.info("👈 Téléchargez des captures d'écran du tableau pour commencer.")
