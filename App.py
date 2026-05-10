import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import re
import base64
import time
import cv2
import easyocr
from datetime import datetime

# Configuration de la page
st.set_page_config(page_title="PronoHippique AI v2.3", layout="wide")

# -------------------------------
# 1. PRÉTRAITEMENT IMAGE (amélioré)
# -------------------------------
def preprocess_image(img):
    """Améliore le contraste et nettoie l'image pour l'OCR."""
    # Convertir PIL en OpenCV
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # Passage en niveaux de gris
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # CLAHE (contraste local)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    # Binarisation adaptative
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 8)
    # Réduction du bruit
    denoised = cv2.medianBlur(binary, 3)
    # Redressement si besoin (détection d'angle simplifiée)
    coords = np.column_stack(np.where(denoised > 0))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[2]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > 0.5:
            h, w = denoised.shape
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            denoised = cv2.warpAffine(denoised, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
    return Image.fromarray(denoised)

# -------------------------------
# 2. OCR AVEC EASYOCR (local, pas de clé API nécessaire)
# -------------------------------
@st.cache_resource
def get_reader():
    return easyocr.Reader(['fr', 'en'], gpu=False, verbose=False)

def extract_text_from_image(image):
    """Retourne le texte brut extrait par EasyOCR."""
    reader = get_reader()
    img_array = np.array(preprocess_image(image))
    results = reader.readtext(img_array, detail=0, paragraph=False)
    # Joint toutes les lignes détectées
    full_text = " ".join(results)
    # Alternative : on récupère aussi le texte avec mise en page (important pour les tableaux)
    # On refait une lecture avec les positions pour reconstruire les lignes
    results_with_bbox = reader.readtext(img_array, detail=1, paragraph=False)
    lines = {}
    for (bbox, text, conf) in results_with_bbox:
        if conf < 0.3:
            continue
        y_center = (bbox[0][1] + bbox[2][1]) // 2
        bucket = y_center // 20
        lines.setdefault(bucket, []).append((bbox[0][0], text))
    ordered_lines = []
    for y in sorted(lines.keys()):
        line = " ".join([t for _, t in sorted(lines[y], key=lambda x: x[0])])
        ordered_lines.append(line)
    layout_text = "\n".join(ordered_lines)
    return layout_text if len(layout_text) > len(full_text) else full_text

# -------------------------------
# 3. EXTRACTION DES DONNÉES PAR REGEX (spécial tableau PMU)
# -------------------------------
def parse_horses_from_text(text):
    """
    Extrait les chevaux à partir du texte OCR.
    Exemple de ligne typique :
    1  Divide And Rule   ♉  8  F5  60  -  A. Lemaitre  J.-Pi. Gauvin  0p1p(25)0p  42  16  13,8
    """
    horses = []
    # Nettoyer les caractères parasites
    text = text.replace('|', ' ').replace('\t', ' ')
    lines = text.split('\n')
    
    # Patterns
    # On cherche une ligne qui commence par un nombre (numéro) et contient ensuite un nom (lettres + espaces)
    # et plus loin deux nombres (cotes) ou un nombre et une cote
    pattern = re.compile(
        r'^(\d+)\s+'                     # numéro
        r'([A-Za-z][A-Za-z\s\-\.]+?)\s+' # nom (jusqu'au prochain espace ou symbole)
        r'(?:[♉♌♍♎♏♐♑♒♓♔♕♖♗♘♙♚]\s+)?'  # symbole corde (optionnel)
        r'(?:\d+\s+)?'                    # un nombre (corde? age?) optionnel
        r'([FMH]\d+\s+)?'                # SA (optionnel)
        r'(?:\d+(?:\.\d+)?\s+)?'         # poids (optionnel)
        r'(?:[-\d]+\s+)?'                # décharge (optionnel)
        r'(?:[A-Z][a-z]\.?\s+[A-Z][a-z]+)?\s*'  # jockey (optionnel)
        r'(?:[A-Z][a-z\.\-]+\s+[A-Z][a-z\.\-]+)?\s*' # entraineur (optionnel)
        r'(?:[0-9pP\(\)a-zA-Z]+\s+)?'    # musique (optionnel)
        r'(?:(\d+(?:[.,]\d+)?)\s+)?'     # cote 1 (PMU)
        r'(\d+(?:[.,]\d+)?)\s*$'         # cote 2 (Genybet) - obligatoire
    )
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Essai du pattern principal
        m = pattern.search(line)
        if m:
            num = int(m.group(1))
            name = m.group(2).strip()
            sa = m.group(3).strip() if m.group(3) else ""
            cote1 = m.group(4) if m.group(4) else "0"
            cote2 = m.group(5) if m.group(5) else "0"
            # Nettoyage des cotes
            try:
                cote_pmu = float(cote1.replace(',', '.'))
            except:
                cote_pmu = 0.0
            try:
                cote_geny = float(cote2.replace(',', '.'))
            except:
                cote_geny = 0.0
            # Chercher la musique dans la ligne (entre entraineur et cotes)
            # Méthode plus simple : on prend tout ce qui reste après l'entraineur jusqu'aux cotes
            # On va plutôt extraire la musique par un autre pattern sur la ligne entière
            music_match = re.search(r'([0-9pP\(\)a-zA-Z]+)\s+\d+(?:[.,]\d+)?\s+\d+(?:[.,]\d+)?$', line)
            musique = music_match.group(1) if music_match else ""
            
            horses.append({
                "numero": num,
                "cheval": name,
                "sa": sa,
                "musique": musique,
                "cote_pmu": cote_pmu,
                "cote_genybet": cote_geny,
                "driver": "",   # on ne le capture pas proprement ici, mais on peut le faire plus tard
                "entraineur": ""
            })
            continue
        
        # Fallback : pattern plus simple (numéro + nom + deux nombres à la fin)
        simple = re.search(r'^(\d+)\s+([A-Za-z][A-Za-z\s\-]+?)\s+.*?(\d+(?:[.,]\d+)?)\s+(\d+(?:[.,]\d+)?)$', line)
        if simple:
            num = int(simple.group(1))
            name = simple.group(2).strip()
            cote1 = simple.group(3)
            cote2 = simple.group(4)
            try:
                cote_pmu = float(cote1.replace(',', '.'))
            except:
                cote_pmu = 0.0
            try:
                cote_geny = float(cote2.replace(',', '.'))
            except:
                cote_geny = 0.0
            horses.append({
                "numero": num,
                "cheval": name,
                "sa": "",
                "musique": "",
                "cote_pmu": cote_pmu,
                "cote_genybet": cote_geny,
                "driver": "",
                "entraineur": ""
            })
    
    # Déduplication par numéro
    unique = {}
    for h in horses:
        if h["numero"] not in unique:
            unique[h["numero"]] = h
    return list(unique.values())

# -------------------------------
# 4. SCORING SIMPLE
# -------------------------------
def decode_musique(musique):
    if not musique:
        return []
    musique = re.sub(r'\([^)]+\)', '', musique)
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

def musique_score(musique):
    scores = decode_musique(musique)
    if not scores:
        return 5.0
    recent = scores[-5:]
    return sum(recent) / len(recent) if recent else 5.0

def compute_scores(df):
    if df.empty:
        return df
    df = df.copy()
    df["score_musique"] = df["musique"].apply(musique_score)
    # Cote inversée (plus la cote est faible, meilleur est le score)
    df["score_cote"] = df["cote_pmu"].apply(lambda x: max(0, 10 - (x / 4)) if x > 0 else 5)
    df["score_global"] = (df["score_musique"] * 0.6 + df["score_cote"] * 0.4).round(2)
    df["rang"] = df["score_global"].rank(ascending=False, method='min').astype(int)
    # Probabilité simple
    total = df["score_global"].sum()
    df["proba"] = (df["score_global"] / total * 100).round(1) if total > 0 else 0
    return df.sort_values("score_global", ascending=False)

# -------------------------------
# 5. INTERFACE STREAMLIT
# -------------------------------
st.markdown("""
<style>
.big-title { background: #1a6b3c; color: white; padding: 1.5rem; border-radius: 15px; text-align: center; }
.stButton>button { background: #1a6b3c; color: white; border-radius: 8px; width: 100%; }
</style>
<div class="big-title">
    <h1>🏇 PronoHippique AI v2.3</h1>
    <p>Extraction directe par OCR + Regex – Pas de JSON, que du texte</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("🔧 Configuration")
    show_raw_text = st.checkbox("📄 Afficher le texte OCR brut", value=True)
    st.info("Aucune clé API nécessaire. EasyOCR fonctionne en local.")

uploaded_files = st.file_uploader("📸 Téléchargez les captures d'écran du tableau", 
                                  type=["png", "jpg", "jpeg"], 
                                  accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(min(len(uploaded_files), 4))
    for i, f in enumerate(uploaded_files):
        with cols[i % 4]:
            st.image(Image.open(f), caption=f.name, use_container_width=True)
    
   if st.button("🚀 Lancer l'analyse", use_container_width=True):
    all_horses = []
    raw_texts = []
    progress = st.progress(0)
    for i, f in enumerate(uploaded_files):
        progress.progress((i+1) / len(uploaded_files))   # ✅ correction ici
        img = Image.open(f).convert("RGB")
        with st.spinner(f"OCR sur {f.name}..."):
            text = extract_text_from_image(img)
            raw_texts.append(f"--- {f.name} ---\n{text}\n")
            horses = parse_horses_from_text(text)
            all_horses.extend(horses)
        # Fusion et déduplication
        unique_horses = {}
        for h in all_horses:
            num = h["numero"]
            if num not in unique_horses:
                unique_horses[num] = h
            else:
                # Fusionne les champs manquants
                for k, v in h.items():
                    if v and not unique_horses[num].get(k):
                        unique_horses[num][k] = v
        
        df_raw = pd.DataFrame(list(unique_horses.values()))
        
        if show_raw_text:
            st.subheader("🔍 Texte OCR brut (pour diagnostic)")
            st.text_area("", "\n".join(raw_texts), height=300)
        
        if df_raw.empty:
            st.error("❌ Aucun cheval n'a pu être extrait. Le texte OCR ci-dessus vous permet de vérifier ce que l'ordinateur lit réellement.")
            st.info("💡 Si le texte ne contient pas les numéros et noms comme attendu, cela signifie que la qualité de l'image est insuffisante (floue, trop petite, mal cadrée). Essayez avec une image plus nette ou redimensionnez-la en plus grand avant téléchargement.")
        else:
            df = compute_scores(df_raw)
            st.success(f"✅ {len(df)} chevaux extraits avec succès !")
            
            st.subheader("🏆 Classement pronostiqué")
            st.dataframe(df[["rang", "numero", "cheval", "score_global", "proba", "cote_pmu", "musique"]], use_container_width=True)
            
            # Top 3
            top3 = df.head(3)
            cols = st.columns(3)
            for i, (_, row) in enumerate(top3.iterrows()):
                with cols[i]:
                    st.metric(f"#{row['numero']} {row['cheval']}", f"{row['score_global']}/10", f"Proba {row['proba']}%")
            
            # Graphique
            fig = px.bar(df, x="cheval", y="score_global", color="score_global", color_continuous_scale="Viridis", title="Scores globaux")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Export CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Exporter en CSV", csv, f"pronostic_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
