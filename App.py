# ══════════════════════════════════════════════════════════════════
#  PRONOHIPPIQUE AI v2.2 – CORRECTION POUR TABLEAUX PMU SPECIFIQUES
#  - Extraction robuste même sans JSON valide
#  - Mapping automatique des colonnes (Rapports PMU, Genybet, etc.)
#  - Mode debug pour voir le texte OCR brut
# ══════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import re
import json
import base64
import time
import hashlib
import cv2
from datetime import datetime
from typing import Optional, List, Dict, Tuple

# OCR APIs
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

# ══════════════════════════════════════════════════════════════════
#  CONFIG PAGE
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="🏇 PronoHippique AI v2.2", layout="wide")

APP_VERSION = "2.2.0"
MAX_IMAGES = 8

# ══════════════════════════════════════════════════════════════════
#  PREPROCESSING IMAGE (inchangé, mais conservé)
# ══════════════════════════════════════════════════════════════════
def _pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def _cv2_to_pil(cv2_img):
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

def _detect_table_region(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = cv_img.shape[:2]
        return 0, 0, w, h
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    padding = max(10, int(w * 0.02))
    return max(0, x-padding), max(0, y-padding), min(cv_img.shape[1], x+w+padding), min(cv_img.shape[0], y+h+padding)

def _deskew_image(gray_img):
    try:
        coords = np.column_stack(np.where(gray_img > 127))
        if len(coords) < 100:
            return gray_img
        angle = cv2.minAreaRect(coords)[2]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > 0.5:
            h, w = gray_img.shape
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            gray_img = cv2.warpAffine(gray_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return gray_img
    except:
        return gray_img

def _enhance_image_for_ocr(pil_img, aggressive=False):
    w, h = pil_img.size
    if max(w, h) < 1200:
        ratio = 1500 / max(w, h)
        pil_img = pil_img.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)
    cv_img = _pil_to_cv2(pil_img)
    x1, y1, x2, y2 = _detect_table_region(cv_img)
    cv_img = cv_img[y1:y2, x1:x2]
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    if aggressive:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    gray = _deskew_image(gray)
    return _cv2_to_pil(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

def _hash_image(image):
    buf = io.BytesIO()
    image.resize((128, 128)).save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()

# ══════════════════════════════════════════════════════════════════
#  PROMPT AMÉLIORÉ POUR LE FORMAT PMU AVEC RAPPORTS PROBABLES
# ══════════════════════════════════════════════════════════════════
def _build_extraction_prompt_advanced():
    return """Tu es un EXPERT en extraction de données hippiques françaises (PMU).

IDENTIFIE le tableau type "partants". Les colonnes observées sont généralement :

N° | Cheval | C | SA | Poids | Déch. | Jockey | Entraîneur | Musique | Valeur | Rapports probables PMU | Genybet

Il peut y avoir aussi des symboles comme ♉ ♌ etc. (ce sont des indicateurs de corde/couleur, à ignorer).

EXTRACTION : pour chaque cheval, retourne un JSON avec les champs suivants :

{
  "table_type": "partants",
  "nb_partants": 16,
  "chevaux": [
    {
      "numero": 1,
      "cheval": "Divide And Rule",
      "sa": "F5",            // Sexe (F/H/P) + Age (ex: F5, H4, M5)
      "poids": 60,           // (optionnel)
      "decharge": "-",       // (optionnel)
      "driver": "A. Lemaitre",
      "entraineur": "J.-Pi. Gauvin",
      "musique": "0p1p(25)0p",
      "valeur": 42,          // cote indicatrice (peut être ignorée)
      "cote_pmu": 16,        // Rapports probables PMU (première cote)
      "cote_genybet": 13.8   // Genybet (deuxième cote)
    }
  ]
}

⚠️ RÈGLES :
- Le champ "C" (corde) est un symbole à ignorer.
- "SA" = sexe+âge (ex: F5, M6, H7). Extraire tel quel.
- "Jockey" → driver
- "Entraîneur" → entraineur
- "Rapports probables PMU" → cote_pmu (peut être avec une virgule ou un point)
- "Genybet" → cote_genybet
- La musique peut contenir des lettres p, D, A, M, (25) etc. Garder la chaîne brute.
- Si une colonne est absente, mets null.

Retourne UNIQUEMENT le JSON valide, sans markdown, sans explication.
"""

# ══════════════════════════════════════════════════════════════════
#  PARSING TEXTE BRUT (FALLBACK LORSQUE LE JSON ÉCHOUE)
# ══════════════════════════════════════════════════════════════════
def extract_table_from_raw_text(text: str) -> List[Dict]:
    """Parse ligne par ligne un tableau texte issu de l'OCR."""
    lines = text.splitlines()
    chevaux = []
    # Patterns pour identifier une ligne de cheval
    # Exemple : "1  Divide And Rule   ♉  8  F5  60  -  A. Lemaitre  J.-Pi. Gauvin  0p1p(25)0p  42  16  13,8"
    # On cherche un numéro au début, puis un nom, puis des données séparées par des espaces/tabulations
    pattern = re.compile(r'^(\d+)\s+([A-Za-z][A-Za-z\s\-]+?)\s+(?:[♉♌♍♎♏♐♑♒♓♔♕♖♗♘♙♚]\s+)?(\d+)\s+([FMH]\d+)\s+(\d+(?:\.\d+)?)\s+(\-|\d+)?\s+([A-Z][a-z]\.?\s?[A-Z][a-z]+|[A-Z]\.\s?[A-Z][a-z]+)?\s+([A-Z][a-z\.\-]+(?:\s+[A-Z][a-z\.\-]+)?)\s+([0-9pP\(\)a-zA-Z]+)\s+(\d+)\s+(\d+(?:\.\d+)?)\s+(\d+(?:[.,]\d+)?)', re.UNICODE)
    
    # Plus simple : on cherche les lignes qui contiennent un nombre (le numéro) et au moins une cote (deux nombres après)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Essai de découpage par tabulations ou espaces multiples
        parts = re.split(r'\s{2,}|\t', line)
        if len(parts) < 5:
            continue
        # Le premier élément doit être un nombre (le numéro)
        if not parts[0].isdigit():
            continue
        try:
            numero = int(parts[0])
        except:
            continue
        
        # Recherche du nom du cheval (entre numéro et la prochaine donnée qui est souvent un chiffre ou un symbole)
        # Méthode robuste : on parcourt les parties pour trouver le nom
        name = ""
        idx = 1
        # Le nom peut contenir des espaces, on le reconstruit jusqu'à rencontrer un nombre ou un symbole de corde
        while idx < len(parts) and not re.match(r'^\d+$|^[♉♌♍♎♏♐♑♒♓♔♕♖♗♘♙♚]$', parts[idx]):
            name += parts[idx] + " "
            idx += 1
        name = name.strip()
        if not name:
            continue
        
        # On cherche maintenant la SA (F5, H4, etc.)
        # Le prochain token après le nom est souvent le symbole de corde, on saute
        while idx < len(parts) and re.match(r'^[♉♌♍♎♏♐♑♒♓♔♕♖♗♘♙♚]$', parts[idx]):
            idx += 1
        if idx >= len(parts):
            continue
        sa = parts[idx] if re.match(r'^[FMH]\d+$', parts[idx]) else ""
        idx += 1
        
        # On avance jusqu'à trouver le driver (nom avec point ou deux mots)
        # Mais on va plutôt chercher les cotes à la fin
        # On récupère les 3 dernières valeurs : musique, cote_pmu, cote_genybet ? Pas toujours fiable.
        # On va simplement stocker tout ce qu'on peut
        
        cheval = {
            "numero": numero,
            "cheval": name,
            "sa": sa,
            "driver": "",
            "entraineur": "",
            "musique": "",
            "cote_pmu": 0,
            "cote_genybet": 0
        }
        
        # Recherche du driver (souvent initiale + point + nom, ex: "A. Lemaitre")
        driver_pattern = re.compile(r'[A-Z]\.\s+[A-Z][a-z]+')
        for part in parts:
            if driver_pattern.search(part):
                cheval["driver"] = part
                break
        
        # Recherche entraîneur (souvent avec un tiret ou deux mots)
        ent_pattern = re.compile(r'[A-Z][a-z]+[\-\.]?[A-Z][a-z]+')
        for part in parts:
            if ent_pattern.search(part) and part != cheval["driver"]:
                cheval["entraineur"] = part
                break
        
        # Recherche musique (contient des lettres, parenthèses, chiffres)
        musique_pattern = re.compile(r'[\d\(\)pPaADM]+')
        for part in parts:
            if musique_pattern.search(part) and len(part) > 2:
                cheval["musique"] = part
                break
        
        # Recherche cotes (deux nombres décimaux à la fin)
        cotes = []
        for part in parts[::-1]:
            if re.match(r'^\d+(?:[.,]\d+)?$', part):
                cotes.append(float(part.replace(',', '.')))
                if len(cotes) == 2:
                    break
        if cotes:
            cheval["cote_genybet"] = cotes[0]
            if len(cotes) > 1:
                cheval["cote_pmu"] = cotes[1]
        
        if cheval["cheval"]:
            chevaux.append(cheval)
    
    return chevaux

# ══════════════════════════════════════════════════════════════════
#  PARSE JSON ROBUSTE
# ══════════════════════════════════════════════════════════════════
def _parse_json_response_robust(raw_text):
    if not raw_text:
        return {}
    clean = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
    clean = re.sub(r"```$", "", clean).strip()
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    json_str = clean[start:end]
    json_str = re.sub(r':\s*(\d+),(\d+)', r':\1.\2', json_str)
    json_str = re.sub(r',\s*}', '}', json_str)
    try:
        return json.loads(json_str)
    except:
        # Tentative avec guillemets simples
        try:
            json_str2 = re.sub(r"'", '"', json_str)
            return json.loads(json_str2)
        except:
            return {}

# ══════════════════════════════════════════════════════════════════
#  OCR MULTI-MOTEURS
# ══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=3600)
def _cached_extract_gemini(img_hash, img_bytes, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        img = Image.open(io.BytesIO(img_bytes))
        img_proc = _enhance_image_for_ocr(img, aggressive=True)
        response = model.generate_content(
            [_build_extraction_prompt_advanced(), img_proc],
            generation_config={"temperature": 0.1, "max_output_tokens": 8192}
        )
        parsed = _parse_json_response_robust(response.text)
        if not parsed.get("chevaux"):
            # Si le JSON est vide, on tente d'extraire du texte brut
            parsed["raw_text"] = response.text
        parsed["ocr_engine"] = "Gemini"
        return parsed
    except Exception as e:
        return {"error": str(e), "ocr_engine": "Gemini"}

def extract_with_gemini(image, api_key):
    if not api_key:
        return {}
    img_hash = _hash_image(image)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return _cached_extract_gemini(img_hash, buf.getvalue(), api_key)

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_extract_openai(img_hash, img_bytes, api_key):
    try:
        client = OpenAI(api_key=api_key)
        img = Image.open(io.BytesIO(img_bytes))
        img_proc = _enhance_image_for_ocr(img, aggressive=False)
        buf = io.BytesIO()
        img_proc.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_extraction_prompt_advanced()},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }],
            max_tokens=8192,
            temperature=0.1
        )
        parsed = _parse_json_response_robust(response.choices[0].message.content)
        if not parsed.get("chevaux"):
            parsed["raw_text"] = response.choices[0].message.content
        parsed["ocr_engine"] = "OpenAI"
        return parsed
    except Exception as e:
        return {"error": str(e), "ocr_engine": "OpenAI"}

def extract_with_openai(image, api_key):
    if not api_key:
        return {}
    img_hash = _hash_image(image)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return _cached_extract_openai(img_hash, buf.getvalue(), api_key)

def extract_with_easyocr(image):
    if not HAS_EASYOCR:
        return {}
    try:
        import easyocr
        reader = easyocr.Reader(['fr', 'en'], gpu=False, verbose=False)
        img_proc = _enhance_image_for_ocr(image, aggressive=True)
        img_array = np.array(img_proc)
        results = reader.readtext(img_array, detail=1)
        lines_by_y = {}
        for (bbox, text, conf) in results:
            if conf < 0.25:
                continue
            y = int((bbox[0][1] + bbox[2][1]) // 2)
            y_bucket = (y // 15) * 15
            lines_by_y.setdefault(y_bucket, []).append((bbox[0][0], text))
        lines = []
        for y in sorted(lines_by_y):
            items = sorted(lines_by_y[y], key=lambda x: x[0])
            lines.append(" ".join(t for _, t in items))
        full_text = "\n".join(lines)
        # Extraction par regex
        chevaux = extract_table_from_raw_text(full_text)
        return {"chevaux": chevaux, "ocr_engine": "EasyOCR", "raw_text": full_text}
    except Exception as e:
        return {"error": str(e), "ocr_engine": "EasyOCR"}

def extract_data_from_image(image, gemini_key, openai_key, preferred="auto"):
    if preferred == "gemini" and gemini_key:
        res = extract_with_gemini(image, gemini_key)
        if res.get("chevaux"):
            return res
    if preferred == "openai" and openai_key:
        res = extract_with_openai(image, openai_key)
        if res.get("chevaux"):
            return res
    # Fallback EasyOCR
    res = extract_with_easyocr(image)
    if res.get("chevaux"):
        return res
    # Dernier recours : tenter de parser le raw_text s'il existe
    if res.get("raw_text"):
        chevaux = extract_table_from_raw_text(res["raw_text"])
        if chevaux:
            res["chevaux"] = chevaux
            return res
    return {"chevaux": [], "error": "Aucune donnée extraite"}

def merge_extracted_data(extractions):
    all_chevaux = {}
    for ext in extractions:
        chevaux = ext.get("chevaux", [])
        for h in chevaux:
            if not h.get("numero") or not h.get("cheval"):
                continue
            num = int(h["numero"])
            if num not in all_chevaux:
                all_chevaux[num] = {}
            for k, v in h.items():
                if v and k != "numero":
                    all_chevaux[num][k] = v
    merged = list(all_chevaux.values())
    return {"chevaux": merged, "nb_partants": len(merged)}

# ══════════════════════════════════════════════════════════════════
#  DATA CLEANER (adapte au mapping des colonnes)
# ══════════════════════════════════════════════════════════════════
def clean_horse_data(chevaux_raw):
    if not chevaux_raw:
        return pd.DataFrame()
    cleaned = []
    for h in chevaux_raw:
        if not h.get("numero") or not h.get("cheval"):
            continue
        # Mapping des champs possibles
        driver = h.get("driver") or h.get("jockey") or ""
        entraineur = h.get("entraineur") or h.get("entraîneur") or ""
        cote_pmu = h.get("cote_pmu") or h.get("rapports_pmu") or h.get("Rapports probables PMU") or 0
        cote_genybet = h.get("cote_genybet") or h.get("genybet") or 0
        musique = h.get("musique") or ""
        sa = h.get("sa") or ""
        # Conversion des cotes
        try:
            cote_pmu = float(str(cote_pmu).replace(',', '.'))
        except:
            cote_pmu = 0
        try:
            cote_genybet = float(str(cote_genybet).replace(',', '.'))
        except:
            cote_genybet = 0
        
        horse = {
            "numero": int(h["numero"]),
            "cheval": str(h["cheval"]).strip(),
            "sa": sa,
            "sexe": sa[0] if sa else "",
            "age": int(re.search(r'\d+', sa).group()) if re.search(r'\d+', sa) else 0,
            "driver": driver,
            "entraineur": entraineur,
            "musique": musique,
            "cote_pmu": cote_pmu,
            "cote_genybet": cote_genybet,
            "gains": 0,   # non dispo dans ce tableau
            "ecart_driver": 99,
            "reussite_driver": 0.0,
            "reussite_entraineur": 0.0,
        }
        cleaned.append(horse)
    return pd.DataFrame(cleaned)

# ══════════════════════════════════════════════════════════════════
#  SCORING SIMPLIFIÉ (mais fonctionnel)
# ══════════════════════════════════════════════════════════════════
def decode_musique(musique):
    if not musique:
        return []
    # nettoie
    musique = re.sub(r'\([^)]+\)', '', musique)
    tokens = re.findall(r'[0-9]+|[A-Za-z]', musique)
    scores = []
    for t in tokens:
        if t.isdigit():
            p = int(t)
            if p == 1:
                scores.append(10)
            elif p == 2:
                scores.append(7)
            elif p == 3:
                scores.append(5)
            elif p <= 5:
                scores.append(3)
            else:
                scores.append(1)
        else:
            scores.append(0)  # D, A, M, p
    return scores

def calc_musique_score(musique, n=5):
    scores = decode_musique(musique)
    if not scores:
        return 5.0
    recent = scores[-n:]
    return sum(recent) / max(1, len(recent)) * 1.0

def calculate_scores(df):
    if df.empty:
        return df
    df = df.copy()
    df["score_musique"] = df["musique"].apply(calc_musique_score)
    # Cote inversée (plus petite cote = meilleur score)
    df["score_cote"] = df["cote_pmu"].apply(lambda x: max(0, 10 - (x / 10)) if x > 0 else 5)
    # Score global simple
    df["score_global"] = (df["score_musique"] * 0.7 + df["score_cote"] * 0.3).round(2)
    df["rang_score"] = df["score_global"].rank(ascending=False, method='min').astype(int)
    df["proba_victoire"] = (df["score_global"] / df["score_global"].sum() * 100).round(1)
    return df.sort_values("score_global", ascending=False).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════
#  INTERFACE STREAMLIT
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.main-header { background: linear-gradient(135deg,#0d3320,#1a6b3c); color: white; padding: 2rem; border-radius: 16px; text-align: center; }
.stButton>button { background: #1a6b3c; color: white; border-radius: 10px; }
</style>
<div class='main-header'>
    <h1>🏇 PronoHippique AI v2.2</h1>
    <p>Spécial tableau PMU – extraction robuste même sans JSON parfait</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("🔑 Clés API")
    gemini_key = st.text_input("Google Gemini (recommandé)", type="password")
    openai_key = st.text_input("OpenAI (optionnel)", type="password")
    st.divider()
    st.subheader("⚙️ Options")
    debug = st.checkbox("🔍 Afficher le texte OCR brut (debug)", value=False)

uploaded = st.file_uploader("📸 Images du tableau (PNG/JPG)", type=["png","jpg","jpeg"], accept_multiple_files=True)

if uploaded:
    cols = st.columns(min(len(uploaded), 4))
    for i, f in enumerate(uploaded):
        with cols[i % 4]:
            st.image(Image.open(f), caption=f.name, use_container_width=True)
    
    if st.button("🚀 ANALYSER"):
        progress = st.progress(0)
        extractions = []
        for i, f in enumerate(uploaded):
            progress.progress(int((i+1)/len(uploaded)*50))
            img = Image.open(f).convert("RGB")
            res = extract_data_from_image(img, gemini_key, openai_key, preferred="gemini" if gemini_key else "easyocr")
            extractions.append(res)
            if debug and res.get("raw_text"):
                st.text_area(f"Texte OCR brut – {f.name}", res["raw_text"][:1000], height=150)
        
        progress.progress(75)
        merged = merge_extracted_data(extractions)
        df_raw = clean_horse_data(merged.get("chevaux", []))
        
        if df_raw.empty:
            st.error("❌ Aucun cheval trouvé. Vérifiez la qualité des images ou activez le mode debug pour voir le texte OCR.")
            if debug:
                st.write("Extractions brutes:", extractions)
        else:
            df = calculate_scores(df_raw)
            progress.progress(100)
            st.success(f"✅ {len(df)} chevaux extraits !")
            
            st.subheader("🏆 Classement pronostiqué")
            st.dataframe(df[["rang_score", "numero", "cheval", "score_global", "proba_victoire", "cote_pmu", "musique"]], use_container_width=True)
            
            # Top 3
            top3 = df.head(3)
            cols = st.columns(3)
            for i, (_, row) in enumerate(top3.iterrows()):
                with cols[i]:
                    st.metric(f"#{row['numero']} {row['cheval']}", f"{row['score_global']}/10", f"Proba {row['proba_victoire']}%")
            
            # Graphique simple
            fig = go.Figure(go.Bar(x=df["cheval"], y=df["score_global"], marker_color="#1a6b3c"))
            fig.update_layout(title="Score global par cheval", xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Export CSV
            csv = df.to_csv(index=False).encode()
            st.download_button("📥 Télécharger CSV", csv, "pronostic.csv", "text/csv")
