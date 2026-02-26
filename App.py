"""
╔══════════════════════════════════════════════════════════════════╗
║          🏇  PronoHippique AI  —  Script complet unique          ║
...
    Le jeu peut créer une dépendance. Jouez de manière responsable.</em><br>
    <small>Joueurs Info Service : <strong>09 74 75 13 13</strong></small>
</div>
""", unsafe_allow_html=True)
Tool Call
Function Name:
Write
Arguments:
file_path:
/home/user/single_app/app.py
content:
"""
╔══════════════════════════════════════════════════════════════════╗
║          🏇  PronoHippique AI  —  Script complet unique          ║
║     Application Streamlit de pronostics hippiques intelligente   ║
║         Déployable directement sur Streamlit Cloud               ║
╚══════════════════════════════════════════════════════════════════╝

Tous les modules sont intégrés dans ce fichier unique :
  → OCR Extractor   : extraction Gemini / OpenAI / EasyOCR
  → Data Cleaner    : nettoyage & parsing des données
  → Scorer          : algorithme de scoring multi-critères
  → Pronostic       : génération Trio / Quinté / classement
  → Visualizer      : graphiques Plotly interactifs
  → App             : interface Streamlit complète
"""

# ══════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import os
import re
import json
import base64
import time

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (doit être le 1er appel Streamlit)
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🏇 PronoHippique AI",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════
#  ██████╗  ██╗      ██████╗   ██████╗
#  ██╔══██╗ ██║     ██╔═══██╗ ██╔════╝
#  ██████╔╝ ██║     ██║   ██║ ██║
#  ██╔══██╗ ██║     ██║   ██║ ██║
#  ██████╔╝ ███████╗╚██████╔╝ ╚██████╗
#  ╚═════╝  ╚══════╝ ╚═════╝   ╚═════╝
#  MODULE 1 — OCR EXTRACTOR
# ══════════════════════════════════════════════════════════════════

def _encode_image_base64(image: Image.Image) -> str:
    """Encode une image PIL en base64 PNG."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _build_extraction_prompt() -> str:
    """Prompt universel pour l'extraction structurée des tableaux hippiques."""
    return """Tu es un expert en analyse de tableaux de statistiques hippiques françaises.
Analyse cette image et extrais TOUTES les données visibles sous forme JSON structuré.

Détermine d'abord le TYPE de tableau parmi :
- "partants"          : liste des partants (N°, Cheval, SA, Dist, Driver, Entraîneur, Musique, Gains, Cote_PMU, Cote_Genybet)
- "records"           : records absolus (N°, Cheval, SA, Dist, Driver, Record, Date_Record)
- "stats_drivers"     : statistiques drivers PMU (N°, Cheval, Dist, Driver, Courses_Driver, Victoires_Driver, Ecart_Driver, Reussite_Driver, Musique_Driver)
- "stats_entraineurs" : statistiques entraîneurs PMU (N°, Cheval, Dist, Entraineur, Courses_Entraineur, Victoires_Entraineur, Ecart_Entraineur, Reussite_Entraineur, Musique_Entraineur)
- "unknown"           : si non reconnu

Réponds UNIQUEMENT avec un JSON valide, sans markdown, sans explication.
Format attendu :
{
  "table_type": "partants",
  "nb_partants": 15,
  "chevaux": [
    {
      "numero": 1,
      "cheval": "NomDuCheval",
      "sa": "M7",
      "dist": 2100,
      "driver": "M. Mottier",
      "entraineur": "J. Westholm",
      "musique": "(25)1aDaDa",
      "gains": 219481,
      "cote_pmu": 1.9,
      "cote_genybet": 2.1,
      "record": "1'10\"0",
      "date_record": "08/05/25 - Bergsaker 1640m, 1eme",
      "courses_driver": 1288,
      "victoires_driver": 213,
      "ecart_driver": 4,
      "reussite_driver": 16,
      "musique_driver": "Da8a9aDa1aDaDa4m",
      "courses_entraineur": 187,
      "victoires_entraineur": 31,
      "ecart_entraineur": 7,
      "reussite_entraineur": 16,
      "musique_entraineur": "3a0aDa8a5a6a2a1a"
    }
  ]
}

Notes importantes :
- Les champs absents de cette image doivent valoir null
- La musique encode les performances : chiffre=position, D=distancé, m/a=disqualifié, (25)=non partant
- Les pourcentages de réussite sont en valeur numérique (16 pour 16%)
- Les cotes utilisent la virgule comme séparateur décimal
- Extrais TOUS les chevaux visibles (généralement 15)"""


def _parse_json_response(raw_text: str) -> dict:
    """Parse robuste de la réponse JSON de l'IA."""
    if not raw_text:
        return {"error": "Réponse vide"}
    clean = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
    clean = re.sub(r"```\s*$", "", clean).strip()
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start == -1 or end == 0:
        return {"error": "JSON introuvable", "raw_text": raw_text}
    json_str = clean[start:end]
    # Réparer les virgules décimales françaises dans les cotes
    json_str = re.sub(r'("cote_[^"]+"\s*:\s*)(\d+),(\d+)', r'\1\2.\3', json_str)
    # Supprimer trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    try:
        data = json.loads(json_str)
        data["raw_text"] = raw_text
        return data
    except json.JSONDecodeError as e:
        return {"error": f"JSON invalide : {e}", "raw_text": raw_text}


def extract_with_gemini(image: Image.Image, api_key: str) -> dict:
    """Extraction via Google Gemini Vision."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([_build_extraction_prompt(), image])
        result = _parse_json_response(response.text)
        result["ocr_engine"] = "Gemini Vision"
        return result
    except Exception as e:
        return {"error": str(e), "ocr_engine": "Gemini Vision"}


def extract_with_openai(image: Image.Image, api_key: str) -> dict:
    """Extraction via OpenAI GPT-4o Vision."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        img_b64 = _encode_image_base64(image)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_extraction_prompt()},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }],
            max_tokens=4096,
        )
        result = _parse_json_response(response.choices[0].message.content)
        result["ocr_engine"] = "OpenAI GPT-4o"
        return result
    except Exception as e:
        return {"error": str(e), "ocr_engine": "OpenAI GPT-4o"}


def extract_with_easyocr(image: Image.Image) -> dict:
    """Extraction via EasyOCR (fallback local, précision réduite)."""
    try:
        import easyocr
        import numpy as np as np_local
        reader = easyocr.Reader(["fr", "en"], gpu=False, verbose=False)
        img_array = np_local.array(image)
        results = reader.readtext(img_array, detail=1)
        lines_by_y = {}
        for bbox, text, conf in results:
            if conf < 0.3:
                continue
            y_center = int((bbox[0][1] + bbox[2][1]) / 2)
            y_bucket = (y_center // 20) * 20
            lines_by_y.setdefault(y_bucket, []).append((bbox[0][0], text.strip()))
        lines = []
        for y in sorted(lines_by_y):
            items = sorted(lines_by_y[y], key=lambda x: x[0])
            lines.append(" | ".join(t for _, t in items))
        return {
            "type": "raw_ocr",
            "raw_text": "\n".join(lines),
            "lines": lines,
            "ocr_engine": "EasyOCR (mode basique)",
            "chevaux": [],          # EasyOCR ne produit pas de JSON structuré
            "table_type": "unknown",
        }
    except Exception as e:
        return {
            "error": str(e),
            "ocr_engine": "EasyOCR",
            "chevaux": [],
            "table_type": "unknown",
        }


def extract_data_from_image(
    image: Image.Image,
    gemini_api_key: str = "",
    openai_api_key: str = "",
) -> dict:
    """Orchestre l'extraction : Gemini → OpenAI → EasyOCR."""
    if gemini_api_key:
        result = extract_with_gemini(image, gemini_api_key)
        if "chevaux" in result and result["chevaux"]:
            return result
    if openai_api_key:
        result = extract_with_openai(image, openai_api_key)
        if "chevaux" in result and result["chevaux"]:
            return result
    return extract_with_easyocr(image)


def merge_extracted_data(extractions: list) -> dict:
    """Fusionne les données extraites de plusieurs images de la même course."""
    merged = {}
    table_types = []
    for ext in extractions:
        if not ext.get("chevaux"):
            continue
        if "table_type" in ext:
            table_types.append(ext["table_type"])
        for horse in ext["chevaux"]:
            num = horse.get("numero")
            if num is None:
                continue
            try:
                num = int(num)
            except (ValueError, TypeError):
                pass
            if num not in merged:
                merged[num] = {"numero": num}
            for key, val in horse.items():
                if val is not None and val != "" and key != "numero":
                    if key not in merged[num] or merged[num][key] is None:
                        merged[num][key] = val
    chevaux_list = sorted(merged.values(), key=lambda x: x.get("numero", 99))
    return {
        "chevaux": chevaux_list,
        "nb_partants": len(chevaux_list),
        "table_types_detectes": list(set(table_types)),
    }


# ══════════════════════════════════════════════════════════════════
#   ██████╗ ██╗     ███████╗ █████╗ ███╗   ██╗███████╗██████╗
#  ██╔════╝ ██║     ██╔════╝██╔══██╗████╗  ██║██╔════╝██╔══██╗
#  ██║      ██║     █████╗  ███████║██╔██╗ ██║█████╗  ██████╔╝
#  ██║      ██║     ██╔══╝  ██╔══██║██║╚██╗██║██╔══╝  ██╔══██╗
#  ╚██████╗ ███████╗███████╗██║  ██║██║ ╚████║███████╗██║  ██║
#   ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝
#  MODULE 2 — DATA CLEANER
# ══════════════════════════════════════════════════════════════════

def _safe_int(val, default=0) -> int:
    if val is None:
        return default
    try:
        s = str(val).replace(" ", "").replace("\xa0", "").strip()
        m = re.match(r"(\d+)", s)
        return int(m.group(1)) if m else default
    except Exception:
        return default


def _safe_float(val, default=0.0) -> float:
    if val is None:
        return default
    try:
        s = str(val).replace(" ", "").replace("\xa0", "").replace(",", ".").strip()
        s2 = re.sub(r"[^\d.]", "", s)
        return float(s2) if s2 else default
    except Exception:
        return default


def _clean_str(val) -> str:
    if val is None:
        return ""
    return re.sub(r"[\x00-\x1f\x7f]", "", str(val).strip())


def _extract_sexe(sa: str) -> str:
    if not sa:
        return ""
    m = re.match(r"([A-Za-z]+)", sa.strip())
    return m.group(1).upper() if m else ""


def _extract_age(sa: str) -> int:
    if not sa:
        return 0
    m = re.search(r"(\d+)", sa.strip())
    return int(m.group(1)) if m else 0


def _parse_record_to_seconds(record: str) -> float:
    """Convertit 1'10\"0 → 70.0 secondes."""
    if not record:
        return 0.0
    record = record.replace(",", ".").strip()
    patterns = [
        r"(\d+)'(\d+)[\"](\\d+)",
        r"(\d+)'(\d+)\.(\d+)",
        r"(\d+)'(\d+)\"(\d+)",
        r"(\d+)'(\d+)(\d)",
        r"(\d+)'(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, record)
        if m:
            g = m.groups()
            minutes = int(g[0])
            seconds = int(g[1])
            tenths  = int(g[2]) / 10 if len(g) > 2 else 0.0
            return minutes * 60 + seconds + tenths
    return 0.0


def _parse_cote(val) -> float:
    if val is None:
        return 0.0
    try:
        s = str(val).replace(",", ".").replace(" ", "").strip()
        m = re.search(r"[\d.]+", s)
        return float(m.group()) if m else 0.0
    except Exception:
        return 0.0


def _parse_pct(val) -> float:
    if val is None:
        return 0.0
    try:
        s = str(val).replace("%", "").replace(",", ".").strip()
        m = re.search(r"[\d.]+", s)
        return float(m.group()) if m else 0.0
    except Exception:
        return 0.0


def _parse_ecart(val) -> int:
    if val is None:
        return 99
    s = str(val).strip()
    if s in ("000", "0000", "---", "-", ""):
        return 99
    if s == "00":
        return 50
    try:
        return int(s)
    except ValueError:
        return 99


def clean_horse_data(chevaux_raw: list) -> pd.DataFrame:
    """Nettoie et structure la liste des chevaux en DataFrame pandas."""
    if not chevaux_raw:
        return pd.DataFrame()
    cleaned = []
    for h in chevaux_raw:
        c = {
            "numero":               _safe_int(h.get("numero")),
            "cheval":               _clean_str(h.get("cheval", "")),
            "sa":                   _clean_str(h.get("sa", "")),
            "sexe":                 _extract_sexe(_clean_str(h.get("sa", ""))),
            "age":                  _extract_age(_clean_str(h.get("sa", ""))),
            "distance":             _safe_int(h.get("dist", 2100)),
            "driver":               _clean_str(h.get("driver", "")),
            "entraineur":           _clean_str(h.get("entraineur", "")),
            "record_brut":          _clean_str(str(h.get("record", "")) if h.get("record") else ""),
            "record_secondes":      _parse_record_to_seconds(_clean_str(str(h.get("record", "")))),
            "date_record":          _clean_str(h.get("date_record", "")),
            "gains":                _safe_float(h.get("gains", 0)),
            "cote_pmu":             _parse_cote(h.get("cote_pmu")),
            "cote_genybet":         _parse_cote(h.get("cote_genybet")),
            "courses_driver":       _safe_int(h.get("courses_driver", 0)),
            "victoires_driver":     _safe_int(h.get("victoires_driver", 0)),
            "ecart_driver":         _parse_ecart(h.get("ecart_driver")),
            "reussite_driver":      _parse_pct(h.get("reussite_driver")),
            "musique_driver":       _clean_str(h.get("musique_driver", "")),
            "courses_entraineur":   _safe_int(h.get("courses_entraineur", 0)),
            "victoires_entraineur": _safe_int(h.get("victoires_entraineur", 0)),
            "ecart_entraineur":     _parse_ecart(h.get("ecart_entraineur")),
            "reussite_entraineur":  _parse_pct(h.get("reussite_entraineur")),
            "musique_entraineur":   _clean_str(h.get("musique_entraineur", "")),
            "musique":              _clean_str(h.get("musique", "")),
        }
        # Musique principale : partants > driver > entraîneur
        if not c["musique"] and c["musique_driver"]:
            c["musique"] = c["musique_driver"]
        cleaned.append(c)

    df = pd.DataFrame(cleaned)
    if "numero" in df.columns:
        df = df.sort_values("numero").reset_index(drop=True)
    return df


# ── Décodeur de musique ─────────────────────────────────────────

def decode_musique(musique: str) -> list:
    """
    Décode la musique hippique en liste de résultats.
    Ex: "3a0aDa8a5a" → [{'pos':3,'type':'placé','score_base':5.0}, ...]
    """
    if not musique:
        return []
    cleaned = re.sub(r"\(\d+\)", "NP", str(musique))
    tokens = re.findall(r"NP|0{2,3}|\d+|[DdMmAa]", cleaned)
    results = []
    for tok in tokens:
        t = tok.upper()
        if t == "NP":
            continue
        elif t == "D":
            results.append({"pos": None, "type": "distancé",   "score_base": 0.0})
        elif t in ("M", "A"):
            results.append({"pos": None, "type": "disqualifié", "score_base": 0.0})
        elif re.match(r"^\d+$", tok):
            p = int(tok)
            if p == 0:
                results.append({"pos": 0, "type": "non_classé",     "score_base": 0.3})
            elif p == 1:
                results.append({"pos": 1, "type": "victoire",        "score_base": 10.0})
            elif p == 2:
                results.append({"pos": 2, "type": "placé",           "score_base": 7.0})
            elif p == 3:
                results.append({"pos": 3, "type": "placé",           "score_base": 5.0})
            elif p == 4:
                results.append({"pos": 4, "type": "proche_podium",   "score_base": 3.5})
            elif p == 5:
                results.append({"pos": 5, "type": "proche_podium",   "score_base": 2.5})
            elif p <= 7:
                results.append({"pos": p, "type": "milieu",          "score_base": 1.5})
            else:
                results.append({"pos": p, "type": "arrière",         "score_base": 0.5})
    return results


def calc_musique_score(musique: str, n_recent: int = 5) -> float:
    """Score pondéré de la musique, entre 0 et 10."""
    results = decode_musique(musique)
    if not results:
        return 0.0
    recent = results[-n_recent:]
    weights = [1.5 ** i for i in range(len(recent))]
    total_w = sum(weights)
    weighted = sum(r["score_base"] * w for r, w in zip(recent, weights))
    max_possible = 10.0 * total_w
    return min(10.0, weighted / max_possible * 10.0) if max_possible > 0 else 0.0


def count_wins(musique: str, n: int = 5) -> int:
    results = decode_musique(musique)
    return sum(1 for r in results[-n:] if r.get("pos") == 1)


def count_placed(musique: str, n: int = 5) -> int:
    results = decode_musique(musique)
    return sum(1 for r in results[-n:] if r.get("pos") in (1, 2, 3))


def assess_data_quality(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"qualite": 0, "nb_chevaux": 0}
    total = len(df)
    scores = {}
    for field in ["cheval", "numero", "driver", "entraineur",
                  "musique", "reussite_driver", "record_secondes"]:
        if field in df.columns:
            filled = df[field].notna() & (df[field] != "") & (df[field] != 0)
            scores[field] = filled.sum() / total
    quality = sum(scores.values()) / len(scores) * 100 if scores else 0
    return {"qualite": round(quality), "nb_chevaux": total, "details": scores}


# ══════════════════════════════════════════════════════════════════
#  ███████╗ ██████╗  ██████╗ ██████╗ ███████╗██████╗
#  ██╔════╝██╔════╝ ██╔═══██╗██╔══██╗██╔════╝██╔══██╗
#  ███████╗██║      ██║   ██║██████╔╝█████╗  ██████╔╝
#  ╚════██║██║      ██║   ██║██╔══██╗██╔══╝  ██╔══██╗
#  ███████║╚██████╗ ╚██████╔╝██║  ██║███████╗██║  ██║
#  ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
#  MODULE 3 — SCORER
# ══════════════════════════════════════════════════════════════════

WEIGHTS = {
    "record_absolu":        0.18,
    "musique_recente":      0.20,
    "reussite_driver":      0.12,
    "reussite_entraineur":  0.10,
    "ecart":                0.10,
    "gains":                0.08,
    "victoires_driver":     0.07,
    "cote_inverse":         0.09,
    "regularite":           0.06,
}

WEIGHTS_QUINTE = {**WEIGHTS, "musique_recente": 0.22, "record_absolu": 0.16}
WEIGHTS_PRIX   = {**WEIGHTS, "record_absolu": 0.22,   "musique_recente": 0.18}

RACE_WEIGHTS = {
    "quinté":  WEIGHTS_QUINTE,
    "prix":    WEIGHTS_PRIX,
    "trot":    WEIGHTS,
    "default": WEIGHTS,
}


def _normalize(series: pd.Series) -> pd.Series:
    col = series.fillna(0)
    mn, mx = col.min(), col.max()
    if mx == mn:
        return pd.Series([5.0] * len(col), index=col.index)
    return (col - mn) / (mx - mn) * 10.0


def _score_record_col(df: pd.DataFrame) -> pd.Series:
    sec = df["record_secondes"].replace(0, np.nan)
    if sec.isna().all():
        return pd.Series([5.0] * len(df), index=df.index)
    mn, mx = sec.min(), sec.max()
    if mx == mn:
        return pd.Series([5.0] * len(df), index=df.index)
    return ((mx - sec) / (mx - mn) * 10.0).fillna(3.0)


def _score_ecart_col(df: pd.DataFrame) -> pd.Series:
    def _map(e):
        if e == 0:   return 10.0
        if e == 1:   return 9.0
        if e <= 3:   return 7.0
        if e <= 5:   return 5.5
        if e <= 8:   return 4.0
        if e <= 15:  return 2.5
        if e <= 30:  return 1.5
        return 0.5
    return df["ecart_driver"].fillna(99).apply(_map)


def _score_cote_col(df: pd.DataFrame) -> pd.Series:
    cote = df["cote_pmu"].replace(0, np.nan)
    if cote.isna().all():
        cote = df["cote_genybet"].replace(0, np.nan)
    if cote.isna().all():
        return pd.Series([5.0] * len(df), index=df.index)
    inv = (1.0 / cote.fillna(100.0))
    mn, mx = inv.min(), inv.max()
    if mx == mn:
        return pd.Series([5.0] * len(df), index=df.index)
    return (inv - mn) / (mx - mn) * 10.0


def _categorize(row) -> str:
    rang = row.get("rang_score", 99)
    if rang == 1:  return "🥇 Favori IA"
    if rang == 2:  return "🥈 Dauphin"
    if rang == 3:  return "🥉 Outsider solide"
    if rang <= 5:  return "⭐ Top 5"
    if rang <= 8:  return "💡 Outsider"
    if rang <= 12: return "🎲 Long shot"
    return "❓ Très outsider"


def calculate_scores(df: pd.DataFrame, race_type: str = "default") -> pd.DataFrame:
    """Calcule les scores de chaque cheval et retourne le DataFrame enrichi."""
    if df.empty:
        return df
    df = df.copy()
    W = RACE_WEIGHTS.get(race_type, WEIGHTS)

    # Scores individuels
    df["score_record"]               = _score_record_col(df)
    df["score_musique"]              = df["musique"].apply(lambda m: calc_musique_score(str(m) if m else ""))
    df["score_musique_driver"]       = df["musique_driver"].apply(lambda m: calc_musique_score(str(m) if m else ""))
    df["score_musique_entraineur"]   = df["musique_entraineur"].apply(lambda m: calc_musique_score(str(m) if m else ""))
    df["score_musique_combine"]      = (df["score_musique"] * 0.5
                                        + df["score_musique_driver"] * 0.3
                                        + df["score_musique_entraineur"] * 0.2)
    df["score_reussite_driver"]      = df["reussite_driver"].fillna(0).clip(0, 100) / 10.0
    df["score_reussite_entraineur"]  = df["reussite_entraineur"].fillna(0).clip(0, 100) / 10.0
    df["score_ecart"]                = _score_ecart_col(df)
    df["score_gains"]                = _normalize(df["gains"])
    df["score_victoires_driver"]     = _normalize(df["victoires_driver"])
    df["score_cote"]                 = _score_cote_col(df)
    df["wins_recents"]               = df["musique"].apply(lambda m: count_wins(str(m) if m else ""))
    df["places_recents"]             = df["musique"].apply(lambda m: count_placed(str(m) if m else ""))
    df["score_regularite"]           = (df["wins_recents"] * 2 + df["places_recents"]) / 15.0 * 10.0

    # Score global pondéré (normalisé sur 10)
    total_w = sum(W.values())
    df["score_global"] = (
        df["score_record"]              * W["record_absolu"]       +
        df["score_musique_combine"]     * W["musique_recente"]     +
        df["score_reussite_driver"]     * W["reussite_driver"]     +
        df["score_reussite_entraineur"] * W["reussite_entraineur"] +
        df["score_ecart"]               * W["ecart"]               +
        df["score_gains"]               * W["gains"]               +
        df["score_victoires_driver"]    * W["victoires_driver"]    +
        df["score_cote"]                * W["cote_inverse"]        +
        df["score_regularite"]          * W["regularite"]
    ) / total_w

    score_cols = [c for c in df.columns if c.startswith("score_")]
    df[score_cols] = df[score_cols].round(2)
    df["rang_score"] = df["score_global"].rank(ascending=False, method="min").astype(int)
    df["categorie"]  = df.apply(_categorize, axis=1)

    return df.sort_values("score_global", ascending=False).reset_index(drop=True)


def get_score_breakdown(row: pd.Series) -> dict:
    return {
        "⏱️ Record Absolu":        round(row.get("score_record", 0), 2),
        "🎵 Musique Récente":       round(row.get("score_musique_combine", 0), 2),
        "🏇 Réussite Driver":       round(row.get("score_reussite_driver", 0), 2),
        "👨‍🏫 Réussite Entraîneur": round(row.get("score_reussite_entraineur", 0), 2),
        "🔄 Fraîcheur (Écart)":     round(row.get("score_ecart", 0), 2),
        "💎 Gains":                 round(row.get("score_gains", 0), 2),
        "🏆 Victoires Driver":      round(row.get("score_victoires_driver", 0), 2),
        "💰 Favori (Cote)":         round(row.get("score_cote", 0), 2),
        "📈 Régularité":            round(row.get("score_regularite", 0), 2),
    }


# ══════════════════════════════════════════════════════════════════
#  ██████╗ ██████╗  ██████╗ ███╗   ██╗ ██████╗
#  ██╔══██╗██╔══██╗██╔═══██╗████╗  ██║██╔═══██╗
#  ██████╔╝██████╔╝██║   ██║██╔██╗ ██║██║   ██║
#  ██╔═══╝ ██╔══██╗██║   ██║██║╚██╗██║██║   ██║
#  ██║     ██║  ██║╚██████╔╝██║ ╚████║╚██████╔╝
#  ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝
#  MODULE 4 — PRONOSTIC
# ══════════════════════════════════════════════════════════════════

def generate_trio_combinations(df: pd.DataFrame, n: int = 10) -> list:
    """Génère n combinaisons Trio intelligentes."""
    if len(df) < 3:
        return []
    sorted_df = df.sort_values("score_global", ascending=False)
    nums   = sorted_df["numero"].tolist()
    scores = sorted_df["score_global"].tolist()
    combos = set()

    # Top 3 strict
    combos.add(tuple(sorted(nums[:3])))

    # Top 2 + outsider
    for i in range(2, min(7, len(nums))):
        combos.add(tuple(sorted([nums[0], nums[1], nums[i]])))

    # 1er + 2 outsiders
    for i in range(2, min(5, len(nums))):
        for j in range(i + 1, min(7, len(nums))):
            combos.add(tuple(sorted([nums[0], nums[i], nums[j]])))

    # Compléter par tirage pondéré
    pool = min(10, len(nums))
    total_s = sum(scores[:pool]) or 1
    w_norm = [s / total_s for s in scores[:pool]]
    attempts = 0
    while len(combos) < n and attempts < 1000:
        attempts += 1
        idx = np.random.choice(range(pool), size=3, replace=False, p=w_norm[:pool])
        combos.add(tuple(sorted([nums[i] for i in idx])))

    return [list(c) for c in list(combos)[:n]]


def generate_quinte_combinations(df: pd.DataFrame, n: int = 10) -> list:
    """Génère n combinaisons Quinté+ intelligentes."""
    if len(df) < 5:
        return []
    sorted_df = df.sort_values("score_global", ascending=False)
    nums   = sorted_df["numero"].tolist()
    scores = sorted_df["score_global"].tolist()
    combos = set()

    combos.add(tuple(sorted(nums[:5])))
    for i in range(4, min(8, len(nums))):
        combos.add(tuple(sorted(nums[:4] + [nums[i]])))
    for i in range(3, min(7, len(nums))):
        for j in range(i + 1, min(9, len(nums))):
            combos.add(tuple(sorted(nums[:3] + [nums[i], nums[j]])))
    for i in range(2, min(6, len(nums))):
        for j in range(i + 1, min(8, len(nums))):
            for k in range(j + 1, min(10, len(nums))):
                combos.add(tuple(sorted(nums[:2] + [nums[i], nums[j], nums[k]])))

    pool = min(12, len(nums))
    total_s = sum(scores[:pool]) or 1
    w_norm = [s / total_s for s in scores[:pool]]
    attempts = 0
    while len(combos) < n and attempts < 2000:
        attempts += 1
        idx = np.random.choice(range(pool), size=5, replace=False, p=w_norm[:pool])
        combos.add(tuple(sorted([nums[i] for i in idx])))

    return [list(c) for c in list(combos)[:n]]


def _build_arguments(row: pd.Series) -> list:
    args = []
    if row.get("record_secondes", 0) > 0:
        sec = row["record_secondes"]
        args.append(f"⏱️ Record : {int(sec//60)}'{sec%60:.1f}\" — Vitesse pure élevée")
    rd = row.get("reussite_driver", 0)
    if rd >= 15:   args.append(f"🏇 Driver en grande forme ({rd:.0f}% de réussite)")
    elif rd >= 10: args.append(f"🏇 Driver compétent ({rd:.0f}% de réussite)")
    re_ = row.get("reussite_entraineur", 0)
    if re_ >= 15:   args.append(f"🎯 Entraîneur excellent ({re_:.0f}% de réussite)")
    elif re_ >= 10: args.append(f"🎯 Entraîneur solide ({re_:.0f}% de réussite)")
    ecart = row.get("ecart_driver", 99)
    if ecart == 0:    args.append("🔥 Vient de gagner — Pleine confiance !")
    elif ecart <= 2:  args.append(f"✅ Victoire récente (il y a {ecart} course(s))")
    elif ecart > 20:  args.append(f"⚠️ Long sans victoire ({ecart} courses)")
    w = row.get("wins_recents", 0)
    p = row.get("places_recents", 0)
    if w >= 2:   args.append(f"🏆 {w} victoire(s) dans les 5 dernières courses")
    elif p >= 3: args.append(f"📊 {p} fois dans le Top 3 récemment")
    elif row.get("score_musique_combine", 0) < 2:
        args.append("📉 Forme récente mitigée — vigilance requise")
    cote = row.get("cote_pmu", 0)
    if cote > 0:
        if cote <= 3:   args.append(f"💰 Grand favori PMU (cote {cote})")
        elif cote <= 7: args.append(f"💰 Favori PMU (cote {cote})")
        elif cote >= 50:args.append(f"🎲 Longshot potentiellement intéressant (cote {cote})")
    if not args:
        args.append("📋 Données partielles — analyse limitée")
    return args


def generate_pronostic_report(df: pd.DataFrame) -> dict:
    """Génère le rapport de pronostic complet."""
    if df.empty:
        return {"error": "Aucune donnée"}
    sdf = df.sort_values("score_global", ascending=False).reset_index(drop=True)
    nums = sdf["numero"].tolist()

    gap = float(sdf.iloc[0]["score_global"] - sdf.iloc[1]["score_global"]) if len(sdf) > 1 else 0
    if gap > 2.0:   confiance = "Haute 🔥"
    elif gap > 1.0: confiance = "Moyenne ⭐"
    else:           confiance = "Faible ⚠️ Course ouverte"

    trios  = generate_trio_combinations(sdf, 10)
    quinte = generate_quinte_combinations(sdf, 10)

    arguments = {
        row["cheval"]: _build_arguments(row)
        for _, row in sdf.head(5).iterrows()
    }

    return {
        "classement": sdf[["numero", "cheval", "rang_score", "score_global",
                            "categorie", "reussite_driver", "reussite_entraineur",
                            "cote_pmu"]].to_dict("records"),
        "top3":      sdf.head(3)[["numero", "cheval", "score_global", "categorie"]].to_dict("records"),
        "bases":     sdf.head(2)[["numero", "cheval", "score_global"]].to_dict("records"),
        "outsiders": sdf.iloc[2:6][["numero", "cheval", "score_global", "cote_pmu"]].to_dict("records"),
        "trios":     trios,
        "quintes":   quinte,
        "confiance": confiance,
        "gap":       round(gap, 2),
        "arguments": arguments,
        "nb_partants": len(df),
        "favori": {
            "numero": int(sdf.iloc[0]["numero"]),
            "cheval": sdf.iloc[0]["cheval"],
            "score":  round(float(sdf.iloc[0]["score_global"]), 2),
        },
    }


# ══════════════════════════════════════════════════════════════════
#  ██╗   ██╗██╗███████╗██╗   ██╗ █████╗ ██╗
#  ██║   ██║██║██╔════╝██║   ██║██╔══██╗██║
#  ██║   ██║██║███████╗██║   ██║███████║██║
#  ╚██╗ ██╔╝██║╚════██║██║   ██║██╔══██║██║
#   ╚████╔╝ ██║███████║╚██████╔╝██║  ██║███████╗
#    ╚═══╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝
#  MODULE 5 — VISUALIZER
# ══════════════════════════════════════════════════════════════════

_C = {
    "primary":  "#1a6b3c",
    "accent":   "#f28a00",
    "light_bg": "#f0f7f3",
    "dark":     "#0d3320",
}


def plot_scores_bar(df: pd.DataFrame) -> go.Figure:
    sdf = df.sort_values("score_global", ascending=True).tail(15)
    labels = [f"#{int(r['numero'])} {r['cheval']}" for _, r in sdf.iterrows()]
    scores = sdf["score_global"].round(2).tolist()
    n = len(scores)
    colors = []
    for i in range(n):
        rank = n - i
        if rank == 1:   colors.append("#ffd700")
        elif rank == 2: colors.append("#c0c0c0")
        elif rank == 3: colors.append("#cd7f32")
        elif rank <= 5: colors.append("#2c9e5e")
        elif rank <= 8: colors.append("#5ab87e")
        else:           colors.append("#90d4a8")

    fig = go.Figure(go.Bar(
        x=scores, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(color="white", width=0.5)),
        text=[f"{s:.1f}" for s in scores], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Score : %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="🏆 Scores Globaux des Partants",
                   font=dict(size=17, color=_C["dark"]), x=0.5),
        xaxis=dict(title="Score (sur 10)", range=[0, max(scores) * 1.15] if scores else [0, 10],
                   gridcolor="#e8f5ee"),
        yaxis=dict(title="", tickfont=dict(size=11)),
        plot_bgcolor=_C["light_bg"], paper_bgcolor="white",
        height=max(380, n * 34),
        margin=dict(l=190, r=70, t=55, b=35),
        showlegend=False,
    )
    return fig


def plot_radar_top3(df: pd.DataFrame) -> go.Figure:
    sdf = df.sort_values("score_global", ascending=False).head(3)
    cats = ["Record", "Musique", "Réussite Driver",
            "Réussite Entr.", "Fraîcheur", "Gains", "Régularité"]
    cols = ["score_record", "score_musique_combine", "score_reussite_driver",
            "score_reussite_entraineur", "score_ecart", "score_gains", "score_regularite"]
    fig = go.Figure()
    pal = ["#ffd700", "#c0c0c0", "#cd7f32"]
    for idx, (_, row) in enumerate(sdf.iterrows()):
        vals = [row.get(c, 0) for c in cols]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill="toself",
            name=f"#{int(row['numero'])} {row['cheval']}",
            line=dict(color=pal[idx], width=2),
            opacity=0.75,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10], gridcolor="#ccddcc"),
                   bgcolor=_C["light_bg"]),
        title=dict(text="🎯 Profil des 3 Premiers",
                   font=dict(size=17, color=_C["dark"]), x=0.5),
        paper_bgcolor="white", height=430,
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
    )
    return fig


def plot_form_history(df: pd.DataFrame, top_n: int = 5) -> go.Figure:
    sdf = df.sort_values("score_global", ascending=False).head(top_n)
    fig = go.Figure()
    pal = ["#ffd700", "#c0c0c0", "#cd7f32", "#2c9e5e", "#5ab87e"]
    for idx, (_, row) in enumerate(sdf.iterrows()):
        results = decode_musique(str(row.get("musique", "") or ""))
        if not results:
            continue
        positions = [
            r.get("pos") if r.get("pos") is not None else 14
            for r in results[-8:]
        ]
        y_disp = [15 - p for p in positions]
        label = f"#{int(row['numero'])} {row['cheval']}"
        fig.add_trace(go.Scatter(
            x=list(range(1, len(y_disp) + 1)),
            y=y_disp,
            mode="lines+markers",
            name=label,
            line=dict(color=pal[idx % len(pal)], width=2.5),
            marker=dict(size=8),
            text=[str(p) if p < 14 else "D/Disq" for p in positions],
            hovertemplate=f"<b>{label}</b><br>Course : %{{x}}<br>Position : %{{text}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text="📈 Historique de Forme (8 dernières courses)",
                   font=dict(size=16, color=_C["dark"]), x=0.5),
        xaxis=dict(title="← Ancienne  |  Récente →", dtick=1),
        yaxis=dict(title="Performance →",
                   tickvals=[1, 5, 9, 13, 14],
                   ticktext=["14ème+", "10ème", "6ème", "2ème", "1ère"]),
        plot_bgcolor=_C["light_bg"], paper_bgcolor="white", height=370,
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
    )
    return fig


def plot_driver_comparison(df: pd.DataFrame) -> go.Figure:
    disp = df[df["reussite_driver"] > 0].sort_values("reussite_driver", ascending=False).head(10)
    if disp.empty:
        return go.Figure()
    labels = [f"#{int(r['numero'])} {r['driver']}" for _, r in disp.iterrows()]
    fig = go.Figure([
        go.Bar(name="% Driver",     x=labels, y=disp["reussite_driver"],    marker_color=_C["primary"]),
        go.Bar(name="% Entraîneur", x=labels, y=disp["reussite_entraineur"],marker_color=_C["accent"]),
    ])
    fig.update_layout(
        title=dict(text="📊 Réussite Driver vs Entraîneur",
                   font=dict(size=16, color=_C["dark"]), x=0.5),
        xaxis=dict(tickangle=-30),
        yaxis=dict(title="Réussite (%)", gridcolor="#e8f5ee"),
        barmode="group",
        plot_bgcolor=_C["light_bg"], paper_bgcolor="white", height=370,
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
    )
    return fig


def plot_gauge(score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Indice de Confiance", "font": {"size": 15}},
        number={"suffix": "/10", "font": {"size": 26}},
        gauge={
            "axis": {"range": [0, 10], "tickwidth": 1},
            "bar":  {"color": _C["primary"]},
            "steps": [
                {"range": [0, 3], "color": "#e74c3c"},
                {"range": [3, 6], "color": "#f39c12"},
                {"range": [6, 8], "color": "#2ecc71"},
                {"range": [8, 10],"color": "#27ae60"},
            ],
        },
    ))
    fig.update_layout(paper_bgcolor="white", height=240,
                      margin=dict(l=20, r=20, t=40, b=20))
    return fig


# ══════════════════════════════════════════════════════════════════
#   █████╗ ██████╗ ██████╗
#  ██╔══██╗██╔══██╗██╔══██╗
#  ███████║██████╔╝██████╔╝
#  ██╔══██║██╔═══╝ ██╔═══╝
#  ██║  ██║██║     ██║
#  ╚═╝  ╚═╝╚═╝     ╚═╝
#  MODULE 6 — STREAMLIT APP
# ══════════════════════════════════════════════════════════════════

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --primary:#1a6b3c; --secondary:#2c9e5e;
    --accent:#f28a00;  --dark:#0d3320;
    --bg:#f0f7f3;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #0d3320 0%, #1a6b3c 55%, #2c9e5e 100%);
    color: white; padding: 2rem 2.5rem; border-radius: 16px;
    text-align: center; margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(13,51,32,.4);
}
.main-header h1 { font-size: 2.7rem; margin: 0; letter-spacing: 2px; }
.main-header p  { font-size: 1.05rem; margin: .5rem 0 0; opacity: .88; }

/* Cards */
.card {
    background: white; border-radius: 12px; padding: 1.4rem;
    box-shadow: 0 4px 16px rgba(0,0,0,.07); margin-bottom: .9rem;
    border-left: 5px solid var(--primary);
}
.card-accent { border-left-color: var(--accent); }
.card-gold   { border-left-color: #ffd700; background: #fffef0; }

/* Podium */
.p1 { background: linear-gradient(135deg,#fff7d6,#ffe66d);
      border: 2px solid #ffd700; border-radius:12px; padding:1rem 1.4rem; }
.p2 { background: linear-gradient(135deg,#f8f8f8,#e0e0e0);
      border: 2px solid #c0c0c0; border-radius:12px; padding:1rem 1.4rem; }
.p3 { background: linear-gradient(135deg,#fff3e0,#ffcc90);
      border: 2px solid #cd7f32; border-radius:12px; padding:1rem 1.4rem; }

/* Badges */
.badge { display:inline-block; padding:.2rem .7rem; border-radius:20px;
         font-size:.82rem; font-weight:600; margin:.12rem; }
.bg { background:#d4edda; color:#155724; }   /* green  */
.bo { background:#fff3cd; color:#856404; }   /* orange */
.bb { background:#d1ecf1; color:#0c5460; }   /* blue   */

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg,#1a6b3c,#2c9e5e) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: .7rem 2rem !important;
    font-size: 1.1rem !important; font-weight: 700 !important;
    box-shadow: 0 4px 12px rgba(26,107,60,.3) !important;
    width: 100% !important; transition: all .2s !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(26,107,60,.5) !important;
    transform: translateY(-1px) !important;
}

/* Combo boxes */
.combo {
    border-radius:8px; padding:.45rem 1rem; margin:.22rem 0;
    font-family: monospace; font-size:1.05rem; font-weight:700;
    border: 2px solid; color:#0d3320;
}

/* Metrics */
[data-testid="metric-container"] {
    background: white; border: 1px solid #e8f5ee;
    border-radius: 10px; padding: .7rem;
    box-shadow: 0 2px 8px rgba(0,0,0,.05);
}
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────
for key in ("df_cleaned", "df_scored", "pronostic", "raw_extractions", "done"):
    if key not in st.session_state:
        st.session_state[key] = None if key != "raw_extractions" else []
if "done" not in st.session_state:
    st.session_state.done = False

# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:.8rem 0'>
        <span style='font-size:3rem'>🏇</span>
        <h2 style='color:#1a6b3c;margin:.4rem 0'>PronoHippique AI</h2>
        <p style='color:#666;font-size:.82rem'>Pronostics intelligents par IA</p>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("### ⚙️ Moteur OCR")
    ocr_choice = st.radio(
        "Choisir le moteur",
        ["🤖 Google Gemini (Recommandé)", "🧠 OpenAI GPT-4o", "📷 EasyOCR (Local)"],
        help="Gemini et OpenAI donnent de bien meilleurs résultats sur les tableaux."
    )
    gemini_key = openai_key = ""
    if "Gemini" in ocr_choice:
        gemini_key = st.text_input("Clé API Gemini", type="password", placeholder="AIza...")
        if not gemini_key:
            st.info("💡 Sans clé → EasyOCR en fallback (précision réduite).")
    elif "OpenAI" in ocr_choice:
        openai_key = st.text_input("Clé API OpenAI", type="password", placeholder="sk-...")

    # Lire aussi depuis les secrets Streamlit Cloud
    if not gemini_key:
        gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    if not openai_key:
        openai_key = st.secrets.get("OPENAI_API_KEY", "")

    st.divider()
    st.markdown("### 🎯 Type de Course")
    race_type = st.selectbox(
        "Type",
        ["default", "quinté", "prix", "trot"],
        format_func=lambda x: {
            "default": "🏇 Standard",
            "quinté":  "🌟 Quinté+",
            "prix":    "🏆 Grand Prix",
            "trot":    "🏇 Trot Attelé",
        }[x],
    )
    st.divider()
    with st.expander("🔧 Options avancées"):
        show_raw  = st.checkbox("Afficher données brutes OCR", False)
        show_dtl  = st.checkbox("Détail des scores", True)

    st.markdown("""
    <div style='background:#f0f7f3;border-radius:10px;padding:.9rem;
                margin:.5rem 0;border-left:4px solid #2c9e5e;font-size:.82rem'>
        <strong>📋 Images supportées</strong><br>
        ✅ Liste des partants<br>✅ Records absolus<br>
        ✅ Stats drivers<br>✅ Stats entraîneurs
    </div>""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>🏇 PronoHippique AI</h1>
    <p>Intelligence Artificielle pour vos pronostics hippiques — Analysez · Scorez · Gagnez !</p>
</div>""", unsafe_allow_html=True)

# ── SECTION 1 : UPLOAD ───────────────────────────────────────────
st.markdown("## 📤 Étape 1 — Téléchargez vos captures d'écran")
st.markdown("""
<div class='card'>
    <h3>💡 Instructions</h3>
    <p>Téléchargez <strong>1 à 4 captures d'écran</strong> de statistiques hippiques de la même course :</p>
    <ul>
      <li>📊 <strong>Liste des partants</strong> — cotes, musique, gains</li>
      <li>🏆 <strong>Records absolus</strong> — meilleure performance chronométrique</li>
      <li>🏇 <strong>Statistiques drivers</strong> — courses, victoires, % réussite</li>
      <li>👨‍🏫 <strong>Statistiques entraîneurs</strong> — courses, victoires, % réussite</li>
    </ul>
    <p><em>Plus vous uploadez d'images complémentaires, plus l'analyse sera précise !</em></p>
</div>""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "📷 Glissez vos images ici ou cliquez pour sélectionner",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

if uploaded:
    st.markdown(f"**{len(uploaded)} image(s) chargée(s)** ✅")
    cols = st.columns(min(len(uploaded), 4))
    for i, f in enumerate(uploaded):
        with cols[i % 4]:
            img = Image.open(f)
            st.image(img, caption=f.name, use_column_width=True)

st.divider()

# ── SECTION 2 : BOUTON ANALYSER ──────────────────────────────────
st.markdown("## 🧠 Étape 2 — Lancer l'Analyse")
col_btn, col_msg = st.columns([2, 3])
with col_btn:
    clicked = st.button("🚀 Analyser la Course", use_container_width=True)
with col_msg:
    if not uploaded:
        st.warning("⚠️ Veuillez d'abord télécharger au moins une image.")
    elif "Gemini" in ocr_choice and not gemini_key:
        st.warning("⚠️ Aucune clé Gemini → EasyOCR utilisé (résultats moins précis).")
    else:
        st.success("✅ Prêt pour l'analyse !")

# ── TRAITEMENT ───────────────────────────────────────────────────
if clicked and uploaded:
    st.session_state.done = False
    progress = st.progress(0)
    status   = st.empty()
    extractions = []
    total_steps = len(uploaded) + 3

    for i, f in enumerate(uploaded):
        status.markdown(f"🔍 **OCR** — Image {i+1}/{len(uploaded)} : `{f.name}`...")
        progress.progress(int(i / total_steps * 100))
        img = Image.open(f).convert("RGB")
        result = extract_data_from_image(img, gemini_key, openai_key)
        extractions.append(result)
        time.sleep(0.15)

    st.session_state.raw_extractions = extractions

    progress.progress(int(len(uploaded) / total_steps * 100))
    status.markdown("🔀 **Fusion** des données extraites...")
    merged = merge_extracted_data(extractions)
    time.sleep(0.2)

    progress.progress(int((len(uploaded) + 1) / total_steps * 100))
    status.markdown("🧹 **Nettoyage** et structuration...")
    df_clean = clean_horse_data(merged.get("chevaux", []))
    st.session_state.df_cleaned = df_clean
    time.sleep(0.2)

    progress.progress(int((len(uploaded) + 2) / total_steps * 100))
    status.markdown("📊 **Calcul des scores**...")
    df_scored = calculate_scores(df_clean, race_type)
    st.session_state.df_scored = df_scored
    time.sleep(0.2)

    progress.progress(100)
    status.markdown("🎯 **Génération du pronostic**...")
    pronostic = generate_pronostic_report(df_scored)
    st.session_state.pronostic = pronostic
    st.session_state.done = True
    time.sleep(0.2)

    progress.empty()
    status.success("✅ Analyse terminée avec succès !")
    time.sleep(0.6)
    status.empty()
    st.rerun()

# ── RÉSULTATS ────────────────────────────────────────────────────
if st.session_state.done and st.session_state.df_scored is not None:
    df        = st.session_state.df_scored
    pronostic = st.session_state.pronostic
    n_part    = len(df)

    st.divider()
    st.markdown("## 📊 Résultats de l'Analyse")

    # Métriques rapides
    qual = assess_data_quality(df)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🐎 Partants", n_part)
    with c2:
        fav = pronostic.get("favori", {})
        st.metric("🏆 Favori IA", f"#{fav.get('numero','?')}", fav.get("cheval", "-"))
    with c3:
        st.metric("📈 Qualité données", f"{qual.get('qualite',0)}%")
    with c4:
        eng = (st.session_state.raw_extractions[0].get("ocr_engine", "?")
               if st.session_state.raw_extractions else "?")
        st.metric("🤖 Moteur OCR", eng.split(" ")[0])

    st.divider()

    # Onglets
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "🏆 Pronostic", "📊 Classement", "📈 Graphiques",
        "🔍 Données",  "🎰 Combinaisons", "📋 Détail Scores",
    ])

    # ── TAB 1 : PRONOSTIC ────────────────────────────────────────
    with t1:
        st.markdown("### 🎯 Pronostic Intelligent")
        gap       = pronostic.get("gap", 0)
        gauge_val = min(10.0, 5.0 + gap * 1.5)
        confiance = pronostic.get("confiance", "?")

        g_col, c_col = st.columns([1, 2])
        with g_col:
            st.plotly_chart(plot_gauge(round(gauge_val, 1)), use_container_width=True)
        with c_col:
            st.markdown(f"""
            <div class='card'>
                <h3>📌 Niveau de Confiance</h3>
                <p style='font-size:1.4rem;font-weight:700;color:#1a6b3c'>{confiance}</p>
                <p>Écart de score entre le favori IA et ses poursuivants : <strong>{gap:.2f} pts</strong><br>
                Plus l'écart est grand, plus le favori est dominant.</p>
                <p><strong>Partants analysés :</strong> {n_part}</p>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # Podium
        st.markdown("### 🥇 Top 3 Conseillé")
        top3 = pronostic.get("top3", [])
        pc = st.columns(3)
        styles  = ["p1", "p2", "p3"]
        medals  = ["🥇", "🥈", "🥉"]
        for i, horse in enumerate(top3[:3]):
            with pc[i]:
                st.markdown(f"""
                <div class='{styles[i]}'>
                    <div style='font-size:2rem;text-align:center'>{medals[i]}</div>
                    <h3 style='text-align:center;margin:.3rem 0'>
                        #{horse['numero']} {horse['cheval']}</h3>
                    <p style='text-align:center;font-size:1.2rem;
                              font-weight:700;color:#1a6b3c'>
                        Score : {horse['score_global']:.2f}/10</p>
                    <p style='text-align:center;font-size:.88rem'>
                        {horse.get('categorie','')}</p>
                </div>""", unsafe_allow_html=True)

        st.divider()

        # Bases & Outsiders
        bc, oc = st.columns(2)
        with bc:
            st.markdown("### 💎 2 Bases Solides")
            for h in pronostic.get("bases", []):
                st.markdown(f"""
                <div class='card card-gold'>
                    <strong>#{h['numero']} {h['cheval']}</strong>
                    <span class='badge bg'>Score : {h['score_global']:.2f}</span>
                </div>""", unsafe_allow_html=True)
        with oc:
            st.markdown("### 💡 Outsiders Intéressants")
            for h in pronostic.get("outsiders", []):
                cote_txt = f"Cote {h.get('cote_pmu','?')}" if h.get("cote_pmu", 0) > 0 else ""
                st.markdown(f"""
                <div class='card card-accent'>
                    <strong>#{h['numero']} {h['cheval']}</strong>
                    <span class='badge bo'>Score : {h['score_global']:.2f}</span>
                    {f"<span class='badge bb'>{cote_txt}</span>" if cote_txt else ""}
                </div>""", unsafe_allow_html=True)

        st.divider()

        # Arguments
        st.markdown("### 💬 Analyse Argumentée du Top 5")
        for hname, args in pronostic.get("arguments", {}).items():
            with st.expander(f"🏇 {hname}"):
                for a in args:
                    st.markdown(f"- {a}")

    # ── TAB 2 : CLASSEMENT ───────────────────────────────────────
    with t2:
        st.markdown("### 📊 Classement Complet")
        rows = []
        for _, row in df.sort_values("score_global", ascending=False).iterrows():
            rows.append({
                "Rang":      int(row.get("rang_score", 0)),
                "N°":        int(row.get("numero", 0)),
                "Cheval":    row.get("cheval", ""),
                "Score":     f"{row['score_global']:.2f}",
                "Catégorie": row.get("categorie", ""),
                "Driver":    row.get("driver", ""),
                "Entraîneur":row.get("entraineur", ""),
                "% Driver":  f"{row.get('reussite_driver',0):.0f}%",
                "% Entr.":   f"{row.get('reussite_entraineur',0):.0f}%",
                "Cote PMU":  row.get("cote_pmu", 0) if row.get("cote_pmu", 0) > 0 else "-",
                "Écart":     int(row.get("ecart_driver", 99)) if row.get("ecart_driver", 99) < 99 else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                     column_config={"Rang": st.column_config.NumberColumn("🥇", width="small")})
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Exporter CSV", csv, "pronostic_hippique.csv", "text/csv")

    # ── TAB 3 : GRAPHIQUES ───────────────────────────────────────
    with t3:
        st.markdown("### 📈 Visualisations")
        st.plotly_chart(plot_scores_bar(df), use_container_width=True)
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(plot_radar_top3(df), use_container_width=True)
        with g2:
            st.plotly_chart(plot_driver_comparison(df), use_container_width=True)
        st.plotly_chart(plot_form_history(df), use_container_width=True)

    # ── TAB 4 : DONNÉES ──────────────────────────────────────────
    with t4:
        st.markdown("### 🔍 Données Extraites")
        q = assess_data_quality(df)
        st.markdown(f"""
        <div class='card'>
            <strong>Qualité des données : {q.get('qualite',0)}%</strong>
            &nbsp;|&nbsp; Partants : {q.get('nb_chevaux',0)}
        </div>""", unsafe_allow_html=True)
        keep = ["numero","cheval","sa","driver","entraineur",
                "record_brut","reussite_driver","reussite_entraineur",
                "ecart_driver","gains","cote_pmu","musique"]
        st.dataframe(df[[c for c in keep if c in df.columns]],
                     use_container_width=True, hide_index=True)
        if show_raw:
            st.markdown("#### 📝 Réponses brutes OCR")
            for i, ext in enumerate(st.session_state.raw_extractions or []):
                with st.expander(f"Image {i+1} — OCR brut"):
                    st.json(ext)

    # ── TAB 5 : COMBINAISONS ─────────────────────────────────────
    with t5:
        st.markdown("### 🎰 Combinaisons de Paris")
        trios  = pronostic.get("trios",  [])
        quinte = pronostic.get("quintes",[])

        cc1, cc2 = st.columns(2)

        def _render_combos(combos, label, col):
            with col:
                st.markdown(f"#### {label}")
                for i, combo in enumerate(combos, 1):
                    nums_s = "  —  ".join(str(n) for n in sorted(combo))
                    if i == 1:   bg, border, em = "#fff7d6", "#ffd700", "🥇"
                    elif i == 2: bg, border, em = "#f8f8f8", "#c0c0c0", "🥈"
                    else:        bg, border, em = "#f0f7f3", "#2c9e5e", "▶️"
                    st.markdown(
                        f"<div class='combo' style='background:{bg};border-color:{border}'>"
                        f"{em} {label.split()[1]} {i} : [ {nums_s} ]</div>",
                        unsafe_allow_html=True,
                    )

        _render_combos(trios,  "🎯 Trio",    cc1)
        _render_combos(quinte, "🌟 Quinté+", cc2)

        st.divider()
        bases = pronostic.get("bases", [])
        if bases:
            bstr = " et ".join(f"**#{b['numero']} {b['cheval']}**" for b in bases)
            st.success(f"💎 Bases recommandées : {bstr}")
        outsiders = pronostic.get("outsiders", [])
        if outsiders:
            ostr = ", ".join(f"#{o['numero']}" for o in outsiders)
            st.info(f"💡 Outsiders à inclure : {ostr}")

    # ── TAB 6 : DÉTAIL SCORES ────────────────────────────────────
    with t6:
        st.markdown("### 📋 Détail des Scores par Critère")
        if show_dtl:
            for _, row in df.sort_values("score_global", ascending=False).head(10).iterrows():
                bd = get_score_breakdown(row)
                label = f"#{int(row['numero'])} {row['cheval']} — {row['score_global']:.2f}/10"
                with st.expander(label):
                    cols3 = st.columns(3)
                    for idx, (crit, sc) in enumerate(bd.items()):
                        with cols3[idx % 3]:
                            pct = int(sc / 10 * 100)
                            color = "#1a6b3c" if sc >= 7 else ("#f28a00" if sc >= 4 else "#e74c3c")
                            st.markdown(
                                f"<div style='margin:.25rem 0'>"
                                f"<small><b>{crit}</b></small>"
                                f"<div style='background:#e8f5ee;border-radius:4px;height:8px;margin:3px 0'>"
                                f"<div style='background:{color};width:{pct}%;height:8px;border-radius:4px'></div>"
                                f"</div><small>{sc:.1f}/10</small></div>",
                                unsafe_allow_html=True,
                            )
        else:
            st.info("Activez 'Détail des scores' dans la barre latérale.")

# ── FOOTER ───────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center;color:#888;font-size:.83rem;padding:.8rem'>
    🏇 <strong>PronoHippique AI</strong> — Développé avec Streamlit & IA Vision<br>
    <em>⚠️ Avertissement : Les pronostics sont fournis à titre indicatif uniquement.
    Le jeu peut créer une dépendance. Jouez de manière responsable.</em><br>
    <small>Joueurs Info Service : <strong>09 74 75 13 13</strong></small>
</div>
""", unsafe_allow_html=True)
Response
Created file /home/user/single_app/app.py (57109 characters)
