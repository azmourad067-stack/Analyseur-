"""
╔══════════════════════════════════════════════════════════════════╗
║       🏇  PronoHippique AI v2.0  —  Script complet unique        ║
║   Application Streamlit de pronostics hippiques intelligente     ║
║         Déployable directement sur Streamlit Cloud               ║
╚══════════════════════════════════════════════════════════════════╝

Améliorations v2.0 :
  ✅ Cache OCR (évite re-traitement coûteux)
  ✅ Validation robuste des entrées
  ✅ Algorithme de scoring affiné (11 critères)
  ✅ Gestion erreurs API améliorée
  ✅ Détection automatique du type d'image
  ✅ Export PDF/Excel
  ✅ Sauvegarde session (localStorage simulation)
  ✅ Mode "comparaison historique"
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
import hashlib
from datetime import datetime
from functools import lru_cache
from typing import Optional, Union

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🏇 PronoHippique AI",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "🏇 PronoHippique AI v2.0 — Pronostics hippiques par IA"
    }
)

# ══════════════════════════════════════════════════════════════════
#  CONSTANTES GLOBALES
# ══════════════════════════════════════════════════════════════════
APP_VERSION = "2.0.0"
MAX_IMAGE_SIZE_MB = 10
MAX_IMAGES_PER_ANALYSIS = 6
SUPPORTED_FORMATS = ["png", "jpg", "jpeg", "webp"]

# Pré-compilation des regex (gain de performance)
_RE_SA          = re.compile(r"^[A-Za-z]{1,2}\d{1,2}$")
_RE_DIST        = re.compile(r"^\d{4}$")
_RE_PERSON_INIT = re.compile(r"^[A-ZÀ-Ü][a-zA-ZÀ-Ü]{0,2}[.\-]")
_RE_PERSON_FULL = re.compile(r"^[A-ZÀ-Ü][a-zà-ü]{2,}")
_RE_CAP         = re.compile(r"^[A-ZÀ-Ü][a-zà-ü]")
_RE_NUMBER_3    = re.compile(r"^\d{3}$")
_RE_INITIAL_1   = re.compile(r"^[A-Z]\.$")
_RE_INITIAL_C   = re.compile(r"^[A-ZÀ-Ü]\.-[A-ZÀ-Ü]\.$")
_RE_INITIAL_S   = re.compile(r"^[A-ZÀ-Ü][a-zA-ZÀ-Ü]{0,2}\.$")
_RE_GAINS       = re.compile(r"^\d{5,7}$")
_RE_COTE        = re.compile(r"^(\d{1,3}(?:\.\d)?)$")


# ══════════════════════════════════════════════════════════════════
#  MODULE 1 — OCR EXTRACTOR (avec cache)
# ══════════════════════════════════════════════════════════════════

def _hash_image(image: Image.Image) -> str:
    """Génère un hash unique d'une image pour cache."""
    buf = io.BytesIO()
    image.resize((128, 128)).save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()


def _encode_image_base64(image: Image.Image, max_size: int = 1600) -> str:
    """Encode une image PIL en base64 PNG, avec redimensionnement si trop grosse."""
    # Réduit la taille si nécessaire (économise des tokens API)
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _build_extraction_prompt() -> str:
    """Prompt universel pour l'extraction structurée des tableaux hippiques."""
    return """Tu es un expert en analyse de tableaux de statistiques hippiques françaises.
Analyse cette image et extrais TOUTES les données visibles sous forme JSON structuré.

Détermine d'abord le TYPE de tableau parmi :
- "partants"          : liste des partants (N°, Cheval, SA, Dist, Driver, Entraîneur, Musique, Gains, Cote_PMU, Cote_Genybet)
- "records"           : records absolus (N°, Cheval, SA, Dist, Driver, Record, Date_Record)
- "stats_drivers"     : statistiques drivers PMU
- "stats_entraineurs" : statistiques entraîneurs PMU
- "unknown"           : si non reconnu

Réponds UNIQUEMENT avec un JSON valide, sans markdown, sans explication.
Format attendu :
{
  "table_type": "partants",
  "nb_partants": 15,
  "chevaux": [
    {
      "numero": 1, "cheval": "NomDuCheval", "sa": "M7", "dist": 2100,
      "driver": "M. Mottier", "entraineur": "J. Westholm",
      "musique": "(25)1aDaDa", "gains": 219481,
      "cote_pmu": 1.9, "cote_genybet": 2.1,
      "record": "1'10\"0", "date_record": "08/05/25",
      "courses_driver": 1288, "victoires_driver": 213,
      "ecart_driver": 4, "reussite_driver": 16,
      "musique_driver": "Da8a9aDa1aDaDa4m",
      "courses_entraineur": 187, "victoires_entraineur": 31,
      "ecart_entraineur": 7, "reussite_entraineur": 16,
      "musique_entraineur": "3a0aDa8a5a6a2a1a"
    }
  ]
}

Notes :
- Champs absents = null
- Musique : chiffre=position, D=distancé, m/a=disqualifié, (25)=non partant
- Pourcentages = valeur numérique (16 pour 16%)
- Cotes : virgule décimale française acceptée
- Extrais TOUS les chevaux visibles"""


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
    # Réparer les virgules décimales françaises
    json_str = re.sub(r'("cote_[^"]+"\s*:\s*)(\d+),(\d+)', r'\1\2.\3', json_str)
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    try:
        data = json.loads(json_str)
        data["raw_text"] = raw_text
        return data
    except json.JSONDecodeError as e:
        return {"error": f"JSON invalide : {e}", "raw_text": raw_text}


@st.cache_data(show_spinner=False, ttl=3600, max_entries=50)
def _cached_extract_gemini(image_hash: str, image_bytes: bytes, api_key: str) -> dict:
    """Cache l'extraction Gemini pour éviter les appels répétés."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        img = Image.open(io.BytesIO(image_bytes))
        response = model.generate_content(
            [_build_extraction_prompt(), img],
            generation_config={"temperature": 0.1, "max_output_tokens": 4096}
        )
        result = _parse_json_response(response.text)
        result["ocr_engine"] = "Gemini Vision"
        return result
    except Exception as e:
        err = str(e)
        # Messages d'erreur user-friendly
        if "API_KEY" in err or "invalid" in err.lower():
            err = "Clé API Gemini invalide ou expirée"
        elif "quota" in err.lower():
            err = "Quota Gemini dépassé"
        elif "PERMISSION_DENIED" in err:
            err = "Permission refusée — vérifiez votre clé API"
        return {"error": err, "ocr_engine": "Gemini Vision"}


def extract_with_gemini(image: Image.Image, api_key: str) -> dict:
    """Extraction via Google Gemini Vision avec cache."""
    if not api_key or len(api_key.strip()) < 10:
        return {"error": "Clé API Gemini manquante", "ocr_engine": "Gemini Vision"}
    img_hash = _hash_image(image)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return _cached_extract_gemini(img_hash, buf.getvalue(), api_key)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=50)
def _cached_extract_openai(image_hash: str, image_bytes: bytes, api_key: str) -> dict:
    """Cache l'extraction OpenAI."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, timeout=60)
        img = Image.open(io.BytesIO(image_bytes))
        img_b64 = _encode_image_base64(img)
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
            temperature=0.1,
        )
        result = _parse_json_response(response.choices[0].message.content)
        result["ocr_engine"] = "OpenAI GPT-4o"
        return result
    except Exception as e:
        err = str(e)
        if "Incorrect API key" in err or "invalid_api_key" in err:
            err = "Clé API OpenAI invalide"
        elif "rate_limit" in err.lower():
            err = "Limite de débit OpenAI atteinte"
        elif "insufficient_quota" in err:
            err = "Quota OpenAI épuisé"
        return {"error": err, "ocr_engine": "OpenAI GPT-4o"}


def extract_with_openai(image: Image.Image, api_key: str) -> dict:
    """Extraction via OpenAI GPT-4o avec cache."""
    if not api_key or not api_key.startswith("sk-"):
        return {"error": "Clé API OpenAI invalide", "ocr_engine": "OpenAI GPT-4o"}
    img_hash = _hash_image(image)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return _cached_extract_openai(img_hash, buf.getvalue(), api_key)


def _is_musique_token(p: str) -> bool:
    if _RE_SA.match(p) or _RE_DIST.match(p):
        return False
    return bool(re.search(r"\(\d+\)|[DdMmAa]", p) and re.search(r"\d", p) and len(p) >= 3)


def _is_person_name_token(p: str) -> bool:
    if len(p) < 3 or _RE_SA.match(p) or _RE_DIST.match(p):
        return False
    if _is_musique_token(p):
        return False
    if re.match(r"^\d", p):
        return False
    if p.strip().lower() in ("non-partant", "non partant", "partant", "absent", "np",
                              "capture", "d'écran", "png", "jpg"):
        return False
    if _RE_PERSON_INIT.match(p):
        return True
    if _RE_PERSON_FULL.match(p):
        return True
    return False


def _smart_tokenize_ocr_line(line: str) -> list:
    """Tokenise une ligne OCR en tokens sémantiques."""
    line = re.sub(r"\s+", " ", line.strip())
    if "|" in line:
        return [p.strip() for p in line.split("|") if p.strip()]

    LINKING = {"du", "de", "des", "le", "la", "les", "au", "aux", "d'"}
    words  = line.split(" ")
    tokens = []
    i = 0
    while i < len(words):
        w    = words[i]
        nxt  = words[i+1] if i+1 < len(words) else ""
        nxt2 = words[i+2] if i+2 < len(words) else ""

        if _RE_NUMBER_3.match(w) and _RE_NUMBER_3.match(nxt):
            tokens.append(w + nxt); i += 2; continue
        if _RE_INITIAL_1.match(w) and _RE_INITIAL_1.match(nxt) and nxt2:
            tokens.append(w + " " + nxt + " " + nxt2); i += 3; continue
        if _RE_INITIAL_C.match(w) and nxt and not re.match(r"^\d", nxt):
            tokens.append(w + " " + nxt); i += 2; continue
        if _RE_INITIAL_S.match(w) and nxt and not re.match(r"^\d", nxt):
            tokens.append(w + " " + nxt); i += 2; continue

        if (_RE_CAP.match(w) and not _RE_SA.match(w) and not _RE_DIST.match(w)
                and not _is_musique_token(w)):
            parts = [w]
            j = i + 1
            while j < len(words):
                nw = words[j]
                if _RE_SA.match(nw) or _RE_DIST.match(nw) or _is_musique_token(nw): break
                if re.match(r"^\d", nw): break
                if not (_RE_CAP.match(nw) or nw.lower() in LINKING): break
                parts.append(nw)
                j += 1
            if len(parts) >= 2:
                tokens.append(" ".join(parts)); i = j; continue

        tokens.append(w); i += 1
    return tokens


def _parse_horse_from_tokens(toks: list) -> Optional[dict]:
    """Extrait les données d'un cheval depuis des tokens."""
    if not toks:
        return None
    num_match = re.match(r"^(\d{1,2})\b", toks[0])
    if not num_match:
        return None
    numero = int(num_match.group(1))
    if not (1 <= numero <= 20):
        return None

    horse = {"numero": numero}
    if len(toks) > 1 and len(toks[1]) >= 2:
        horse["cheval"] = toks[1]

    for t in toks[2:7]:
        if _RE_SA.match(t):
            horse["sa"] = t; break
    for t in toks[2:8]:
        if _RE_DIST.match(t):
            horse["dist"] = int(t); break
    for t in toks:
        if _is_musique_token(t):
            horse["musique"] = t; break

    names = [t for t in toks[2:] if _is_person_name_token(t)]
    if names:
        horse["driver"] = names[0]
    if len(names) >= 2:
        horse["entraineur"] = names[1]

    if any(t.lower() in ("non-partant", "non partant") for t in toks):
        horse["non_partant"] = True

    for t in toks:
        c = re.sub(r"\s", "", t)
        if _RE_GAINS.match(c):
            horse["gains"] = float(c); break

    for t in reversed(toks):
        if t in ("-", "—", ""):
            continue
        cs = t.replace(",", ".")
        m = _RE_COTE.match(cs)
        if m:
            v = float(m.group(1))
            if 1.0 <= v <= 150.0:
                horse["cote_pmu"] = v; break

    return horse if horse.get("cheval") else None


def _parse_easyocr_lines_to_chevaux(lines: list) -> list:
    """Parse les lignes EasyOCR en chevaux."""
    chevaux = []
    seen_nums = set()
    for raw_line in lines:
        toks = _smart_tokenize_ocr_line(raw_line)
        h    = _parse_horse_from_tokens(toks)
        if h and h["numero"] not in seen_nums:
            seen_nums.add(h["numero"])
            chevaux.append(h)
    return sorted(chevaux, key=lambda x: x["numero"])


@st.cache_resource(show_spinner=False)
def _get_easyocr_reader():
    """Singleton EasyOCR (chargement lourd, on évite les re-créations)."""
    import easyocr
    return easyocr.Reader(["fr", "en"], gpu=False, verbose=False)


def extract_with_easyocr(image: Image.Image) -> dict:
    """Extraction via EasyOCR (fallback local) avec reader cached."""
    try:
        import numpy as np_local
        reader = _get_easyocr_reader()
        img_array = np_local.array(image)
        results = reader.readtext(img_array, detail=1)

        lines_by_y = {}
        for bbox, text, conf in results:
            if conf < 0.25:
                continue
            y_center = int((bbox[0][1] + bbox[2][1]) / 2)
            y_bucket = (y_center // 12) * 12
            lines_by_y.setdefault(y_bucket, []).append((bbox[0][0], text.strip()))

        lines = []
        for y in sorted(lines_by_y):
            items = sorted(lines_by_y[y], key=lambda x: x[0])
            line_text = " | ".join(t for _, t in items)
            lines.append(line_text)

        raw_text = "\n".join(lines)
        chevaux = _parse_easyocr_lines_to_chevaux(lines)
        return {
            "type": "raw_ocr",
            "raw_text": raw_text,
            "lines": lines,
            "ocr_engine": "EasyOCR (mode local)",
            "chevaux": chevaux,
            "table_type": "partants" if chevaux else "unknown",
        }
    except ImportError:
        return {
            "error": "EasyOCR non installé (pip install easyocr)",
            "ocr_engine": "EasyOCR",
            "chevaux": [], "table_type": "unknown",
        }
    except Exception as e:
        return {
            "error": str(e), "ocr_engine": "EasyOCR",
            "chevaux": [], "table_type": "unknown",
        }


def extract_data_from_image(
    image: Image.Image,
    gemini_api_key: str = "",
    openai_api_key: str = "",
    preferred: str = "auto",
) -> dict:
    """Orchestre l'extraction avec ordre de priorité configurable."""
    engines = []
    if preferred == "gemini" or preferred == "auto":
        if gemini_api_key:
            engines.append(("gemini", lambda: extract_with_gemini(image, gemini_api_key)))
    if preferred == "openai" or preferred == "auto":
        if openai_api_key:
            engines.append(("openai", lambda: extract_with_openai(image, openai_api_key)))
    engines.append(("easyocr", lambda: extract_with_easyocr(image)))

    last_result = None
    for name, fn in engines:
        result = fn()
        last_result = result
        if result.get("chevaux"):
            return result
    return last_result or {"error": "Aucun moteur OCR disponible"}


def merge_extracted_data(extractions: list) -> dict:
    """Fusionne les données extraites de plusieurs images."""
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
                continue
            if num not in merged:
                merged[num] = {"numero": num}
            for key, val in horse.items():
                if val is not None and val != "" and key != "numero":
                    if key not in merged[num] or merged[num][key] is None or merged[num][key] == "":
                        merged[num][key] = val
    chevaux_list = sorted(merged.values(), key=lambda x: x.get("numero", 99))
    return {
        "chevaux": chevaux_list,
        "nb_partants": len(chevaux_list),
        "table_types_detectes": list(set(table_types)),
    }


# ══════════════════════════════════════════════════════════════════
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
        r"(\d+)'(\d+)\"(\d+)",
        r"(\d+)'(\d+)\.(\d+)",
        r"(\d+)'(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, record)
        if m:
            g = m.groups()
            minutes = int(g[0])
            seconds = int(g[1])
            tenths  = int(g[2]) / 10 if len(g) > 2 else 0.0
            total = minutes * 60 + seconds + tenths
            # Validation : un record hippique réaliste se situe entre 50s et 200s
            if 50 <= total <= 200:
                return total
    return 0.0


def _parse_cote(val) -> float:
    if val is None:
        return 0.0
    try:
        s = str(val).replace(",", ".").replace(" ", "").strip()
        m = re.search(r"[\d.]+", s)
        if not m:
            return 0.0
        v = float(m.group())
        # Validation : cote PMU réaliste entre 1.0 et 999.0
        return v if 1.0 <= v <= 999.0 else 0.0
    except Exception:
        return 0.0


def _parse_pct(val) -> float:
    if val is None:
        return 0.0
    try:
        s = str(val).replace("%", "").replace(",", ".").strip()
        m = re.search(r"[\d.]+", s)
        if not m:
            return 0.0
        v = float(m.group())
        return min(100.0, max(0.0, v))  # Clamp 0-100
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
        v = int(s)
        return min(999, max(0, v))
    except ValueError:
        return 99


def clean_horse_data(chevaux_raw: list) -> pd.DataFrame:
    """Nettoie et structure la liste des chevaux en DataFrame."""
    if not chevaux_raw:
        return pd.DataFrame()
    cleaned = []
    for h in chevaux_raw:
        # Filtre les chevaux non-partants
        if h.get("non_partant"):
            continue
        sa_str = _clean_str(h.get("sa", ""))
        c = {
            "numero":               _safe_int(h.get("numero")),
            "cheval":               _clean_str(h.get("cheval", "")),
            "sa":                   sa_str,
            "sexe":                 _extract_sexe(sa_str),
            "age":                  _extract_age(sa_str),
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
        if not c["musique"] and c["musique_driver"]:
            c["musique"] = c["musique_driver"]
        # Validation : il faut au minimum un numéro et un nom
        if c["numero"] > 0 and c["cheval"]:
            cleaned.append(c)

    df = pd.DataFrame(cleaned)
    if not df.empty and "numero" in df.columns:
        df = df.drop_duplicates(subset=["numero"], keep="first")
        df = df.sort_values("numero").reset_index(drop=True)
    return df


# ── Décodeur de musique amélioré ─────────────────────────────────

def decode_musique(musique: str) -> list:
    """Décode la musique hippique en liste de résultats."""
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
            if p == 0:    results.append({"pos": 0, "type": "non_classé",     "score_base": 0.3})
            elif p == 1:  results.append({"pos": 1, "type": "victoire",        "score_base": 10.0})
            elif p == 2:  results.append({"pos": 2, "type": "placé",           "score_base": 7.0})
            elif p == 3:  results.append({"pos": 3, "type": "placé",           "score_base": 5.0})
            elif p == 4:  results.append({"pos": 4, "type": "proche_podium",   "score_base": 3.5})
            elif p == 5:  results.append({"pos": 5, "type": "proche_podium",   "score_base": 2.5})
            elif p <= 7:  results.append({"pos": p, "type": "milieu",          "score_base": 1.5})
            else:         results.append({"pos": p, "type": "arrière",         "score_base": 0.5})
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


def count_disqualifications(musique: str, n: int = 5) -> int:
    """Compte les disqualifications/distancements récents (signal de risque)."""
    results = decode_musique(musique)
    return sum(1 for r in results[-n:] if r.get("type") in ("distancé", "disqualifié"))


def calc_consistency(musique: str, n: int = 8) -> float:
    """Mesure la régularité (faible écart-type des positions)."""
    results = decode_musique(musique)
    if len(results) < 3:
        return 0.0
    positions = [r["pos"] if r.get("pos") is not None else 15 for r in results[-n:]]
    if not positions:
        return 0.0
    mean = np.mean(positions)
    std  = np.std(positions)
    # Plus la moyenne est basse et l'écart-type faible, meilleure la régularité
    score = 10.0 - mean - (std * 0.5)
    return max(0.0, min(10.0, score))


def assess_data_quality(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"qualite": 0, "nb_chevaux": 0, "details": {}}
    total = len(df)
    scores = {}
    fields_check = {
        "cheval":             "Nom",
        "numero":             "Numéro",
        "driver":             "Driver",
        "entraineur":         "Entraîneur",
        "musique":            "Musique",
        "reussite_driver":    "% Driver",
        "record_secondes":    "Record",
        "gains":              "Gains",
        "cote_pmu":           "Cote",
    }
    for field, label in fields_check.items():
        if field in df.columns:
            filled = df[field].notna() & (df[field] != "") & (df[field] != 0)
            scores[label] = round(filled.sum() / total * 100, 1)
    quality = round(sum(scores.values()) / len(scores), 1) if scores else 0
    return {"qualite": round(quality), "nb_chevaux": total, "details": scores}


# ══════════════════════════════════════════════════════════════════
#  MODULE 3 — SCORER (algo amélioré)
# ══════════════════════════════════════════════════════════════════

WEIGHTS = {
    "record_absolu":        0.16,
    "musique_recente":      0.20,
    "reussite_driver":      0.11,
    "reussite_entraineur":  0.09,
    "ecart":                0.09,
    "gains":                0.08,
    "victoires_driver":     0.06,
    "cote_inverse":         0.10,
    "regularite":           0.06,
    "consistency":          0.03,  # NOUVEAU
    "penalite_disq":        0.02,  # NOUVEAU
}

WEIGHTS_QUINTE = {**WEIGHTS, "musique_recente": 0.22, "record_absolu": 0.15}
WEIGHTS_PRIX   = {**WEIGHTS, "record_absolu": 0.20,   "musique_recente": 0.18}
WEIGHTS_TROT   = {**WEIGHTS, "regularite": 0.09,      "ecart": 0.11}

RACE_WEIGHTS = {
    "quinté":  WEIGHTS_QUINTE,
    "prix":    WEIGHTS_PRIX,
    "trot":    WEIGHTS_TROT,
    "default": WEIGHTS,
}


def _normalize(series: pd.Series) -> pd.Series:
    col = series.fillna(0).astype(float)
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


def _score_disqualif_penalty(musique: str) -> float:
    """Score 0-10 inversé : 10 = aucune disqualif récente, 0 = beaucoup."""
    n_disq = count_disqualifications(musique, 5)
    return max(0.0, 10.0 - n_disq * 3.0)


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
    """Calcule les scores et retourne le DataFrame enrichi."""
    if df.empty:
        return df
    df = df.copy()
    W = RACE_WEIGHTS.get(race_type, WEIGHTS)

    for col in ("musique", "musique_driver", "musique_entraineur"):
        if col not in df.columns:
            df[col] = ""

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
    df["disq_recents"]               = df["musique"].apply(lambda m: count_disqualifications(str(m) if m else ""))
    df["score_regularite"]           = ((df["wins_recents"] * 2 + df["places_recents"]) / 15.0 * 10.0).clip(0, 10)
    df["score_consistency"]          = df["musique"].apply(lambda m: calc_consistency(str(m) if m else ""))
    df["score_penalite_disq"]        = df["musique"].apply(lambda m: _score_disqualif_penalty(str(m) if m else ""))

    # Score global pondéré
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
        df["score_regularite"]          * W["regularite"]          +
        df["score_consistency"]         * W.get("consistency", 0)  +
        df["score_penalite_disq"]       * W.get("penalite_disq", 0)
    ) / total_w

    score_cols = [c for c in df.columns if c.startswith("score_")]
    df[score_cols] = df[score_cols].round(2)
    df["rang_score"] = df["score_global"].rank(ascending=False, method="min").astype(int)
    df["categorie"]  = df.apply(_categorize, axis=1)

    # Probabilité estimée (softmax des scores)
    exp_scores = np.exp((df["score_global"] - df["score_global"].max()) / 1.5)
    df["proba_victoire"] = (exp_scores / exp_scores.sum() * 100).round(1)

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
        "📊 Constance":             round(row.get("score_consistency", 0), 2),
        "⚠️ Fiabilité":             round(row.get("score_penalite_disq", 0), 2),
    }


# ══════════════════════════════════════════════════════════════════
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

    combos.add(tuple(sorted(nums[:3])))
    for i in range(2, min(7, len(nums))):
        combos.add(tuple(sorted([nums[0], nums[1], nums[i]])))
    for i in range(2, min(5, len(nums))):
        for j in range(i + 1, min(7, len(nums))):
            combos.add(tuple(sorted([nums[0], nums[i], nums[j]])))

    pool = min(10, len(nums))
    total_s = sum(scores[:pool]) or 1
    w_norm = [s / total_s for s in scores[:pool]]
    rng = np.random.default_rng(seed=42)
    attempts = 0
    while len(combos) < n and attempts < 1000:
        attempts += 1
        idx = rng.choice(range(pool), size=3, replace=False, p=w_norm[:pool])
        combos.add(tuple(sorted([nums[i] for i in idx])))

    return [list(c) for c in list(combos)[:n]]


def generate_quinte_combinations(df: pd.DataFrame, n: int = 10) -> list:
    """Génère n combinaisons Quinté+."""
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
    rng = np.random.default_rng(seed=42)
    attempts = 0
    while len(combos) < n and attempts < 2000:
        attempts += 1
        idx = rng.choice(range(pool), size=5, replace=False, p=w_norm[:pool])
        combos.add(tuple(sorted([nums[i] for i in idx])))

    return [list(c) for c in list(combos)[:n]]


def _build_arguments(row: pd.Series) -> list:
    args = []
    if row.get("record_secondes", 0) > 0:
        sec = row["record_secondes"]
        args.append(f"⏱️ Record : {int(sec//60)}'{sec%60:.1f}\" — Vitesse pure élevée")
    rd = row.get("reussite_driver", 0)
    if rd >= 20:   args.append(f"🏇 Driver en pleine forme ({rd:.0f}% de réussite)")
    elif rd >= 15: args.append(f"🏇 Driver en forme ({rd:.0f}% de réussite)")
    elif rd >= 10: args.append(f"🏇 Driver compétent ({rd:.0f}% de réussite)")
    re_ = row.get("reussite_entraineur", 0)
    if re_ >= 20:   args.append(f"🎯 Entraîneur top ({re_:.0f}% de réussite)")
    elif re_ >= 15: args.append(f"🎯 Entraîneur excellent ({re_:.0f}% de réussite)")
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
    disq = row.get("disq_recents", 0)
    if disq >= 2:
        args.append(f"⚠️ {disq} disqualifications/distancements récents — risque !")
    cote = row.get("cote_pmu", 0)
    if cote > 0:
        if cote <= 3:   args.append(f"💰 Grand favori PMU (cote {cote})")
        elif cote <= 7: args.append(f"💰 Favori PMU (cote {cote})")
        elif cote >= 50:args.append(f"🎲 Longshot potentiellement intéressant (cote {cote})")
    proba = row.get("proba_victoire", 0)
    if proba > 0:
        args.append(f"📈 Probabilité estimée de podium : {proba:.1f}%")
    if not args:
        args.append("📋 Données partielles — analyse limitée")
    return args


def generate_pronostic_report(df: pd.DataFrame) -> dict:
    """Génère le rapport de pronostic complet."""
    if df.empty:
        return {"error": "Aucune donnée"}
    sdf = df.sort_values("score_global", ascending=False).reset_index(drop=True)

    gap = float(sdf.iloc[0]["score_global"] - sdf.iloc[1]["score_global"]) if len(sdf) > 1 else 0
    if gap > 2.0:   confiance = "Haute 🔥"
    elif gap > 1.0: confiance = "Moyenne ⭐"
    elif gap > 0.5: confiance = "Modérée 📊"
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
                            "cote_pmu", "proba_victoire"]].to_dict("records"),
        "top3":      sdf.head(3)[["numero", "cheval", "score_global", "categorie", "proba_victoire"]].to_dict("records"),
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
            "proba":  round(float(sdf.iloc[0].get("proba_victoire", 0)), 1),
        },
        "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
    }


# ══════════════════════════════════════════════════════════════════
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
        xaxis=dict(title="Score (sur 10)",
                   range=[0, max(scores) * 1.15] if scores else [0, 10],
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


def plot_proba_pie(df: pd.DataFrame) -> go.Figure:
    """Camembert des probabilités de victoire des Top 6."""
    sdf = df.sort_values("score_global", ascending=False).head(6).copy()
    others_proba = max(0, 100 - sdf["proba_victoire"].sum())
    labels = [f"#{int(r['numero'])} {r['cheval']}" for _, r in sdf.iterrows()]
    values = sdf["proba_victoire"].tolist()
    if others_proba > 0:
        labels.append("Autres")
        values.append(others_proba)
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.4,
        marker=dict(colors=["#ffd700", "#c0c0c0", "#cd7f32", "#2c9e5e",
                            "#5ab87e", "#90d4a8", "#cccccc"]),
        textinfo="label+percent",
    ))
    fig.update_layout(
        title=dict(text="🎲 Probabilité estimée par cheval",
                   font=dict(size=16, color=_C["dark"]), x=0.5),
        paper_bgcolor="white", height=400,
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
#  EXPORT UTILITIES
# ══════════════════════════════════════════════════════════════════

def export_to_excel(df: pd.DataFrame, pronostic: dict) -> bytes:
    """Exporte le résultat complet en Excel multi-onglets."""
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            # Feuille 1 : Classement
            df_sorted = df.sort_values("score_global", ascending=False)
            keep = ["rang_score", "numero", "cheval", "score_global",
                    "proba_victoire", "categorie", "driver", "entraineur",
                    "reussite_driver", "reussite_entraineur",
                    "cote_pmu", "musique", "ecart_driver", "gains"]
            df_sorted[[c for c in keep if c in df_sorted.columns]].to_excel(
                writer, sheet_name="Classement", index=False)

            # Feuille 2 : Combinaisons
            trios  = pronostic.get("trios", [])
            quinte = pronostic.get("quintes", [])
            combo_df = pd.DataFrame({
                "Trio_N°":  list(range(1, len(trios) + 1)),
                "Trio":     [" - ".join(map(str, c)) for c in trios],
            })
            combo_df.to_excel(writer, sheet_name="Trios", index=False)

            if quinte:
                qdf = pd.DataFrame({
                    "Quinté_N°": list(range(1, len(quinte) + 1)),
                    "Quinté":    [" - ".join(map(str, c)) for c in quinte],
                })
                qdf.to_excel(writer, sheet_name="Quintes", index=False)

            # Feuille 3 : Détails scores
            score_cols = [c for c in df.columns if c.startswith("score_")]
            df_scores = df[["numero", "cheval"] + score_cols].sort_values(
                "score_global", ascending=False)
            df_scores.to_excel(writer, sheet_name="Détails_Scores", index=False)

        return buf.getvalue()
    except Exception as e:
        st.error(f"Erreur export Excel : {e}")
        return b""


def export_to_text_report(df: pd.DataFrame, pronostic: dict) -> str:
    """Génère un rapport texte complet."""
    lines = [
        "═" * 60,
        "🏇 PRONOHIPPIQUE AI — RAPPORT DE PRONOSTIC",
        "═" * 60,
        f"📅 Généré le : {pronostic.get('timestamp', 'N/A')}",
        f"🐎 Partants analysés : {pronostic.get('nb_partants', 0)}",
        f"📊 Niveau de confiance : {pronostic.get('confiance', '?')}",
        f"📏 Écart favori/dauphin : {pronostic.get('gap', 0):.2f} pts",
        "",
        "─" * 60,
        "🏆 PODIUM IA",
        "─" * 60,
    ]
    for i, h in enumerate(pronostic.get("top3", []), 1):
        medal = ["🥇", "🥈", "🥉"][i-1]
        lines.append(f"{medal} #{h['numero']} {h['cheval']}")
        lines.append(f"   Score: {h['score_global']:.2f}/10  |  "
                     f"Proba: {h.get('proba_victoire', 0):.1f}%  |  "
                     f"{h.get('categorie', '')}")
    lines.append("")
    lines.append("─" * 60)
    lines.append("💎 BASES & 💡 OUTSIDERS")
    lines.append("─" * 60)
    for h in pronostic.get("bases", []):
        lines.append(f"💎 BASE: #{h['numero']} {h['cheval']} (Score: {h['score_global']:.2f})")
    for h in pronostic.get("outsiders", []):
        lines.append(f"💡 OUTSIDER: #{h['numero']} {h['cheval']} "
                     f"(Score: {h['score_global']:.2f}, Cote: {h.get('cote_pmu', '?')})")
    lines.append("")
    lines.append("─" * 60)
    lines.append("🎯 COMBINAISONS TRIO RECOMMANDÉES")
    lines.append("─" * 60)
    for i, c in enumerate(pronostic.get("trios", [])[:5], 1):
        lines.append(f"  Trio {i}: {' - '.join(map(str, c))}")
    lines.append("")
    lines.append("🌟 COMBINAISONS QUINTÉ+ RECOMMANDÉES")
    for i, c in enumerate(pronostic.get("quintes", [])[:5], 1):
        lines.append(f"  Quinté {i}: {' - '.join(map(str, c))}")
    lines.append("")
    lines.append("═" * 60)
    lines.append("⚠️ Les jeux d'argent comportent des risques. Jouez avec modération.")
    lines.append("═" * 60)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
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
.main-header {
    background: linear-gradient(135deg, #0d3320 0%, #1a6b3c 55%, #2c9e5e 100%);
    color: white; padding: 2rem 2.5rem; border-radius: 16px;
    text-align: center; margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(13,51,32,.4);
}
.main-header h1 { font-size: 2.7rem; margin: 0; letter-spacing: 2px; }
.main-header p  { font-size: 1.05rem; margin: .5rem 0 0; opacity: .88; }
.card {
    background: white; border-radius: 12px; padding: 1.4rem;
    box-shadow: 0 4px 16px rgba(0,0,0,.07); margin-bottom: .9rem;
    border-left: 5px solid var(--primary);
}
.card-accent { border-left-color: var(--accent); }
.card-gold   { border-left-color: #ffd700; background: #fffef0; }
.p1 { background: linear-gradient(135deg,#fff7d6,#ffe66d);
      border: 2px solid #ffd700; border-radius:12px; padding:1rem 1.4rem; }
.p2 { background: linear-gradient(135deg,#f8f8f8,#e0e0e0);
      border: 2px solid #c0c0c0; border-radius:12px; padding:1rem 1.4rem; }
.p3 { background: linear-gradient(135deg,#fff3e0,#ffcc90);
      border: 2px solid #cd7f32; border-radius:12px; padding:1rem 1.4rem; }
.badge { display:inline-block; padding:.2rem .7rem; border-radius:20px;
         font-size:.82rem; font-weight:600; margin:.12rem; }
.bg { background:#d4edda; color:#155724; }
.bo { background:#fff3cd; color:#856404; }
.bb { background:#d1ecf1; color:#0c5460; }
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
.combo {
    border-radius:8px; padding:.45rem 1rem; margin:.22rem 0;
    font-family: monospace; font-size:1.05rem; font-weight:700;
    border: 2px solid; color:#0d3320;
}
[data-testid="metric-container"] {
    background: white; border: 1px solid #e8f5ee;
    border-radius: 10px; padding: .7rem;
    box-shadow: 0 2px 8px rgba(0,0,0,.05);
}
.disclaimer {
    background:#fff3cd; border-left: 4px solid #f39c12;
    padding:.8rem 1.1rem; border-radius:8px;
    font-size:.85rem; color:#856404; margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init (centralisé) ──────────────────────────────
def init_session_state():
    defaults = {
        "df_cleaned": None,
        "df_scored": None,
        "pronostic": None,
        "raw_extractions": [],
        "done": False,
        "analysis_count": 0,
        "last_analysis_time": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:.8rem 0'>
        <span style='font-size:3rem'>🏇</span>
        <h2 style='color:#1a6b3c;margin:.4rem 0'>PronoHippique AI</h2>
        <p style='color:#666;font-size:.82rem'>v{APP_VERSION} • Pronostics par IA</p>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("### ⚙️ Moteur OCR")
    ocr_choice = st.radio(
        "Choisir le moteur",
        ["🤖 Google Gemini (Recommandé)", "🧠 OpenAI GPT-4o", "📷 EasyOCR (Local)"],
        help="Gemini et OpenAI donnent de bien meilleurs résultats."
    )
    gemini_key = openai_key = ""
    if "Gemini" in ocr_choice:
        gemini_key = st.text_input("Clé API Gemini", type="password",
                                    placeholder="AIza...",
                                    help="Obtenez une clé sur aistudio.google.com")
        if gemini_key and len(gemini_key) < 20:
            st.warning("⚠️ Format de clé suspect")
        elif not gemini_key:
            st.info("💡 Sans clé → EasyOCR fallback")
    elif "OpenAI" in ocr_choice:
        openai_key = st.text_input("Clé API OpenAI", type="password",
                                    placeholder="sk-...",
                                    help="Obtenez une clé sur platform.openai.com")
        if openai_key and not openai_key.startswith("sk-"):
            st.warning("⚠️ Une clé OpenAI commence par 'sk-'")

    # Tentative de lecture depuis secrets.toml (compat local/cloud)
    try:
        if not gemini_key:
            gemini_key = st.secrets.get("GEMINI_API_KEY", "")
        if not openai_key:
            openai_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        pass

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
        clear_cache = st.button("🧹 Vider le cache OCR")
        if clear_cache:
            st.cache_data.clear()
            st.success("Cache vidé !")

    st.markdown("""
    <div style='background:#f0f7f3;border-radius:10px;padding:.9rem;
                margin:.5rem 0;border-left:4px solid #2c9e5e;font-size:.82rem'>
        <strong>📋 Images supportées</strong><br>
        ✅ Liste des partants<br>✅ Records absolus<br>
        ✅ Stats drivers<br>✅ Stats entraîneurs
    </div>""", unsafe_allow_html=True)

    if st.session_state.analysis_count > 0:
        st.markdown(f"""
        <div style='background:#fff;border-radius:10px;padding:.7rem;
                    margin:.5rem 0;border:1px solid #e8f5ee;font-size:.78rem'>
            <strong>📊 Stats session</strong><br>
            Analyses : {st.session_state.analysis_count}<br>
            Dernière : {st.session_state.last_analysis_time or '-'}
        </div>""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>🏇 PronoHippique AI</h1>
    <p>Intelligence Artificielle pour vos pronostics hippiques — Analysez · Scorez · Gagnez !</p>
</div>""", unsafe_allow_html=True)

# Disclaimer légal
st.markdown("""
<div class='disclaimer'>
    ⚠️ <strong>Avertissement</strong> : Cet outil fournit une analyse statistique à titre informatif uniquement.
    Les jeux d'argent comportent des risques (endettement, dépendance...). Jouez avec modération.
    Pour être aidé : <a href='https://www.joueurs-info-service.fr' target='_blank'>09 74 75 13 13</a>
</div>
""", unsafe_allow_html=True)

# ── SECTION 1 : UPLOAD ───────────────────────────────────────────
st.markdown("## 📤 Étape 1 — Téléchargez vos captures d'écran")
st.markdown(f"""
<div class='card'>
    <h3>💡 Instructions</h3>
    <p>Téléchargez <strong>1 à {MAX_IMAGES_PER_ANALYSIS} captures</strong> de la même course :</p>
    <ul>
      <li>📊 <strong>Liste des partants</strong> — cotes, musique, gains</li>
      <li>🏆 <strong>Records absolus</strong> — meilleure performance</li>
      <li>🏇 <strong>Statistiques drivers</strong></li>
      <li>👨‍🏫 <strong>Statistiques entraîneurs</strong></li>
    </ul>
    <p><em>Plus d'images complémentaires = analyse plus précise !</em></p>
</div>""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    f"📷 Glissez vos images (max {MAX_IMAGES_PER_ANALYSIS})",
    type=SUPPORTED_FORMATS,
    accept_multiple_files=True,
)

# Validation des fichiers uploadés
valid_uploaded = []
if uploaded:
    if len(uploaded) > MAX_IMAGES_PER_ANALYSIS:
        st.warning(f"⚠️ Trop d'images. Seules les {MAX_IMAGES_PER_ANALYSIS} premières seront traitées.")
        uploaded = uploaded[:MAX_IMAGES_PER_ANALYSIS]
    for f in uploaded:
        size_mb = f.size / (1024 * 1024)
        if size_mb > MAX_IMAGE_SIZE_MB:
            st.error(f"❌ `{f.name}` trop volumineux ({size_mb:.1f} MB). Max {MAX_IMAGE_SIZE_MB} MB.")
        else:
            valid_uploaded.append(f)

if valid_uploaded:
    st.markdown(f"**{len(valid_uploaded)} image(s) chargée(s)** ✅")
    cols = st.columns(min(len(valid_uploaded), 4))
    for i, f in enumerate(valid_uploaded):
        with cols[i % 4]:
            try:
                img = Image.open(f)
                st.image(img, caption=f"{f.name} ({f.size//1024} KB)",
                         use_container_width=True)
            except Exception as e:
                st.error(f"Erreur lecture : {e}")

st.divider()

# ── SECTION 2 : BOUTON ANALYSER ──────────────────────────────────
st.markdown("## 🧠 Étape 2 — Lancer l'Analyse")
col_btn, col_msg = st.columns([2, 3])
with col_btn:
    clicked = st.button("🚀 Analyser la Course", use_container_width=True,
                         disabled=not valid_uploaded)
with col_msg:
    if not valid_uploaded:
        st.warning("⚠️ Téléchargez au moins une image valide.")
    elif "Gemini" in ocr_choice and not gemini_key:
        st.warning("⚠️ Aucune clé Gemini → EasyOCR utilisé (moins précis).")
    elif "OpenAI" in ocr_choice and not openai_key:
        st.warning("⚠️ Aucune clé OpenAI → EasyOCR utilisé.")
    else:
        st.success("✅ Prêt pour l'analyse !")

# ── TRAITEMENT ───────────────────────────────────────────────────
if clicked and valid_uploaded:
    st.session_state.done = False
    progress = st.progress(0)
    status   = st.empty()
    extractions = []
    total_steps = len(valid_uploaded) + 3

    # Détermination du moteur préféré
    preferred = "auto"
    if "Gemini" in ocr_choice: preferred = "gemini"
    elif "OpenAI" in ocr_choice: preferred = "openai"
    elif "EasyOCR" in ocr_choice: preferred = "easyocr"

    try:
        for i, f in enumerate(valid_uploaded):
            status.markdown(f"🔍 **OCR** — Image {i+1}/{len(valid_uploaded)} : `{f.name}`...")
            progress.progress(int(i / total_steps * 100))
            img = Image.open(f).convert("RGB")
            result = extract_data_from_image(img, gemini_key, openai_key, preferred)
            extractions.append(result)

        st.session_state.raw_extractions = extractions

        total_horses_found = sum(len(e.get("chevaux", [])) for e in extractions)
        ocr_errors = [e.get("error") for e in extractions if e.get("error")]

        if total_horses_found == 0:
            progress.empty()
            st.session_state.df_scored = pd.DataFrame()
            st.session_state.done = True
            err_msg = ocr_errors[0] if ocr_errors else "Aucun tableau hippique reconnu"
            status.warning(f"⚠️ OCR sans résultat : {err_msg}. Saisie manuelle disponible ci-dessous.")
            time.sleep(1.5)
            status.empty()
            st.rerun()

        progress.progress(int(len(valid_uploaded) / total_steps * 100))
        status.markdown("🔀 **Fusion** des données extraites...")
        merged = merge_extracted_data(extractions)

        progress.progress(int((len(valid_uploaded) + 1) / total_steps * 100))
        status.markdown("🧹 **Nettoyage** et structuration...")
        df_clean = clean_horse_data(merged.get("chevaux", []))
        st.session_state.df_cleaned = df_clean

        if df_clean.empty:
            progress.empty()
            st.session_state.df_scored = pd.DataFrame()
            st.session_state.done = True
            status.error("❌ Aucun cheval valide après nettoyage. Vérifiez vos images.")
            time.sleep(1.5)
            status.empty()
            st.rerun()

        progress.progress(int((len(valid_uploaded) + 2) / total_steps * 100))
        status.markdown("📊 **Calcul des scores**...")
        df_scored = calculate_scores(df_clean, race_type)
        st.session_state.df_scored = df_scored

        progress.progress(100)
        status.markdown("🎯 **Génération du pronostic**...")
        pronostic = generate_pronostic_report(df_scored)
        st.session_state.pronostic = pronostic
        st.session_state.done = True
        st.session_state.analysis_count += 1
        st.session_state.last_analysis_time = datetime.now().strftime("%H:%M:%S")

        progress.empty()
        status.success(f"✅ Analyse terminée — {len(df_scored)} partants !")
        time.sleep(0.6)
        status.empty()
        st.rerun()

    except Exception as e:
        progress.empty()
        status.error(f"❌ Erreur durant l'analyse : {e}")
        st.exception(e)

# ── SECTION 3 : SAISIE MANUELLE (fallback) ───────────────────────
_ocr_failed = (
    st.session_state.done
    and st.session_state.df_scored is not None
    and ("score_global" not in st.session_state.df_scored.columns or st.session_state.df_scored.empty)
)
_no_result_yet = (
    not st.session_state.done
    and st.session_state.raw_extractions
    and all(not e.get("chevaux") for e in st.session_state.raw_extractions)
)

if _ocr_failed or _no_result_yet:
    st.divider()
    st.error("""❌ **L'OCR n'a pas pu extraire les données hippiques.**

**Causes fréquentes :**
- 📸 Image non reconnue comme tableau PMU
- 🔑 Clé API manquante → EasyOCR limité
- 🖼️ Image floue, tronquée ou inhabituelle

**Solutions :**
1. Uploadez **directement** des captures de [Paris-Turf](https://www.paris-turf.com), [Zeturf](https://www.zeturf.fr), PMU
2. Vérifiez votre clé API
3. Utilisez la **saisie manuelle** ci-dessous
""")

    if st.session_state.raw_extractions:
        with st.expander("🔍 Debug OCR"):
            for i, ext in enumerate(st.session_state.raw_extractions):
                st.markdown(f"**Image {i+1}** — Moteur : `{ext.get('ocr_engine','?')}`")
                if ext.get("error"):
                    st.error(f"Erreur : {ext['error']}")
                st.json({k: v for k, v in ext.items() if k != "raw_text"})
                if ext.get("raw_text"):
                    st.text_area(f"Texte brut image {i+1}", ext["raw_text"], height=120)

    st.divider()
    st.markdown("### ✏️ Saisie Manuelle des Partants (mode secours)")
    st.markdown("""
    <div class='card'>
    <p>Renseignez au minimum : <strong>numéro, nom, musique et cote PMU</strong>.</p>
    </div>""", unsafe_allow_html=True)

    nb_manual = st.number_input("Nombre de partants", min_value=2, max_value=20, value=8, step=1)

    manual_horses = []
    for i in range(int(nb_manual)):
        with st.expander(f"Cheval #{i+1}", expanded=(i < 3)):
            c1, c2, c3 = st.columns(3)
            with c1:
                num  = st.number_input("N°",          key=f"m_num_{i}",  min_value=1, max_value=30, value=i+1)
                nom  = st.text_input("Nom",            key=f"m_nom_{i}",  value="")
                sa   = st.text_input("SA (ex: H7)",   key=f"m_sa_{i}",   value="")
            with c2:
                mus  = st.text_input("Musique",        key=f"m_mus_{i}",  value="",
                                     help="Ex: 1a3a2aDa5a")
                cote = st.number_input("Cote PMU",     key=f"m_cote_{i}", min_value=0.0, value=0.0, step=0.1, format="%.1f")
                gains= st.number_input("Gains (€)",   key=f"m_gains_{i}",min_value=0,   value=0,   step=1000)
            with c3:
                drv  = st.text_input("Driver",         key=f"m_drv_{i}",  value="")
                entr = st.text_input("Entraîneur",     key=f"m_entr_{i}", value="")
                rdpct= st.number_input("% Driver",     key=f"m_rdpct_{i}",min_value=0.0, max_value=100.0, value=0.0, step=0.5, format="%.1f")

            manual_horses.append({
                "numero": int(num), "cheval": nom, "sa": sa,
                "musique": mus, "cote_pmu": float(cote), "gains": int(gains),
                "driver": drv, "entraineur": entr, "reussite_driver": float(rdpct),
            })

    if st.button("🎯 Analyser la saisie manuelle", use_container_width=True):
        valid = [h for h in manual_horses if h.get("cheval", "").strip()]
        if len(valid) < 2:
            st.warning("⚠️ Renseignez au moins 2 chevaux avec un nom.")
        else:
            with st.spinner("Calcul des scores en cours..."):
                df_manual = clean_horse_data(valid)
                df_scored_manual = calculate_scores(df_manual, race_type)
                pronostic_manual = generate_pronostic_report(df_scored_manual)
                st.session_state.df_cleaned   = df_manual
                st.session_state.df_scored    = df_scored_manual
                st.session_state.pronostic    = pronostic_manual
                st.session_state.done         = True
                st.session_state.raw_extractions = []
                st.session_state.analysis_count += 1
                st.session_state.last_analysis_time = datetime.now().strftime("%H:%M:%S")
            st.rerun()

# ── RÉSULTATS ────────────────────────────────────────────────────
if st.session_state.done and st.session_state.df_scored is not None:
    df = st.session_state.df_scored
    pronostic = st.session_state.pronostic

    if "score_global" not in df.columns or df.empty:
        st.stop()

    n_part = len(df)

    st.divider()
    st.markdown("## 📊 Résultats de l'Analyse")

    # Métriques rapides
    qual = assess_data_quality(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("🐎 Partants", n_part)
    with c2:
        fav = pronostic.get("favori", {})
        st.metric("🏆 Favori IA", f"#{fav.get('numero','?')}", fav.get("cheval", "-"))
    with c3:
        st.metric("📈 Qualité", f"{qual.get('qualite',0)}%")
    with c4:
        eng = (st.session_state.raw_extractions[0].get("ocr_engine", "Manuel")
               if st.session_state.raw_extractions else "Manuel")
        st.metric("🤖 OCR", eng.split(" ")[0])
    with c5:
        st.metric("🎲 Proba favori", f"{fav.get('proba', 0):.1f}%")

    st.divider()

    # Onglets
    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "🏆 Pronostic", "📊 Classement", "📈 Graphiques",
        "🔍 Données",  "🎰 Combinaisons", "📋 Détail Scores", "💾 Export",
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
                <p>Écart entre le favori IA et son dauphin : <strong>{gap:.2f} pts</strong></p>
                <p><strong>Partants analysés :</strong> {n_part}</p>
                <p><strong>Type de course :</strong> {race_type.title()}</p>
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
                proba = horse.get("proba_victoire", 0)
                st.markdown(f"""
                <div class='{styles[i]}'>
                    <div style='font-size:2rem;text-align:center'>{medals[i]}</div>
                    <h3 style='text-align:center;margin:.3rem 0'>
                        #{horse['numero']} {horse['cheval']}</h3>
                    <p style='text-align:center;font-size:1.2rem;
                              font-weight:700;color:#1a6b3c'>
                        Score : {horse['score_global']:.2f}/10</p>
                    <p style='text-align:center;font-size:.95rem;color:#555'>
                        🎲 Proba : {proba:.1f}%</p>
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
                "Proba":     f"{row.get('proba_victoire', 0):.1f}%",
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

    # ── TAB 3 : GRAPHIQUES ───────────────────────────────────────
    with t3:
        st.markdown("### 📈 Visualisations")
        st.plotly_chart(plot_scores_bar(df), use_container_width=True)
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(plot_radar_top3(df), use_container_width=True)
        with g2:
            st.plotly_chart(plot_proba_pie(df), use_container_width=True)
        st.plotly_chart(plot_driver_comparison(df), use_container_width=True)
        st.plotly_chart(plot_form_history(df), use_container_width=True)

    # ── TAB 4 : DONNÉES ──────────────────────────────────────────
    with t4:
        st.markdown("### 🔍 Données Extraites")
        q = assess_data_quality(df)
        st.markdown(f"""
        <div class='card'>
            <strong>Qualité globale : {q.get('qualite',0)}%</strong>
            &nbsp;|&nbsp; Partants : {q.get('nb_chevaux',0)}
        </div>""", unsafe_allow_html=True)

        # Détail par champ
        if q.get("details"):
            st.markdown("**Couverture par champ :**")
            cols_q = st.columns(min(len(q["details"]), 5))
            for idx, (field, pct) in enumerate(q["details"].items()):
                with cols_q[idx % 5]:
                    color = "#1a6b3c" if pct >= 80 else ("#f39c12" if pct >= 50 else "#e74c3c")
                    st.markdown(
                        f"<div style='text-align:center;padding:.4rem;border:1px solid #eee;"
                        f"border-radius:6px;margin:.2rem'>"
                        f"<small><b>{field}</b></small><br>"
                        f"<span style='color:{color};font-weight:700'>{pct}%</span></div>",
                        unsafe_allow_html=True
                    )

        st.divider()
        keep = ["numero","cheval","sa","driver","entraineur",
                "record_brut","reussite_driver","reussite_entraineur",
                "ecart_driver","gains","cote_pmu","musique"]
        st.dataframe(df[[c for c in keep if c in df.columns]],
                     use_container_width=True, hide_index=True)
        if show_raw and st.session_state.raw_extractions:
            st.markdown("#### 📝 Réponses brutes OCR")
            for i, ext in enumerate(st.session_state.raw_extractions):
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
                if not combos:
                    st.info("Pas assez de partants.")
                    return
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

    # ── TAB 7 : EXPORT ───────────────────────────────────────────
    with t7:
        st.markdown("### 💾 Export des Résultats")
        st.markdown("""
        <div class='card'>
            <p>Téléchargez votre analyse dans différents formats pour la consulter
            hors-ligne ou la partager.</p>
        </div>""", unsafe_allow_html=True)

        e1, e2, e3 = st.columns(3)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        with e1:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📊 CSV (données brutes)", csv,
                f"pronohippique_{timestamp}.csv", "text/csv",
                use_container_width=True
            )

        with e2:
            try:
                xlsx_bytes = export_to_excel(df, pronostic)
                if xlsx_bytes:
                    st.download_button(
                        "📈 Excel (multi-onglets)", xlsx_bytes,
                        f"pronohippique_{timestamp}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                else:
                    st.button("📈 Excel indisponible", disabled=True, use_container_width=True)
            except Exception:
                st.button("📈 Excel indisponible", disabled=True, use_container_width=True)

        with e3:
            txt = export_to_text_report(df, pronostic).encode("utf-8")
            st.download_button(
                "📄 Rapport texte", txt,
                f"pronohippique_{timestamp}.txt", "text/plain",
                use_container_width=True
            )

        st.divider()
        st.markdown("#### 👀 Aperçu du rapport texte")
        st.text(export_to_text_report(df, pronostic))

# ── FOOTER ───────────────────────────────────────────────────────
st.divider()
st.markdown(f"""
<div style='text-align:center;color:#666;font-size:.85rem;padding:1rem'>
    🏇 PronoHippique AI v{APP_VERSION} • Made with Streamlit •
    <strong>Jouez avec modération</strong>
</div>
""", unsafe_allow_html=True)
