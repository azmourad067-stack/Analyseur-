# ══════════════════════════════════════════════════════════════════
#  MODULE 1 — OCR EXTRACTOR v3.0 (HAUTE PRÉCISION)
# ══════════════════════════════════════════════════════════════════

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# ── PRÉ-TRAITEMENT D'IMAGE ──────────────────────────────────────

def preprocess_image_for_ocr(image: Image.Image,
                              upscale: bool = True,
                              enhance_contrast: bool = True,
                              denoise: bool = True,
                              sharpen: bool = True) -> Image.Image:
    """
    Pré-traite l'image pour maximiser la qualité OCR.
    - Upscale x2 si l'image est petite (< 1500px)
    - Améliore le contraste et la netteté
    - Débruite légèrement
    - Convertit en RGB si nécessaire
    """
    img = image.copy()

    # Conversion RGB obligatoire
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Upscaling intelligent : améliore la lecture des petites polices
    if upscale:
        w, h = img.size
        max_dim = max(w, h)
        if max_dim < 1500:
            scale = 2.0
        elif max_dim < 2200:
            scale = 1.5
        else:
            scale = 1.0
        if scale > 1.0:
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)

    # Auto-rotation basée sur les EXIF (corrige photos téléphone)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # Améliore le contraste (utile pour tableaux clairs/délavés)
    if enhance_contrast:
        img = ImageEnhance.Contrast(img).enhance(1.25)
        img = ImageEnhance.Brightness(img).enhance(1.05)

    # Augmente la netteté (texte plus lisible)
    if sharpen:
        img = ImageEnhance.Sharpness(img).enhance(1.5)

    # Débruitage léger (évite les artefacts JPEG)
    if denoise:
        img = img.filter(ImageFilter.MedianFilter(size=3))

    return img


def preprocess_image_aggressive(image: Image.Image) -> Image.Image:
    """
    Pré-traitement agressif pour images de mauvaise qualité.
    Niveaux de gris + binarisation adaptive + upscale x3.
    """
    img = image.copy()
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Upscale fort
    w, h = img.size
    if max(w, h) < 2000:
        img = img.resize((int(w * 2.5), int(h * 2.5)), Image.LANCZOS)

    # Conversion niveaux de gris pour augmenter le contraste
    gray = ImageOps.grayscale(img)

    # Auto-contraste
    gray = ImageOps.autocontrast(gray, cutoff=2)

    # Netteté forte
    gray = ImageEnhance.Sharpness(gray).enhance(2.0)

    # Reconversion RGB pour APIs
    return gray.convert("RGB")


# ── DÉTECTION AUTO DU TYPE DE TABLEAU ───────────────────────────

def detect_table_type_from_image(image: Image.Image, api_key: str) -> str:
    """
    Première passe ultra-rapide pour détecter le type de tableau.
    Permet d'utiliser ensuite un prompt spécialisé bien plus précis.
    """
    if not api_key:
        return "unknown"
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = """Quelle est la nature de ce tableau hippique ?
Réponds UNIQUEMENT par UN MOT parmi :
- "partants" (liste avec cotes, musique, gains)
- "records" (records absolus chronométriques)
- "drivers" (statistiques drivers : courses, victoires, %)
- "entraineurs" (statistiques entraîneurs)
- "mixed" (plusieurs types combinés)
- "unknown"

Réponds par UN SEUL MOT, sans explication."""

        response = model.generate_content(
            [prompt, image],
            generation_config={"temperature": 0.0, "max_output_tokens": 20}
        )
        result = response.text.strip().lower().replace('"', '').replace("'", "")
        valid = ["partants", "records", "drivers", "entraineurs", "mixed", "unknown"]
        for v in valid:
            if v in result:
                return v
        return "unknown"
    except Exception:
        return "unknown"


# ── PROMPTS SPÉCIALISÉS PAR TYPE ────────────────────────────────

def _build_prompt_partants() -> str:
    return """Tu es un EXPERT en analyse de tableaux PMU de partants hippiques français.

MISSION : Extrais EXACTEMENT toutes les données visibles dans ce tableau de partants.

⚠️ RÈGLES STRICTES :
1. Lis CHAQUE LIGNE attentivement, du haut vers le bas
2. Ne JAMAIS inventer de données — si un champ n'est pas visible, mets `null`
3. Le numéro du cheval est TOUJOURS la première colonne
4. Les noms de chevaux peuvent être en MAJUSCULES ou Capitalisés
5. La musique contient chiffres + lettres (ex: "1a3a2aDa", "(24)5a1m6a")
6. Les gains sont en euros, parfois avec espaces ("173 060" = 173060)
7. Les cotes utilisent virgule OU point décimal ("1,9" = 1.9)
8. Si un cheval est NON-PARTANT, indique `"non_partant": true`

📋 FORMAT JSON STRICT (réponds UNIQUEMENT avec ce JSON, sans markdown) :
{
  "table_type": "partants",
  "nb_partants": <nombre>,
  "chevaux": [
    {
      "numero": <int>,
      "cheval": "<nom exact>",
      "sa": "<sexe+age, ex: M7, H6, F5>",
      "dist": <distance en mètres, ex: 2100>,
      "driver": "<nom complet du driver>",
      "entraineur": "<nom complet de l'entraîneur>",
      "musique": "<chaîne brute, ex: 1a3a2aDa5a>",
      "gains": <int en euros>,
      "cote_pmu": <float>,
      "cote_genybet": <float ou null>,
      "non_partant": <true si non partant, sinon false>
    }
  ]
}

EXEMPLE :
{
  "table_type": "partants",
  "nb_partants": 2,
  "chevaux": [
    {"numero": 1, "cheval": "PACHA DE MORTREE", "sa": "M8", "dist": 2100,
     "driver": "A. Abrivard", "entraineur": "P. Lecellier",
     "musique": "1a3a2a4aDa", "gains": 245000, "cote_pmu": 3.2,
     "cote_genybet": 3.5, "non_partant": false},
    {"numero": 2, "cheval": "Quenelle Star", "sa": "F6", "dist": 2100,
     "driver": "M. Mottier", "entraineur": "J. Westholm",
     "musique": "(24)5aDa1a", "gains": 119000, "cote_pmu": 12.0,
     "cote_genybet": null, "non_partant": false}
  ]
}

Vérifie 2 fois ta réponse. Compte le nombre de lignes du tableau et assure-toi d'avoir extrait CHACUNE."""


def _build_prompt_records() -> str:
    return """Tu es un EXPERT en analyse de tableaux de RECORDS ABSOLUS hippiques.

MISSION : Extrais TOUS les records chronométriques visibles.

⚠️ RÈGLES :
- Le record est au format "1'10\\"5" ou "1'10.5" (= 70.5 secondes)
- La date du record est au format "JJ/MM/AA - Lieu Distance, Position"
- Garde la chaîne BRUTE du record (ne convertis pas)

Format JSON STRICT (sans markdown) :
{
  "table_type": "records",
  "chevaux": [
    {
      "numero": <int>,
      "cheval": "<nom>",
      "sa": "<ex: M7>",
      "dist": <int>,
      "driver": "<nom>",
      "record": "<chaîne brute, ex: 1'10\\"5>",
      "date_record": "<chaîne brute>"
    }
  ]
}

Extrais CHAQUE ligne du tableau."""


def _build_prompt_drivers() -> str:
    return """Tu es un EXPERT en statistiques DRIVERS PMU hippiques.

MISSION : Extrais les stats de réussite des drivers pour chaque cheval.

⚠️ RÈGLES :
- Courses_Driver = nombre total de courses du driver
- Victoires_Driver = nombre de victoires
- Ecart_Driver = nombre de courses depuis la dernière victoire (0 = vient de gagner)
- Reussite_Driver = pourcentage entier (16 pour 16%)
- Musique_Driver = chaîne brute des résultats récents

Format JSON STRICT (sans markdown) :
{
  "table_type": "stats_drivers",
  "chevaux": [
    {
      "numero": <int>,
      "cheval": "<nom>",
      "dist": <int>,
      "driver": "<nom>",
      "courses_driver": <int>,
      "victoires_driver": <int>,
      "ecart_driver": <int>,
      "reussite_driver": <int>,
      "musique_driver": "<chaîne brute>"
    }
  ]
}

Extrais CHAQUE ligne, vérifie le nombre de partants."""


def _build_prompt_entraineurs() -> str:
    return """Tu es un EXPERT en statistiques ENTRAÎNEURS PMU hippiques.

MISSION : Extrais les stats de réussite des entraîneurs pour chaque cheval.

⚠️ RÈGLES (mêmes que drivers mais pour entraîneurs).

Format JSON STRICT (sans markdown) :
{
  "table_type": "stats_entraineurs",
  "chevaux": [
    {
      "numero": <int>,
      "cheval": "<nom>",
      "dist": <int>,
      "entraineur": "<nom>",
      "courses_entraineur": <int>,
      "victoires_entraineur": <int>,
      "ecart_entraineur": <int>,
      "reussite_entraineur": <int>,
      "musique_entraineur": "<chaîne brute>"
    }
  ]
}

Extrais CHAQUE ligne."""


def _build_prompt_mixed_or_unknown() -> str:
    """Prompt universel exhaustif, capture tout ce qui est visible."""
    return """Tu es un EXPERT en analyse de tableaux hippiques français (PMU, Paris-Turf, Zeturf, Genybet).

MISSION : Extrais TOUTES les données visibles, peu importe le type de tableau.

⚠️ RÈGLES CRITIQUES :
1. Lis ATTENTIVEMENT chaque ligne du tableau
2. Champs ABSENTS = `null` (ne JAMAIS inventer)
3. Numéro de cheval = première colonne (1 à 20)
4. Pour la musique : conserve la chaîne brute exacte ("1a3aDa" tel quel)
5. Gains : enlève les espaces ("173 060" → 173060)
6. Cotes : virgule décimale → point ("1,9" → 1.9)
7. Pourcentages : valeur entière (16 pour 16%)
8. Records : conserve format brut "1'10\\"5"
9. Si plusieurs tableaux dans l'image, fusionne les données par numéro

📋 FORMAT JSON STRICT (réponds UNIQUEMENT avec ce JSON, pas de markdown) :
{
  "table_type": "<partants|records|stats_drivers|stats_entraineurs|mixed>",
  "nb_partants": <int>,
  "chevaux": [
    {
      "numero": <int>,
      "cheval": "<nom>",
      "sa": "<ex: M7>",
      "dist": <int>,
      "driver": "<nom>",
      "entraineur": "<nom>",
      "musique": "<chaîne brute>",
      "gains": <int>,
      "cote_pmu": <float>,
      "cote_genybet": <float>,
      "record": "<chaîne brute>",
      "date_record": "<chaîne>",
      "courses_driver": <int>,
      "victoires_driver": <int>,
      "ecart_driver": <int>,
      "reussite_driver": <int>,
      "musique_driver": "<chaîne>",
      "courses_entraineur": <int>,
      "victoires_entraineur": <int>,
      "ecart_entraineur": <int>,
      "reussite_entraineur": <int>,
      "musique_entraineur": "<chaîne>"
    }
  ]
}

VÉRIFICATION FINALE :
- Compte les lignes du tableau
- Vérifie que tu as extrait TOUS les chevaux
- Si tu hésites sur une valeur, mets `null` plutôt qu'une donnée fausse"""


def _get_prompt_for_type(table_type: str) -> str:
    prompts = {
        "partants":     _build_prompt_partants(),
        "records":      _build_prompt_records(),
        "drivers":      _build_prompt_drivers(),
        "entraineurs":  _build_prompt_entraineurs(),
        "mixed":        _build_prompt_mixed_or_unknown(),
        "unknown":      _build_prompt_mixed_or_unknown(),
    }
    return prompts.get(table_type, _build_prompt_mixed_or_unknown())


# ── PARSING JSON RENFORCÉ ────────────────────────────────────────

def _parse_json_response(raw_text: str) -> dict:
    """Parse robuste avec récupération d'erreurs avancée."""
    if not raw_text:
        return {"error": "Réponse vide"}

    # 1. Nettoyer markdown
    clean = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
    clean = re.sub(r"```\s*$", "", clean).strip()

    # 2. Trouver le JSON principal
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start == -1 or end == 0:
        return {"error": "JSON introuvable", "raw_text": raw_text}
    json_str = clean[start:end]

    # 3. Réparations courantes
    # Virgules décimales françaises
    json_str = re.sub(r'("cote_[^"]+"\s*:\s*)(\d+),(\d+)', r'\1\2.\3', json_str)
    json_str = re.sub(r'("gains"\s*:\s*)"?(\d+)\s+(\d+)"?', r'\1\2\3', json_str)
    # Trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    # Quotes manquantes sur valeurs string contenant tirets
    # ex: "musique": 1a3aDa  →  "musique": "1a3aDa"
    json_str = re.sub(r'(:\s*)([0-9]+[a-zA-Z]+[0-9a-zA-Z]+)([,\}\]])',
                       r'\1"\2"\3', json_str)
    # null mal écrits
    json_str = re.sub(r':\s*None\b', ': null', json_str)
    json_str = re.sub(r':\s*N/A\b', ': null', json_str)
    json_str = re.sub(r':\s*"-"', ': null', json_str)

    try:
        data = json.loads(json_str)
        data["raw_text"] = raw_text
        return data
    except json.JSONDecodeError as e:
        # Tentative de récupération : extraction des chevaux par regex
        chevaux = _fallback_extract_chevaux_from_json_text(json_str)
        if chevaux:
            return {
                "table_type": "partants",
                "chevaux": chevaux,
                "raw_text": raw_text,
                "warning": f"JSON partiellement parsé : {e}"
            }
        return {"error": f"JSON invalide : {e}", "raw_text": raw_text}


def _fallback_extract_chevaux_from_json_text(json_str: str) -> list:
    """En cas d'échec de json.loads, essaie d'extraire les blocs cheval par cheval."""
    chevaux = []
    # Cherche chaque bloc { ... } à l'intérieur du tableau "chevaux"
    blocks = re.findall(r'\{[^{}]*"numero"\s*:\s*\d+[^{}]*\}', json_str)
    for block in blocks:
        # Répare ce bloc isolé
        b = re.sub(r',\s*}', '}', block)
        try:
            h = json.loads(b)
            if h.get("numero"):
                chevaux.append(h)
        except Exception:
            # Extraction manuelle clé par clé
            h = {}
            for key in ["numero", "cheval", "sa", "dist", "driver", "entraineur",
                        "musique", "gains", "cote_pmu", "cote_genybet", "record",
                        "courses_driver", "victoires_driver", "ecart_driver",
                        "reussite_driver", "courses_entraineur", "victoires_entraineur",
                        "ecart_entraineur", "reussite_entraineur"]:
                # Pattern pour valeurs numériques ou strings
                m = re.search(rf'"{key}"\s*:\s*"?([^",}}]+)"?', b)
                if m:
                    val = m.group(1).strip().strip('"')
                    if val.lower() in ("null", "none", "n/a", ""):
                        continue
                    h[key] = val
            if h.get("numero"):
                chevaux.append(h)
    return chevaux


# ── EXTRACTION GEMINI AMÉLIORÉE ──────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600, max_entries=50)
def _cached_extract_gemini(image_hash: str, image_bytes: bytes,
                            api_key: str, table_type: str = "unknown",
                            model_name: str = "gemini-1.5-flash") -> dict:
    """Extraction Gemini avec prompt spécialisé."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name,
            generation_config={
                "temperature": 0.0,        # Déterministe
                "top_p": 0.95,
                "max_output_tokens": 8192, # Augmenté pour gros tableaux
                "response_mime_type": "application/json",  # JSON forcé
            }
        )
        img = Image.open(io.BytesIO(image_bytes))
        prompt = _get_prompt_for_type(table_type)
        response = model.generate_content([prompt, img])
        result = _parse_json_response(response.text)
        result["ocr_engine"] = f"Gemini ({model_name})"
        result["detected_type"] = table_type
        return result
    except Exception as e:
        err = str(e)
        if "API_KEY" in err or "invalid" in err.lower():
            err = "Clé API Gemini invalide ou expirée"
        elif "quota" in err.lower():
            err = "Quota Gemini dépassé"
        elif "PERMISSION_DENIED" in err:
            err = "Permission refusée"
        elif "response_mime_type" in err.lower():
            # Fallback sans response_mime_type pour anciens modèles
            return _cached_extract_gemini_fallback(image_hash, image_bytes,
                                                     api_key, table_type, model_name)
        return {"error": err, "ocr_engine": f"Gemini ({model_name})"}


@st.cache_data(show_spinner=False, ttl=3600, max_entries=50)
def _cached_extract_gemini_fallback(image_hash: str, image_bytes: bytes,
                                      api_key: str, table_type: str,
                                      model_name: str) -> dict:
    """Fallback Gemini sans response_mime_type."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name,
            generation_config={"temperature": 0.0, "max_output_tokens": 8192}
        )
        img = Image.open(io.BytesIO(image_bytes))
        prompt = _get_prompt_for_type(table_type)
        response = model.generate_content([prompt, img])
        result = _parse_json_response(response.text)
        result["ocr_engine"] = f"Gemini ({model_name})"
        return result
    except Exception as e:
        return {"error": str(e), "ocr_engine": f"Gemini ({model_name})"}


def extract_with_gemini(image: Image.Image, api_key: str,
                          table_type: str = "auto",
                          aggressive_preprocess: bool = False) -> dict:
    """
    Extraction Gemini avec :
    - Pré-traitement d'image
    - Détection auto du type de tableau
    - Prompt spécialisé
    - Modèle pro en fallback si flash échoue
    """
    if not api_key or len(api_key.strip()) < 10:
        return {"error": "Clé API Gemini manquante", "ocr_engine": "Gemini"}

    # 1. Pré-traitement
    if aggressive_preprocess:
        processed = preprocess_image_aggressive(image)
    else:
        processed = preprocess_image_for_ocr(image)

    # 2. Détection auto du type si demandé
    if table_type == "auto":
        table_type = detect_table_type_from_image(processed, api_key)

    # 3. Hash + bytes
    img_hash = _hash_image(processed) + "_" + table_type
    buf = io.BytesIO()
    processed.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    # 4. Premier essai : flash (rapide)
    result = _cached_extract_gemini(img_hash, img_bytes, api_key,
                                      table_type, "gemini-1.5-flash")

    # 5. Si échec ou peu de chevaux, retry avec modèle pro
    nb_chevaux = len(result.get("chevaux", []))
    has_error = bool(result.get("error"))

    if has_error or nb_chevaux < 3:
        # Tentative avec gemini-2.0-flash-exp (plus précis)
        for fallback_model in ["gemini-2.0-flash-exp", "gemini-1.5-pro"]:
            retry = _cached_extract_gemini(
                img_hash + f"_{fallback_model}", img_bytes, api_key,
                table_type, fallback_model
            )
            if retry.get("chevaux") and len(retry["chevaux"]) > nb_chevaux:
                retry["fallback_used"] = fallback_model
                return retry

    return result


# ── EXTRACTION OPENAI AMÉLIORÉE ──────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600, max_entries=50)
def _cached_extract_openai(image_hash: str, image_bytes: bytes,
                            api_key: str, table_type: str = "unknown") -> dict:
    """Extraction OpenAI GPT-4o avec prompt spécialisé et JSON mode."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, timeout=90)
        img = Image.open(io.BytesIO(image_bytes))
        img_b64 = _encode_image_base64(img, max_size=2000)
        prompt = _get_prompt_for_type(table_type)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un expert OCR spécialisé dans les tableaux hippiques français. Tu réponds TOUJOURS en JSON valide."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {
                             "url": f"data:image/png;base64,{img_b64}",
                             "detail": "high"  # Mode haute résolution
                         }},
                    ],
                }
            ],
            max_tokens=8192,
            temperature=0.0,
            response_format={"type": "json_object"},  # JSON garanti
        )
        result = _parse_json_response(response.choices[0].message.content)
        result["ocr_engine"] = "OpenAI GPT-4o (high detail)"
        result["detected_type"] = table_type
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


def extract_with_openai(image: Image.Image, api_key: str,
                          table_type: str = "auto") -> dict:
    """Extraction OpenAI avec pré-traitement et détection auto."""
    if not api_key or not api_key.startswith("sk-"):
        return {"error": "Clé API OpenAI invalide", "ocr_engine": "OpenAI GPT-4o"}

    processed = preprocess_image_for_ocr(image)

    # Détection auto via Gemini si possible (plus rapide), sinon "unknown"
    if table_type == "auto":
        table_type = "unknown"  # OpenAI gère bien le prompt universel

    img_hash = _hash_image(processed) + "_" + table_type
    buf = io.BytesIO()
    processed.save(buf, format="PNG")
    return _cached_extract_openai(img_hash, buf.getvalue(), api_key, table_type)


# ── EXTRACTION MULTI-PASSES (consensus) ──────────────────────────

def extract_with_multi_pass(image: Image.Image,
                              gemini_key: str,
                              openai_key: str) -> dict:
    """
    Extraction avec 2 passes (Gemini + OpenAI) et fusion par consensus.
    Retourne les données les plus fiables (présentes dans au moins 1 source).
    """
    results = []
    if gemini_key:
        r1 = extract_with_gemini(image, gemini_key, "auto")
        if r1.get("chevaux"):
            results.append(r1)
    if openai_key:
        r2 = extract_with_openai(image, openai_key, "auto")
        if r2.get("chevaux"):
            results.append(r2)

    if not results:
        return {"error": "Aucun moteur n'a pu extraire de données", "chevaux": []}
    if len(results) == 1:
        return results[0]

    # Fusion par numéro : garde la valeur non-null la plus fréquente par champ
    merged = _merge_with_consensus(results)
    merged["ocr_engine"] = "Multi-pass (Gemini + OpenAI)"
    return merged


def _merge_with_consensus(results: list) -> dict:
    """Fusionne plusieurs extractions en privilégiant les valeurs concordantes."""
    horse_by_num = {}
    for res in results:
        for h in res.get("chevaux", []):
            num = h.get("numero")
            if num is None:
                continue
            try:
                num = int(num)
            except (ValueError, TypeError):
                continue
            if num not in horse_by_num:
                horse_by_num[num] = []
            horse_by_num[num].append(h)

    merged_chevaux = []
    for num, versions in sorted(horse_by_num.items()):
        if len(versions) == 1:
            merged_chevaux.append(versions[0])
            continue
        # Pour chaque champ, vote majoritaire ou première valeur non-null
        all_keys = set()
        for v in versions:
            all_keys.update(v.keys())

        consensus = {"numero": num}
        for key in all_keys:
            if key == "numero":
                continue
            values = [v.get(key) for v in versions
                       if v.get(key) is not None and v.get(key) != ""]
            if not values:
                continue
            # Vote majoritaire pour string/int courts
            if all(isinstance(x, (str, int, float, bool)) for x in values):
                # Compte les occurrences
                from collections import Counter
                try:
                    counter = Counter(str(v).strip() for v in values)
                    most_common = counter.most_common(1)[0][0]
                    # Reconvertir au type d'origine
                    for v in values:
                        if str(v).strip() == most_common:
                            consensus[key] = v
                            break
                except Exception:
                    consensus[key] = values[0]
            else:
                consensus[key] = values[0]
        merged_chevaux.append(consensus)

    return {
        "chevaux": merged_chevaux,
        "table_type": results[0].get("table_type", "unknown"),
        "nb_partants": len(merged_chevaux),
        "consensus_sources": len(results),
    }


# ── EASYOCR AMÉLIORÉ ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_easyocr_reader():
    import easyocr
    return easyocr.Reader(["fr", "en"], gpu=False, verbose=False)


def extract_with_easyocr(image: Image.Image) -> dict:
    """EasyOCR avec pré-traitement et parsing structuré renforcé."""
    try:
        import numpy as np_local

        # Pré-traitement intensif (EasyOCR aime les images contrastées)
        processed = preprocess_image_aggressive(image)

        reader = _get_easyocr_reader()
        img_array = np_local.array(processed)

        # Paramètres optimisés pour les tableaux
        results = reader.readtext(
            img_array,
            detail=1,
            paragraph=False,
            width_ths=0.7,        # Tolérance largeur
            height_ths=0.7,       # Tolérance hauteur
            decoder="greedy",
            beamWidth=5,
            batch_size=4,
        )

        # Regroupement par lignes (bucket Y adaptatif)
        if not results:
            return {
                "error": "Aucun texte détecté",
                "ocr_engine": "EasyOCR",
                "chevaux": [], "table_type": "unknown",
            }

        # Calcul de la hauteur moyenne pour bucket adaptatif
        heights = [abs(b[2][1] - b[0][1]) for b, _, _ in results]
        avg_h = max(int(np_local.median(heights)), 12) if heights else 12
        bucket_size = max(int(avg_h * 0.6), 8)

        lines_by_y = {}
        for bbox, text, conf in results:
            if conf < 0.20:  # Seuil bas pour capter plus
                continue
            y_center = int((bbox[0][1] + bbox[2][1]) / 2)
            y_bucket = (y_center // bucket_size) * bucket_size
            lines_by_y.setdefault(y_bucket, []).append((bbox[0][0], text.strip(), conf))

        lines = []
        for y in sorted(lines_by_y):
            items = sorted(lines_by_y[y], key=lambda x: x[0])
            line_text = " | ".join(t for _, t, _ in items)
            lines.append(line_text)

        raw_text = "\n".join(lines)
        chevaux = _parse_easyocr_lines_to_chevaux(lines)

        return {
            "type": "raw_ocr",
            "raw_text": raw_text,
            "lines": lines,
            "ocr_engine": "EasyOCR (optimisé)",
            "chevaux": chevaux,
            "table_type": "partants" if chevaux else "unknown",
        }
    except ImportError:
        return {
            "error": "EasyOCR non installé (pip install easyocr)",
            "ocr_engine": "EasyOCR", "chevaux": [], "table_type": "unknown",
        }
    except Exception as e:
        return {
            "error": str(e), "ocr_engine": "EasyOCR",
            "chevaux": [], "table_type": "unknown",
        }


# ── ORCHESTRATEUR PRINCIPAL ──────────────────────────────────────

def extract_data_from_image(
    image: Image.Image,
    gemini_api_key: str = "",
    openai_api_key: str = "",
    preferred: str = "auto",
    use_multi_pass: bool = False,
) -> dict:
    """
    Orchestrateur OCR amélioré avec :
    - Pré-traitement automatique
    - Détection auto du type
    - Prompt spécialisé
    - Retry sur modèle plus puissant si peu de résultats
    - Multi-passes optionnel (Gemini + OpenAI consensus)
    - Fallback EasyOCR
    """
    # Mode multi-passes (le plus précis si 2 clés API)
    if use_multi_pass and gemini_api_key and openai_api_key:
        result = extract_with_multi_pass(image, gemini_api_key, openai_api_key)
        if result.get("chevaux"):
            return result

    # Construction de la chaîne de moteurs
    engines = []
    if preferred in ("gemini", "auto") and gemini_api_key:
        engines.append(("gemini", lambda: extract_with_gemini(image, gemini_api_key, "auto")))
    if preferred in ("openai", "auto") and openai_api_key:
        engines.append(("openai", lambda: extract_with_openai(image, openai_api_key, "auto")))
    engines.append(("easyocr", lambda: extract_with_easyocr(image)))

    last_result = None
    for name, fn in engines:
        result = fn()
        last_result = result
        nb_chevaux = len(result.get("chevaux", []))
        if nb_chevaux >= 3:  # Considéré comme succès si au moins 3 chevaux
            return result

    # Si aucun résultat satisfaisant, tenter pré-traitement agressif sur Gemini
    if gemini_api_key and last_result and len(last_result.get("chevaux", [])) < 3:
        retry = extract_with_gemini(image, gemini_api_key, "auto",
                                       aggressive_preprocess=True)
        if len(retry.get("chevaux", [])) > len(last_result.get("chevaux", [])):
            retry["preprocessing"] = "aggressive"
            return retry

    return last_result or {"error": "Aucun moteur OCR disponible"}


# ── VALIDATION & AUTO-CORRECTION POST-OCR ────────────────────────

def validate_and_correct_horses(chevaux: list) -> tuple:
    """
    Valide et corrige automatiquement les données extraites.
    Retourne (chevaux_corrigés, liste_warnings).
    """
    corrected = []
    warnings = []

    for h in chevaux:
        h2 = dict(h)

        # 1. Numéro obligatoire et valide
        try:
            num = int(h2.get("numero", 0))
            if not (1 <= num <= 20):
                warnings.append(f"Numéro invalide ignoré : {h2.get('numero')}")
                continue
            h2["numero"] = num
        except (ValueError, TypeError):
            warnings.append(f"Numéro non numérique : {h2.get('numero')}")
            continue

        # 2. Nom de cheval — nettoyer caractères parasites
        nom = h2.get("cheval", "")
        if isinstance(nom, str):
            nom = re.sub(r"[\x00-\x1f\x7f]", "", nom).strip()
            nom = re.sub(r"\s+", " ", nom)
            # Supprimer numéros parasites en début ("1 PACHA" → "PACHA")
            nom = re.sub(r"^\d+\s+", "", nom)
            h2["cheval"] = nom

        # 3. Cote PMU — corriger erreurs OCR (1.9 vs 19)
        cote = h2.get("cote_pmu")
        if cote is not None:
            try:
                cote_v = float(str(cote).replace(",", "."))
                # Si cote suspecte (entre 100 et 999 sans décimale), peut être 10.0-99.9
                if 100 <= cote_v <= 999 and len(str(int(cote_v))) == 3:
                    # Probablement 10.0-99.9 mal lu
                    cote_v = cote_v / 10.0
                    warnings.append(f"#{num}: cote corrigée {cote} → {cote_v}")
                if 1.0 <= cote_v <= 999.0:
                    h2["cote_pmu"] = cote_v
                else:
                    h2["cote_pmu"] = None
            except (ValueError, TypeError):
                h2["cote_pmu"] = None

        # 4. Gains — supprimer espaces et caractères non numériques
        gains = h2.get("gains")
        if gains is not None:
            try:
                gains_str = re.sub(r"[^\d]", "", str(gains))
                if gains_str:
                    h2["gains"] = int(gains_str)
            except (ValueError, TypeError):
                h2["gains"] = None

        # 5. Pourcentages — clamper entre 0 et 100
        for pct_field in ("reussite_driver", "reussite_entraineur"):
            v = h2.get(pct_field)
            if v is not None:
                try:
                    pct = float(str(v).replace(",", ".").replace("%", ""))
                    h2[pct_field] = max(0.0, min(100.0, pct))
                except (ValueError, TypeError):
                    h2[pct_field] = None

        # 6. Musique — nettoyer caractères non hippiques
        for mus_field in ("musique", "musique_driver", "musique_entraineur"):
            m = h2.get(mus_field, "")
            if isinstance(m, str):
                # Garder uniquement chiffres, lettres D/M/A/a/m, parenthèses
                m_clean = re.sub(r"[^\dDMAdamap()]", "", m)
                h2[mus_field] = m_clean

        # 7. Distance plausible (1500-4500m)
        dist = h2.get("dist")
        if dist is not None:
            try:
                d = int(dist)
                if 1500 <= d <= 4500:
                    h2["dist"] = d
                else:
                    warnings.append(f"#{num}: distance suspecte {d}")
                    h2["dist"] = None
            except (ValueError, TypeError):
                h2["dist"] = None

        corrected.append(h2)

    return corrected, warnings


def merge_extracted_data(extractions: list) -> dict:
    """Fusionne les données extraites de plusieurs images avec validation."""
    merged = {}
    table_types = []
    all_warnings = []

    for ext in extractions:
        if not ext.get("chevaux"):
            continue
        if "table_type" in ext:
            table_types.append(ext["table_type"])

        # Validation + correction
        validated, warnings = validate_and_correct_horses(ext["chevaux"])
        all_warnings.extend(warnings)

        for horse in validated:
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
                    # Priorité aux valeurs non-vides existantes
                    if key not in merged[num] or merged[num][key] is None or merged[num][key] == "":
                        merged[num][key] = val

    chevaux_list = sorted(merged.values(), key=lambda x: x.get("numero", 99))
    return {
        "chevaux": chevaux_list,
        "nb_partants": len(chevaux_list),
        "table_types_detectes": list(set(table_types)),
        "warnings": all_warnings,
    }
