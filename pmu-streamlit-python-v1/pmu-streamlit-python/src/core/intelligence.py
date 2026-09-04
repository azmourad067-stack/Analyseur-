import math

WEIGHTS_BY_DISCIPLINE = {
    "TROT": {"FORME": .20, "ENTOURAGE": .30, "CONFIANCE": .15, "CONFIGURATION": .20, "APTITUDE": .05, "EXPERT": .10},
    "PLAT": {"FORME": .45, "ENTOURAGE": .10, "CONFIANCE": .15, "CONFIGURATION": .05, "APTITUDE": .20, "EXPERT": .05},
    "OBSTACLE": {"FORME": .40, "ENTOURAGE": .15, "CONFIANCE": .10, "CONFIGURATION": .05, "APTITUDE": .05, "EXPERT": .25},
    "DEFAULT": {"FORME": .30, "ENTOURAGE": .20, "CONFIANCE": .10, "CONFIGURATION": .10, "APTITUDE": .10, "EXPERT": .20},
}


def _form_score(musique, discipline="INCONNUE"):
    if not musique: return 20
    import re
    clean = re.sub(r"\(\d+\)", "", str(musique))
    perfs = re.findall(r"([0-9DA]|Dist)[a-zA-Z]", clean)[:6]
    if not perfs: return 30
    is_trot = any(x in discipline for x in ("TROT", "ATTELE", "MONTE"))
    score = total = 0.0
    for i, perf in enumerate(perfs):
        typ, val = perf[-1].lower(), perf[:-1].upper()
        points = 20
        if val.isdigit():
            place = int(val)
            points = {1:100,2:80,3:65,4:50,5:40}.get(place, 20 if place <= 9 else 5)
        elif val in {"D", "DIST"}: points = 0 if is_trot else 10
        elif val in {"A", "ARR", "T", "TB"}: points = 5
        current = "m" if is_trot and "MONTE" in discipline else ("a" if is_trot else ("p" if "PLAT" in discipline else ("h" if "HAIE" in discipline else "s")))
        if typ == current: points *= 1.1
        w = .85 ** i
        score += min(100, points) * w
        total += w
    return round(score / total)


def _classe(p):
    age = int(p.get("age") or 5); gains = float(p.get("gains") or 0)
    if age < 2: return 50
    return min(round((gains / (age * 12000)) * 45), 100) or 50


def _config(p, discipline):
    if not any(x in discipline for x in ("TROT", "ATTELE", "MONTE")): return 50
    f = str(p.get("ferrage") or "").upper()
    if "D4" in f: return 95
    if "DA" in f or "DP" in f: return 75
    if "PL" in f: return 60
    return 40


def changement_categorie(p, prix_course):
    n = int(p.get("nb_courses") or 0)
    if n < 3: return "STABLE"
    avg = float(p.get("gains") or 0) / n
    if avg > prix_course * .7: return "DESCENTE"
    if avg < prix_course / 12: return "MONTEE"
    return "STABLE"


def regularite(p):
    n = int(p.get("nb_courses") or 0)
    return round((((p.get("nb_victoires") or 0) + (p.get("nb_places") or 0)) / n) * 100) if n else 0


def expert_impact(p, contexte):
    bonus = 0; prix = contexte.get("prixCourse") or 20000
    cat = changement_categorie(p, prix)
    if cat == "DESCENTE": bonus += 20
    elif cat == "MONTEE": bonus -= 10
    if regularite(p) > 50: bonus += 15
    if p.get("oeilleres") and p["oeilleres"] != "SANS_OEILLERES": bonus += 10
    return max(0, min(100, 50 + bonus))


def calculer_prediction(p, contexte_course):
    disc = str(contexte_course.get("discipline") or "PLAT").upper()
    key = "TROT" if any(x in disc for x in ("TROT", "ATTELE", "MONTE")) else ("PLAT" if "PLAT" in disc else ("OBSTACLE" if any(x in disc for x in ("OBSTACLE", "HAIE", "STEEPLE")) else "DEFAULT"))
    w = WEIGHTS_BY_DISCIPLINE[key]
    top_drivers = ["BAZIRE","NIVARD","RAFFIN","ABRIVARD","ROCHARD","MOTTIER","TOMASELLI","GELORMINI"]
    top_jockeys = ["BARZALONA","SOUMILLON","GUYON","PASQUIER","DEMURO","PICCONE","LEMAITRE"]
    driver = str(p.get("driver") or "").upper()
    entourage = 95 if any(x in driver for x in (top_drivers if key == "TROT" else top_jockeys)) else 50
    cote = float(p.get("cote_ref") or 0)
    confiance = 95 if 0 < cote < 3 else (80 if cote < 6 else (60 if cote < 12 else (40 if cote < 25 else 20))) if cote else 50
    score = (_form_score(p.get("musique"), disc)*w["FORME"] + entourage*w["ENTOURAGE"] + confiance*w["CONFIANCE"] + _config(p,disc)*w["CONFIGURATION"] + _classe(p)*w["APTITUDE"] + expert_impact(p,contexte_course)*w["EXPERT"])
    return 50 if math.isnan(score) else round(score)
