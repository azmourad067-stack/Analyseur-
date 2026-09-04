from datetime import datetime
from .intelligence import calculer_prediction


def process_races(raw_races, day_date, reunion_data=None):
    reunion_data = reunion_data or {}
    out = []
    for race in raw_races or []:
        dt = datetime.fromisoformat(race["heureDepart"].replace("Z", "+00:00")) if race.get("heureDepart") else None
        participants = []
        for p in race.get("participants") or []:
            obj = {
                "nom": p.get("nom") or "?", "numero": p.get("numPmu") or 0, "sexe": p.get("sexe") or "?", "age": p.get("age") or 0,
                "musique": p.get("musique") or "", "gains": ((p.get("gainsParticipant") or {}).get("gainsCarriere") or 0) / 100,
                "driver": p.get("driver") or p.get("jockey") or "?", "entraineur": p.get("entraineur") or "?", "proprietaire": p.get("proprietaire") or "?",
                "ferrage": p.get("deferre") or "STANDARD", "oeilleres": p.get("oeilleres") or "SANS_OEILLERES", "nb_courses": p.get("nombreCourses") or 0,
                "nb_victoires": p.get("nombreVictoires") or 0, "nb_places": p.get("nombrePlaces") or 0, "cote_ref": (p.get("dernierRapportDirect") or {}).get("rapport") or 0,
                "statut": p.get("statut") or "PARTANT", "classement": p.get("ordreArrivee"),
            }
            obj["prediction_score"] = calculer_prediction(obj, {"discipline": race.get("discipline") or "PLAT", "corde": race.get("corde"), "prixCourse": race.get("montantPrix") or 0})
            participants.append(obj)
        race_dt = dt.date().isoformat() if dt else day_date
        out.append({
            "date": race_dt, "heure": dt.strftime("%H:%M") if dt else "", "reunion_num": race.get("numReunion"), "course_num": race.get("numOrdre"),
            "hippodrome": (race.get("hippodrome") or {}).get("libelleLong", "Inconnu"), "corde": race.get("corde") or "?", "discipline": race.get("discipline") or "Inconnue",
            "distance": str(race.get("distance") or 0), "categorie": race.get("categorieParticularite") or "", "conditions": race.get("conditions") or "",
            "statut": race.get("statut") or "Inconnu", "partants": race.get("nombreDeclaresPartants") or 0, "prix": race.get("montantPrix") or 0,
            "meteo": reunion_data.get("meteo") or {}, "type_pari": ",".join(x.get("codePari", "") for x in race.get("paris") or []),
            "ordre_arrivee": ",".join("-".join(map(str, x)) for x in race.get("ordreArrivee") or []) or None,
            "rapports": race.get("rapportsDefinitifs"), "participants": participants,
        })
    return out


def process_day_races(raw_data, day_date, filter_options=None):
    races=[]
    for reunion in (raw_data or {}).get("programme", {}).get("reunions", []) or []:
        races.extend(process_races(reunion.get("courses", []), day_date, reunion))
    opts = filter_options or {}
    if opts.get("disciplines"):
        allowed = opts["disciplines"] if isinstance(opts["disciplines"], list) else [opts["disciplines"]]
        races = [r for r in races if r["discipline"] in allowed]
    return races
