"""Géocodage RÉEL et utilitaires géographiques.

Aucune liste de villes codée en dur : n'importe quelle ville du monde est
résolue en coordonnées GPS via deux API publiques gratuites :
  1. Photon (photon.komoot.io)     — géocodeur OpenStreetMap, sans clé
  2. Nominatim (nominatim.openstreetmap.org) — secours, 1 req/s max, User-Agent requis
"""

from __future__ import annotations

import math

import requests

from .models import City

_UA = {"User-Agent": "travel-plan-finder/1.0 (streamlit app)"}
_TIMEOUT = 12


def geocode(name: str) -> City | None:
    """Résout un nom de lieu en coordonnées réelles. None si introuvable."""
    city = _photon(name) or _nominatim(name)
    return city


def haversine_km(a: City, b: City) -> float:
    """Distance grand cercle (orthodromique) entre deux points."""
    r = 6371.0
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dp = math.radians(b.lat - a.lat)
    dl = math.radians(b.lon - a.lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def distance_km_coords(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return haversine_km(City("", "", lat1, lon1), City("", "", lat2, lon2))


def _photon(name: str) -> City | None:
    """Géocodage via Photon (données OpenStreetMap)."""
    try:
        r = requests.get(
            "https://photon.komoot.io/api/",
            params={"q": name, "limit": 1, "lang": "fr"},
            headers=_UA, timeout=_TIMEOUT,
        )
        r.raise_for_status()
        features = r.json().get("features") or []
        if not features:
            return None
        props = features[0]["properties"]
        lon, lat = features[0]["geometry"]["coordinates"]
        label = props.get("city") or props.get("name") or name
        country = props.get("country", "")
        return City(name=str(label), country=str(country), lat=float(lat), lon=float(lon))
    except Exception:
        return None


def _nominatim(name: str) -> City | None:
    """Secours : Nominatim (politique OSM : User-Agent obligatoire)."""
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": name, "format": "json", "limit": 1},
            headers=_UA, timeout=_TIMEOUT,
        )
        r.raise_for_status()
        results = r.json()
        if not results:
            return None
        hit = results[0]
        label = hit.get("name") or hit.get("display_name", name).split(",")[0]
        return City(name=label, country="", lat=float(hit["lat"]), lon=float(hit["lon"]))
    except Exception:
        return None
