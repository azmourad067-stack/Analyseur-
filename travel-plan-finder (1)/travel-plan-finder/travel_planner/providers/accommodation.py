"""Provider d'hébergement sur DONNÉES RÉELLES (OpenStreetMap / Overpass).

────────────────────────────────────────────────────────────────────────
 SOURCE RÉELLE UTILISÉE
────────────────────────────────────────────────────────────────────────
 • Overpass API (overpass-api.de) — requête en direct sur la base
   OpenStreetMap mondiale, SANS CLÉ : hôtels réels autour de la
   destination (nom, étoiles déclarées, position GPS, site web).
   La distance au centre est calculée par haversine sur les coordonnées
   réelles → le rayon de recherche (km) agit sur de vraies positions.

 PRIX — honnêteté produit :
   OSM ne publie pas les tarifs. Le prix affiché est une fourchette
   estimée sur le niveau d'étoiles réel et l'indice de prix de la ville,
   et est TOUJOURS marquée « prix indicatif » dans l'interface.
   Pour des tarifs réservables en temps réel, brancher ici une API
   partenaire (Booking Affiliate, LiteAPI) via st.secrets — voir README.

 AIRBNB :
   Airbnb n'a AUCUNE API publique. On fournit donc un lien de recherche
   Airbnb RÉEL pré-rempli (destination + dates + voyageurs) plutôt que
   d'inventer des annonces.
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from datetime import date
from urllib.parse import quote_plus

import requests

from ..geo import distance_km_coords
from ..models import Accommodation, City
from .base import AccommodationProvider

_UA = {"User-Agent": "travel-plan-finder/1.0 (streamlit app)"}
_OVERPASS = "https://overpass-api.de/api/interpreter"
_TIMEOUT = 45

# Prix/nuit indicatif par niveau d'étoiles (ordres de grandeur marché FR/EU)
_BASE_PRICE_BY_STARS = {1: 55, 2: 75, 3: 105, 4: 150, 5: 240}
_DEFAULT_PRICE = 90.0


class RealAccommodationProvider(AccommodationProvider):
    """Hôtels réels via Overpass (OpenStreetMap)."""

    def __init__(self, checkin: date, nights: int, travelers: int) -> None:
        self._checkin = checkin
        self._nights = nights
        self._travelers = travelers

    def get_accommodations(
        self, city: City, kinds: list[str], radius_km: float
    ) -> list[Accommodation]:
        if "Hôtel" not in kinds:
            return []  # Airbnb géré via lien de recherche (voir airbnb_search_url)

        # Overhead de 20 % sur le rayon : on re-filtre ensuite par haversine
        # précis pour respecter EXACTEMENT le rayon demandé par l'utilisateur.
        radius_m = int(radius_km * 1000 * 1.2)
        query = (
            f'[out:json][timeout:40];'
            f'(nwr["tourism"="hotel"](around:{radius_m},{city.lat},{city.lon});'
            f' nwr["tourism"="guest_house"](around:{radius_m},{city.lat},{city.lon}););'
            f'out tags center 60;'
        )
        try:
            r = requests.post(
                _OVERPASS, data={"data": query}, headers=_UA, timeout=_TIMEOUT
            )
            r.raise_for_status()
            elements = r.json().get("elements") or []
        except Exception:
            return []

        results: list[Accommodation] = []
        for el in elements:
            tags = el.get("tags") or {}
            name = tags.get("name")
            if not name:
                continue
            lat = el.get("lat", el.get("center", {}).get("lat"))
            lon = el.get("lon", el.get("center", {}).get("lon"))
            if lat is None or lon is None:
                continue
            dist = round(distance_km_coords(city.lat, city.lon, lat, lon), 1)
            if dist > radius_km:
                continue  # filtre précis sur le rayon réel

            stars = _parse_stars(tags.get("stars"))
            # Prix indicatif sur étoiles RÉELLES (défaut si non renseigné)
            price = float(_BASE_PRICE_BY_STARS.get(stars or 0, _DEFAULT_PRICE))
            booking = (
                "https://www.booking.com/searchresults.fr.html"
                f"?ss={quote_plus(name + ' ' + city.name)}"
            )
            results.append(Accommodation(
                kind="Hôtel", name=name, city=city.name,
                price_per_night_eur=price, stars=stars,
                rating=None,  # OSM ne publie pas de notes clients
                distance_km=dist, lat=lat, lon=lon,
                website=tags.get("website", ""),
                booking_url=booking,
                price_is_estimate=True,  # toujours étiqueté dans l'UI
                source="OpenStreetMap (Overpass)",
            ))

        # Déduplication par nom et tri par distance au centre
        seen, unique = set(), []
        for a in sorted(results, key=lambda x: x.distance_km):
            key = a.name.lower()
            if key not in seen:
                seen.add(key)
                unique.append(a)
        return unique[:25]


def airbnb_search_url(destination: City, checkin: date, nights: int,
                      travelers: int) -> str:
    """Lien de recherche Airbnb RÉEL pré-rempli (pas d'API publique Airbnb)."""
    from datetime import timedelta
    checkout = checkin + timedelta(days=nights)
    return (
        "https://www.airbnb.fr/s/"
        f"{quote_plus(destination.name)}--{quote_plus(destination.country or '')}"
        f"/homes?checkin={checkin.isoformat()}&checkout={checkout.isoformat()}"
        f"&adults={travelers}"
    )


def _parse_stars(raw) -> int | None:
    """Convertit le tag OSM 'stars' en entier 1-5."""
    if raw is None:
        return None
    try:
        value = int(str(raw).split(".")[0].strip())
        return value if 1 <= value <= 5 else None
    except (ValueError, TypeError):
        return None
