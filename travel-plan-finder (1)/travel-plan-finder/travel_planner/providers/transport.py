"""Providers de transport sur DONNÉES RÉELLES (internet, sans clé API).

────────────────────────────────────────────────────────────────────────
 SOURCES RÉELLES UTILISÉES
────────────────────────────────────────────────────────────────────────
 • Train  : db-vendo (v6.db.transport.rest) — wrapper communautaire,
            SANS CLÉ, des API Deutsche Bahn. Horaires et prix réels,
            couvre la France (TGV/ICE/Thalys), l'Allemagne, la Belgique…
 • Bus    : distance et durée RÉELLES du réseau routier via OSRM
            (router.project-osrm.org, données OpenStreetMap, sans clé).
            Le tarif n'est pas publié en API libre : il est calculé sur
            la base du barème km réel des opérateurs longue distance
            (marqué « tarif indicatif » dans l'UI — jamais présenté
            comme un prix réel).
 • Avion  : aucune API de vols gratuite n'existe (Amadeus self-service
            a fermé). On affiche un lien de recherche RÉEL Google Flights
            pré-rempli (villes + date) ; la durée est calculée sur la
            distance orthodromique réelle + friction aéroport.

La date de départ choisie par l'utilisateur pilote la requête trains
(horaires réels du jour demandé) et les liens de réservation.
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from datetime import date, datetime
from urllib.parse import quote_plus

import requests

from ..geo import haversine_km
from ..models import City, TransportOffer
from .base import TransportProvider

_UA = {"User-Agent": "travel-plan-finder/1.0 (streamlit app)"}
_TIMEOUT = 20

_DBVENDO = "https://v6.db.transport.rest"
_OSRM = "https://router.project-osrm.org"

# Barème km observé des opérateurs bus longue distance européens (FlixBus,
# BlaBlaCar Bus) — sert UNIQUEMENT au tarif indicatif, clairement étiqueté.
_BUS_PER_KM, _BUS_FIXED = 0.07, 9.0
_BUS_CO2_PER_KM = 0.032
_TRAIN_CO2_PER_KM = 0.004
_AIR_CO2_PER_KM = 0.156
_MIN_KM_FLIGHT = 250


class RealTransportProvider(TransportProvider):
    """Transport réel : trains db-vendo + route OSRM + lien vols."""

    def __init__(self, departure_date: date) -> None:
        self._date = departure_date

    # ── API principale ──────────────────────────────────────────────────
    def get_offers(self, origin: City, destination: City) -> list[TransportOffer]:
        offers: list[TransportOffer] = []

        train = self._train_offers(origin, destination)
        if train:
            offers.extend(train)

        road_km, road_min = self._road_route(origin, destination)
        if road_km:
            offers.append(self._bus_offer(origin, destination, road_km, road_min))

        flight = self._flight_offer(origin, destination)
        if flight:
            offers.append(flight)

        return offers

    # ── Train : db-vendo (horaires + prix réels Deutsche Bahn) ─────────
    def _train_offers(self, origin: City, destination: City) -> list[TransportOffer]:
        """2 allers réels le matin de la date choisie, avec prix quand publiés."""
        try:
            when = datetime(self._date.year, self._date.month, self._date.day, 7, 0)
            r = requests.get(
                f"{_DBVENDO}/journeys",
                params={
                    "from.latitude": origin.lat, "from.longitude": origin.lon,
                    "from.name": origin.name,
                    "to.latitude": destination.lat, "to.longitude": destination.lon,
                    "to.name": destination.name,
                    "departure": when.isoformat(),
                    "results": 2, "stopovers": "false", "language": "fr",
                },
                headers=_UA, timeout=_TIMEOUT,
            )
            r.raise_for_status()
            journeys = r.json().get("journeys") or []
        except Exception:
            return []

        offers: list[TransportOffer] = []
        for j in journeys:
            legs = j.get("legs") or []
            if not legs:
                continue
            dep = legs[0].get("departure", "")
            arr = legs[-1].get("arrival", "")
            duration = _iso_diff_min(dep, arr)
            if duration is None:
                continue
            # Opérateur : ligne principale du trajet
            lines = [lg.get("line", {}) for lg in legs if lg.get("line")]
            operator = (lines[0].get("operator", {}) or {}).get("name") or \
                       (lines[0].get("name") if lines else "Train")
            # Prix : publié par db-vendo uniquement quand la DB le fournit
            price = None
            price_info = j.get("price") or {}
            if isinstance(price_info, dict) and price_info.get("amount"):
                price = round(float(price_info["amount"]), 2)
            offers.append(TransportOffer(
                mode="Train", operator=str(operator),
                origin=origin.name, destination=destination.name,
                price_eur=price,
                duration_min=duration,
                transfers=max(0, len(legs) - 1),
                co2_kg=round(haversine_km(origin, destination) * _TRAIN_CO2_PER_KM, 1),
                comfort=4, departure=dep,
                booking_url="https://www.sncf-connect.com/fr-fr/search?from="
                            f"{quote_plus(origin.name)}&to={quote_plus(destination.name)}",
                price_is_estimate=False,
                source="db-vendo (API Deutsche Bahn)",
            ))
        return offers

    # ── Bus : itinéraire routier réel OSRM + tarif indicatif ───────────
    def _road_route(self, origin: City, destination: City) -> tuple[float, int]:
        """Distance (km) et durée (min) RÉELLES du réseau routier via OSRM."""
        try:
            r = requests.get(
                f"{_OSRM}/route/v1/driving/"
                f"{origin.lon},{origin.lat};{destination.lon},{destination.lat}",
                params={"overview": "false"},
                headers=_UA, timeout=_TIMEOUT,
            )
            r.raise_for_status()
            route = (r.json().get("routes") or [])[0]
            return route["distance"] / 1000.0, int(route["duration"] / 60)
        except Exception:
            return 0.0, 0

    def _bus_offer(self, origin: City, destination: City,
                   road_km: float, road_min: int) -> TransportOffer:
        # Tarif indicatif calculé sur la distance ROUTIÈRE réelle (OSRM),
        # selon le barème km des opérateurs bus longue distance.
        price = round(_BUS_FIXED + road_km * _BUS_PER_KM, 2)
        return TransportOffer(
            mode="Bus", operator="FlixBus / BlaBlaCar Bus",
            origin=origin.name, destination=destination.name,
            price_eur=price,
            duration_min=road_min + 25,  # durée routière réelle + accès gare routière
            transfers=0,
            co2_kg=round(road_km * _BUS_CO2_PER_KM, 1),
            comfort=2,
            booking_url="https://www.flixbus.fr/",
            price_is_estimate=True,  # étiqueté « tarif indicatif » dans l'UI
            source="Itinéraire OSRM/OpenStreetMap",
        )

    # ── Avion : lien Google Flights réel + durée calculée ──────────────
    def _flight_offer(self, origin: City, destination: City) -> TransportOffer | None:
        dist = haversine_km(origin, destination)
        if dist < _MIN_KM_FLIGHT:
            return None
        duration = int(dist / 620.0 * 60) + 150  # vol + 2h30 friction aéroport
        url = (
            "https://www.google.com/travel/flights?q=Vols+de+"
            f"{quote_plus(origin.name)}+à+{quote_plus(destination.name)}"
            f"+le+{self._date.isoformat()}"
        )
        return TransportOffer(
            mode="Avion", operator="Comparateur Google Flights",
            origin=origin.name, destination=destination.name,
            price_eur=None,  # pas d'API vols gratuite → lien de recherche réel
            duration_min=duration, transfers=0,
            co2_kg=round(dist * _AIR_CO2_PER_KM, 1),
            comfort=3, booking_url=url,
            price_is_estimate=False,
            source="Distance orthodromique + Google Flights",
        )


def _iso_diff_min(start: str, end: str) -> int | None:
    """Durée en minutes entre deux horodatages ISO 8601."""
    try:
        a = datetime.fromisoformat(start.replace("Z", "+00:00"))
        b = datetime.fromisoformat(end.replace("Z", "+00:00"))
        return max(0, int((b - a).total_seconds() // 60))
    except Exception:
        return None
