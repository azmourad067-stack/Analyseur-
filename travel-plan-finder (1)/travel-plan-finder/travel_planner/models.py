"""Structures de données du moteur de recherche."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class City:
    """Lieu géocodé via une API réelle (Photon / Nominatim)."""
    name: str
    country: str
    lat: float
    lon: float


@dataclass(frozen=True)
class TransportOffer:
    """Offre de transport aller simple, par personne."""
    mode: str                    # "Bus", "Train", "Avion"
    operator: str
    origin: str
    destination: str
    price_eur: Optional[float]   # None = prix non disponible publiquement
    duration_min: int            # porte-à-porte estimé
    transfers: int
    co2_kg: float
    comfort: int                 # 1 à 5
    departure: str = ""          # ISO datetime du départ (si horaire réel)
    booking_url: str = ""        # lien de réservation réel
    price_is_estimate: bool = False  # True = tarif indicatif calculé sur distance réelle
    source: str = ""             # ex. "db-vendo (DB)", "OSRM/OpenStreetMap"


@dataclass(frozen=True)
class Accommodation:
    """Hébergement réel, prix par nuit."""
    kind: str                    # "Hôtel" ou "Airbnb"
    name: str
    city: str
    price_per_night_eur: Optional[float]
    stars: Optional[int] = None
    rating: Optional[float] = None
    distance_km: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    website: str = ""
    booking_url: str = ""
    price_is_estimate: bool = False
    source: str = ""             # ex. "OpenStreetMap (Overpass)"


@dataclass
class TripPlan:
    """Combinaison transport + hébergement évaluée par le moteur."""
    transport: TransportOffer
    accommodation: Accommodation
    nights: int
    travelers: int
    total_cost_eur: float
    score: float = 0.0
    badges: list[str] = field(default_factory=list)
    cost_is_estimate: bool = False

    @property
    def transport_total(self) -> float:
        return (self.transport.price_eur or 0) * 2 * self.travelers

    @property
    def lodging_total(self) -> float:
        return (self.accommodation.price_per_night_eur or 0) * self.nights


@dataclass
class SearchResult:
    """Résultat complet d'une recherche."""
    plans: list[TripPlan]
    airbnb_url: str = ""          # lien de recherche Airbnb réel (pas d'API publique)
    sources: list[str] = field(default_factory=list)  # provenance des données
    warnings: list[str] = field(default_factory=list)  # limites éventuelles
    departure_date: Optional[date] = None
