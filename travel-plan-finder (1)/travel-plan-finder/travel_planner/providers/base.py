"""Interfaces des providers de données.

Pour passer d'une donnée mockée à une vraie API, il suffit de créer
une classe qui respecte ces interfaces, puis de l'injecter dans
`search_plans()` (voir engine.py). Aucun changement côté UI ni métier.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Accommodation, City, TransportOffer


class TransportProvider(ABC):
    """Source d'offres de transport entre deux villes."""

    @abstractmethod
    def get_offers(self, origin: City, destination: City) -> list[TransportOffer]:
        """Retourne les offres aller simple, par personne (liste vide si aucune)."""
        ...


class AccommodationProvider(ABC):
    """Source d'hébergements autour d'une ville."""

    @abstractmethod
    def get_accommodations(
        self,
        city: City,
        kinds: list[str],       # sous-ensemble de ["Hôtel", "Airbnb"]
        radius_km: float,       # rayon de recherche autour du centre
    ) -> list[Accommodation]:
        """Retourne les hébergements dans le rayon demandé (liste vide si aucun)."""
        ...
