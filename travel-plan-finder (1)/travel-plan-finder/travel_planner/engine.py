"""Logique métier : recherche, filtrage, combinaison et classement.

Couche purement métier — aucune dépendance à Streamlit. Toutes les
données proviennent d'API RÉELLES (voir providers/) ; aucune donnée
n'est simulée ou inventée. Quand une donnée n'existe pas en API libre
(prix des vols, prix des hôtels), elle est soit remplacée par un lien
de recherche réel, soit clairement marquée « indicatif ».
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .geo import geocode
from .models import SearchResult, TripPlan
from .providers.accommodation import RealAccommodationProvider, airbnb_search_url
from .providers.transport import RealTransportProvider
from .scoring import assign_badges, assign_scores


@dataclass(frozen=True)
class SearchCriteria:
    """Critères saisis par l'utilisateur."""
    origin_name: str
    destination_name: str
    departure_date: date
    budget_eur: float
    tolerance_eur: float      # marge +/- autour du budget cible
    radius_km: float          # rayon de recherche hébergement
    lodging_kinds: list[str]  # ["Hôtel"], ["Airbnb"] ou les deux
    min_stars: int            # filtre étoiles (hôtels uniquement)
    nights: int
    travelers: int

    @property
    def budget_max(self) -> float:
        return self.budget_eur + self.tolerance_eur


class SearchError(Exception):
    """Erreur métier à afficher telle quelle à l'utilisateur."""


def search_plans(criteria: SearchCriteria) -> SearchResult:
    """Point d'entrée du moteur. Lève SearchError avec un message clair."""
    # ── Validation des entrées ──────────────────────────────────────────
    if not criteria.origin_name.strip() or not criteria.destination_name.strip():
        raise SearchError("Veuillez renseigner une ville de départ et une destination.")
    if criteria.departure_date < date.today():
        raise SearchError("La date de départ ne peut pas être dans le passé.")
    if criteria.budget_eur <= 0:
        raise SearchError("Le budget doit être supérieur à 0 €.")
    if not criteria.lodging_kinds:
        raise SearchError("Sélectionnez au moins un type d'hébergement.")

    # ── Géocodage réel (Photon / Nominatim) ─────────────────────────────
    origin = geocode(criteria.origin_name)
    destination = geocode(criteria.destination_name)
    if origin is None:
        raise SearchError(f"Lieu de départ « {criteria.origin_name} » introuvable. Vérifiez l'orthographe.")
    if destination is None:
        raise SearchError(f"Destination « {criteria.destination_name} » introuvable. Vérifiez l'orthographe.")
    if abs(origin.lat - destination.lat) < 1e-3 and abs(origin.lon - destination.lon) < 1e-3:
        raise SearchError("Le départ et la destination doivent être différents.")

    # ── Transport réel (db-vendo trains + OSRM route + lien vols) ───────
    transport_provider = RealTransportProvider(criteria.departure_date)
    transports = transport_provider.get_offers(origin, destination)
    if not transports:
        raise SearchError(
            "Aucun transport trouvé pour cet itinéraire à cette date "
            "(les API publiques ne couvrent pas toutes les liaisons)."
        )

    # Avertissement transparence si l'API trains publique n'a rien renvoyé
    pre_warnings: list[str] = []
    if not any(t.mode == "Train" for t in transports):
        pre_warnings.append(
            "Aucun horaire de train renvoyé par l'API publique (db-vendo / Deutsche Bahn) "
            "pour cette liaison à cette date — service momentanément indisponible ou "
            "liaison non couverte. Les autres modes restent affichés."
        )

    # ── Hébergements réels (Overpass / OpenStreetMap) ───────────────────
    acc_provider = RealAccommodationProvider(
        criteria.departure_date, criteria.nights, criteria.travelers
    )
    lodgings = acc_provider.get_accommodations(
        destination, criteria.lodging_kinds, criteria.radius_km
    )
    lodgings = [
        a for a in lodgings
        if a.kind != "Hôtel" or (a.stars or 0) >= criteria.min_stars
    ]

    warnings: list[str] = pre_warnings
    if "Hôtel" in criteria.lodging_kinds and not lodgings:
        warnings.append(
            f"Aucun hôtel OpenStreetMap dans un rayon de {criteria.radius_km:.0f} km "
            f"avec {criteria.min_stars}⭐ minimum. Élargissez le rayon ou "
            f"baissez le filtre étoiles."
        )

    # ── Combinaison + filtrage budgétaire ───────────────────────────────
    plans: list[TripPlan] = []
    for t in transports:
        if t.price_eur is None:
            continue  # ex. avion : pas de prix public → carte dédiée dans l'UI
        if lodgings:
            for a in lodgings:
                if a.price_per_night_eur is None:
                    continue
                total = round(
                    t.price_eur * 2 * criteria.travelers
                    + a.price_per_night_eur * criteria.nights, 2
                )
                if total <= criteria.budget_max:
                    plans.append(TripPlan(
                        transport=t, accommodation=a,
                        nights=criteria.nights, travelers=criteria.travelers,
                        total_cost_eur=total,
                        cost_is_estimate=t.price_is_estimate or a.price_is_estimate,
                    ))
        else:
            # Pas d'hôtel trouvé : on présente quand même le transport seul
            total = round(t.price_eur * 2 * criteria.travelers, 2)
            if total <= criteria.budget_max:
                plans.append(TripPlan(
                    transport=t, accommodation=_transport_only_placeholder(destination),
                    nights=criteria.nights, travelers=criteria.travelers,
                    total_cost_eur=total,
                    cost_is_estimate=t.price_is_estimate,
                ))

    if not plans:
        candidates = [
            t.price_eur * 2 * criteria.travelers
            + min((a.price_per_night_eur or 1e9) for a in lodgings) * criteria.nights
            for t in transports if t.price_eur is not None
        ] if lodgings else [
            t.price_eur * 2 * criteria.travelers
            for t in transports if t.price_eur is not None
        ]
        hint = f" Le plan le moins cher trouvé coûte ~{min(candidates):.0f} €." if candidates else ""
        raise SearchError(
            f"Aucun plan dans votre budget ({criteria.budget_max:.0f} € max).{hint} "
            f"Augmentez le budget ou la tolérance."
        )

    # ── Scoring, badges, classement ─────────────────────────────────────
    assign_scores(plans)
    assign_badges(plans)
    plans.sort(key=lambda p: p.score, reverse=True)

    # Déduplication visuelle : max 3 plans par couple transport/hôtel identique
    # déjà garanti par les sources ; on limite l'affichage côté UI.

    sources = sorted({t.source for t in transports} |
                     {a.source for a in lodgings} |
                     {"Photon/Nominatim (géocodage OpenStreetMap)"})

    airbnb_url = ""
    if "Airbnb" in criteria.lodging_kinds:
        airbnb_url = airbnb_search_url(
            destination, criteria.departure_date, criteria.nights, criteria.travelers
        )

    # Plans avec transport sans prix (avion) : ajoutés en fin, pour l'UI
    no_price = [t for t in transports if t.price_eur is None]
    for t in no_price:
        plans.append(TripPlan(
            transport=t, accommodation=_transport_only_placeholder(destination),
            nights=criteria.nights, travelers=criteria.travelers,
            total_cost_eur=0.0, score=0.0,
        ))

    return SearchResult(
        plans=plans, airbnb_url=airbnb_url, sources=sources,
        warnings=warnings, departure_date=criteria.departure_date,
    )


def _transport_only_placeholder(city):
    from .models import Accommodation
    return Accommodation(kind="", name="(transport seul)", city=city.name,
                         price_per_night_eur=None, distance_km=0.0)
