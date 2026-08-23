"""Scoring qualité/prix des plans de voyage.

Adapté aux données réelles : certains champs (prix des vols, notes clients
des hôtels OSM) peuvent être indisponibles — le scoring se réduit alors
élégamment sur les critères réellement connus, avec repondération.

  - 45 %  Prix total     — critère n°1 d'achat
  - 25 %  Temps total    — porte-à-porte, pas seulement le trajet pur
  - 15 %  Confort        — différence forte entre bus et train
  - 15 %  Empreinte CO2  — critère croissant
"""

from __future__ import annotations

from .models import TripPlan

W_PRICE, W_TIME, W_COMFORT, W_CO2 = 0.45, 0.25, 0.15, 0.15


def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def assign_scores(plans: list[TripPlan]) -> None:
    """Calcule et affecte `score` (0-100) à chaque plan, en place."""
    if not plans:
        return

    # Plans sans prix (ex. avion → lien Google Flights) : exclus du critère
    # prix, leur score repose sur temps/confort/CO2 repondérés.
    n_time = _minmax([p.transport.duration_min for p in plans])
    n_comf = _minmax([p.transport.comfort for p in plans])
    n_co2 = _minmax([p.transport.co2_kg for p in plans])

    priced = [p.total_cost_eur for p in plans]
    n_cost = _minmax(priced)

    for i, plan in enumerate(plans):
        if plan.transport.price_eur is not None:
            score = (
                W_PRICE * (1 - n_cost[i])
                + W_TIME * (1 - n_time[i])
                + W_COMFORT * n_comf[i]
                + W_CO2 * (1 - n_co2[i])
            )
        else:
            # Repondération sans le critère prix
            w = W_TIME + W_COMFORT + W_CO2
            score = (
                W_TIME * (1 - n_time[i])
                + W_COMFORT * n_comf[i]
                + W_CO2 * (1 - n_co2[i])
            ) / w
        plan.score = round(100 * score, 1)


def assign_badges(plans: list[TripPlan]) -> None:
    """Étiquettes d'aide à la décision."""
    if not plans:
        return

    best_score = max(plans, key=lambda p: p.score)
    best_score.badges.append("🏆 Meilleur choix")

    priced = [p for p in plans if p.transport.price_eur is not None]
    if priced:
        cheapest = min(priced, key=lambda p: p.total_cost_eur)
        if cheapest is not best_score:
            cheapest.badges.append("💰 Moins cher")

    fastest = min(plans, key=lambda p: p.transport.duration_min)
    if fastest is not best_score and (not priced or fastest is not min(priced, key=lambda p: p.total_cost_eur)):
        fastest.badges.append("⚡ Plus rapide")

    greenest = min(plans, key=lambda p: p.transport.co2_kg)
    if not greenest.badges:
        greenest.badges.append("🌱 Éco-responsable")
