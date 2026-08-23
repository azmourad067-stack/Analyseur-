"""Couche interface Streamlit.

Seul module qui dépend de Streamlit : aucune logique métier ici.
Affiche exclusivement des données provenant d'API réelles (géocodage
Photon/Nominatim, trains db-vendo, route OSRM, hôtels Overpass).
Les données non publiables en API libre (prix des vols, tarifs hôteliers)
sont remplacées par des liens réels ou marquées « indicatif ».
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from .engine import SearchCriteria, SearchError, search_plans
from .models import SearchResult, TripPlan

MODE_ICONS = {"Bus": "🚌", "Train": "🚄", "Avion": "✈️"}

_CSS = """
<style>
.plan-card {
    border: 1px solid #e3e6ea; border-radius: 14px;
    padding: 16px 20px; margin-bottom: 14px; background: #ffffff;
    box-shadow: 0 1px 4px rgba(0,0,0,.05);
}
.plan-card.top { border: 2px solid #1f9d55; background: #f6fffa; }
.plan-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 2px; }
.badge {
    display: inline-block; background: #eef6ee; color: #1f7a3d;
    border-radius: 999px; padding: 2px 10px; font-size: .78rem;
    font-weight: 600; margin-left: 6px;
}
.score-bar-outer { background:#eef0f2; border-radius:6px; height:10px; width:100%; }
.score-bar-inner { background:linear-gradient(90deg,#f4b400,#1f9d55);
    height:10px; border-radius:6px; }
.plan-meta { color:#5f6368; font-size:.88rem; margin-top:6px; }
.price-big { font-size:1.35rem; font-weight:800; color:#1a73e8; }
.tag-est { color:#b06000; font-size:.78rem; font-style:italic; }
.src { color:#80868b; font-size:.75rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# ── Formulaire de recherche ──────────────────────────────────────────────────

def render_search_form() -> SearchCriteria | None:
    with st.form("search_form", border=True):
        st.subheader("🔍 Rechercher un voyage")

        col1, col2 = st.columns(2)
        with col1:
            origin = st.text_input(
                "Ville de départ", value="Paris",
                help="N'importe quelle ville du monde (géocodage OpenStreetMap).",
            )
        with col2:
            destination = st.text_input("Destination", value="Lyon")

        col_date, col_n = st.columns(2)
        with col_date:
            departure_date = st.date_input(
                "📅 Date de départ",
                value=date.today() + timedelta(days=14),
                min_value=date.today(),
                help="Pilote les horaires de trains réels et les liens de réservation.",
            )
        with col_n:
            nights = st.number_input("Nuits", 1, 30, 3)
            travelers = st.number_input("Voyageurs", 1, 9, 2)

        col3, col4 = st.columns(2)
        with col3:
            budget = st.number_input(
                "Budget total cible (€)", min_value=50, max_value=10000,
                value=600, step=25,
                help="Transport aller-retour + hébergement.",
            )
            tolerance = st.slider(
                "Tolérance ± (€)", 0, 500, 100, step=25,
                help="Marge acceptée au-delà du budget cible.",
            )
        with col4:
            radius = st.slider(
                "Rayon hébergement (km)", 1, 30, 10,
                help="Distance réelle (GPS) entre l'hôtel et le centre d'arrivée.",
            )
            kinds_choice = st.selectbox(
                "Type d'hébergement", ["Hôtel + Airbnb", "Hôtel", "Airbnb"],
            )

        stars_disabled = "Hôtel" not in kinds_choice
        min_stars = st.select_slider(
            "Étoiles minimum (hôtels)", options=[1, 2, 3, 4, 5], value=3,
            disabled=stars_disabled,
            help="Actif uniquement quand « Hôtel » est inclus."
            if not stars_disabled else "Sélectionnez « Hôtel » pour activer ce filtre.",
        )

        submitted = st.form_submit_button(
            "🚀 Trouver les meilleurs plans", use_container_width=True, type="primary"
        )

    if not submitted:
        return None

    kinds = ["Hôtel", "Airbnb"] if "+" in kinds_choice else [kinds_choice]
    return SearchCriteria(
        origin_name=origin.strip(), destination_name=destination.strip(),
        departure_date=departure_date,
        budget_eur=float(budget), tolerance_eur=float(tolerance),
        radius_km=float(radius), lodging_kinds=kinds,
        min_stars=int(min_stars), nights=int(nights), travelers=int(travelers),
    )


# ── Résultats ────────────────────────────────────────────────────────────────

def run_and_render(criteria: SearchCriteria) -> None:
    with st.spinner("Interrogation des sources en direct (géocodage, trains, route, hôtels)…"):
        try:
            result = search_plans(criteria)
        except SearchError as e:
            st.warning(f"⚠️ {e}")
            return

    for w in result.warnings:
        st.info(f"ℹ️ {w}")

    _render_summary(result, criteria)
    _render_table(result)
    _render_cards(result, criteria)
    _render_sources(result)


def _render_summary(result: SearchResult, criteria: SearchCriteria) -> None:
    priced = [p for p in result.plans if p.total_cost_eur > 0]
    st.divider()
    st.subheader(
        f"✅ {len(priced)} plans — {criteria.origin_name} → {criteria.destination_name}"
        f" · départ le {criteria.departure_date.strftime('%d/%m/%Y')}"
    )
    if not priced:
        st.info("Aucun plan tarifé dans le budget, mais des options de transport sont listées ci-dessous.")
        return
    best = priced[0]
    cheapest = min(priced, key=lambda p: p.total_cost_eur)
    fastest = min(priced, key=lambda p: p.transport.duration_min)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 Meilleur score", f"{best.score}/100", f"{best.total_cost_eur:.0f} €")
    c2.metric("💰 Moins cher", f"{cheapest.total_cost_eur:.0f} €",
              f"{MODE_ICONS.get(cheapest.transport.mode, '')} {cheapest.transport.mode}")
    c3.metric("⚡ Plus rapide", _fmt_duration(fastest.transport.duration_min),
              f"{fastest.total_cost_eur:.0f} €")
    c4.metric("Budget max autorisé", f"{criteria.budget_max:.0f} €")


def _render_table(result: SearchResult) -> None:
    """Tableau comparatif (remplace le graphique, jugé non nécessaire)."""
    priced = [p for p in result.plans if p.total_cost_eur > 0]
    if not priced:
        return
    with st.expander("📊 Tableau comparatif détaillé", expanded=False):
        df = pd.DataFrame([{
            "Score": p.score,
            "Transport": f"{MODE_ICONS.get(p.transport.mode, '')} {p.transport.operator}",
            "Durée": _fmt_duration(p.transport.duration_min),
            "Corresp.": p.transport.transfers,
            "CO₂ (kg)": p.transport.co2_kg,
            "Hébergement": (
                f"{p.accommodation.name} ({'⭐' * p.accommodation.stars})"
                if p.accommodation.stars else p.accommodation.name
            ),
            "Distance centre": f"{p.accommodation.distance_km} km",
            "Coût total (€)": p.total_cost_eur,
        } for p in priced])
        st.dataframe(
            df, use_container_width=True, hide_index=True,
            column_config={
                "Coût total (€)": st.column_config.NumberColumn(format="%.0f €"),
                "Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
            },
        )


def _render_cards(result: SearchResult, criteria: SearchCriteria, top_n: int = 8) -> None:
    priced = [p for p in result.plans if p.total_cost_eur > 0]
    unpriced = [p for p in result.plans if p.total_cost_eur == 0]

    if priced:
        st.subheader(f"🥇 Top {min(top_n, len(priced))} recommandations")
        for rank, plan in enumerate(priced[:top_n], start=1):
            _plan_card(plan, criteria, rank)

    # Options de transport sans prix public (ex. avion → lien réel)
    if unpriced:
        st.subheader("✈️ Options sans tarif public en API libre")
        st.caption(
            "Aucune API de vols gratuite n'existe : le prix réel est à consulter "
            "via le lien de réservation ci-dessous."
        )
        for plan in unpriced:
            t = plan.transport
            st.markdown(
                f"""<div class="plan-card">
  <div class="plan-title">{MODE_ICONS.get(t.mode, '')} {t.operator}</div>
  <div class="plan-meta">🕒 ~{_fmt_duration(t.duration_min)} porte-à-porte
    &nbsp;·&nbsp; 🌱 {t.co2_kg} kg CO₂ &nbsp;·&nbsp; <span class="src">Source : {t.source}</span></div>
</div>""",
                unsafe_allow_html=True,
            )
            if t.booking_url:
                st.link_button(f"🔎 Prix et horaires réels — {t.mode}", t.booking_url)

    # Lien Airbnb réel
    if result.airbnb_url:
        st.subheader("🏠 Airbnb")
        st.caption(
            "Airbnb n'a pas d'API publique : voici la recherche réelle pré-remplie "
            "(destination, dates, voyageurs)."
        )
        st.link_button("🔎 Voir les logements Airbnb réels à ces dates", result.airbnb_url)


def _plan_card(plan: TripPlan, criteria: SearchCriteria, rank: int) -> None:
    t, a = plan.transport, plan.accommodation
    badges = " ".join(f'<span class="badge">{b}</span>' for b in plan.badges)
    stars = f'{"⭐" * a.stars} ' if a.stars else ""
    width = max(5, int(plan.score))
    est = ' <span class="tag-est">(prix indicatif)</span>' if plan.cost_is_estimate else ""
    lodging_line = (
        f"hébergement {plan.lodging_total:.0f} € — {criteria.nights} nuits"
        if a.price_per_night_eur is not None else "hébergement non inclus"
    )
    st.markdown(
        f"""
<div class="plan-card {'top' if rank == 1 else ''}">
  <div class="plan-title">#{rank} — {MODE_ICONS.get(t.mode, '')} {t.operator} · {stars}{a.name}{badges}</div>
  <div class="plan-meta">
    🕒 {_fmt_duration(t.duration_min)} porte-à-porte
    &nbsp;·&nbsp; 🔁 {t.transfers} correspondance{"s" if t.transfers else ""}
    &nbsp;·&nbsp; 🌱 {t.co2_kg} kg CO₂
    {f"&nbsp;·&nbsp; 📍 {a.distance_km} km du centre" if a.distance_km else ""}
  </div>
  <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px;">
    <div>
      <span class="price-big">{plan.total_cost_eur:.0f} €</span>{est}
      <span class="plan-meta">(transport {plan.transport_total:.0f} € + {lodging_line})</span>
    </div>
    <div style="min-width:160px;">
      <div class="plan-meta" style="text-align:right;">Score {plan.score}/100</div>
      <div class="score-bar-outer"><div class="score-bar-inner" style="width:{width}%;"></div></div>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns([1, 1, 4])
    if t.booking_url:
        cols[0].link_button("🎫 Réserver transport", t.booking_url, key=f"t{rank}")
    if a.booking_url:
        cols[1].link_button("🏨 Réserver hôtel", a.booking_url, key=f"h{rank}")


def _render_sources(result: SearchResult) -> None:
    """Transparence totale sur la provenance des données réelles."""
    with st.expander("🔌 Provenance des données (temps réel)"):
        for s in result.sources:
            st.markdown(f"- {s}")
        st.caption(
            "Aucune donnée simulée. Les prix marqués « indicatif » sont calculés "
            "sur des distances réelles car aucune API publique gratuite ne publie "
            "les tarifs (voir README pour brancher une API partenaire payante)."
        )


def _fmt_duration(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    return f"{h} h {m:02d}" if h else f"{m} min"


def render_sidebar() -> None:
    with st.sidebar:
        st.header("ℹ️ À propos")
        st.markdown(
            "**Travel Plan Finder** interroge en direct des sources **réelles** : "
            "géocodage OpenStreetMap, horaires de trains (API Deutsche Bahn), "
            "routes réelles (OSRM), hôtels OpenStreetMap."
        )
        st.markdown("**Score** : prix 45 % · temps 25 % · confort 15 % · CO₂ 15 %.")
        st.divider()
        st.caption(
            "Pas de données simulées. Prix vols/hôtels non publiés en API libre → "
            "liens de réservation réels ou mention « indicatif »."
        )
