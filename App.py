"""
TripPlanner – Meilleure combinaison transport + hébergement
Application Streamlit autonome (un seul fichier).
Recherche en temps réel sur Internet :
- Géocodage via Nominatim (avec repli Photon)
- Itinéraires routiers via OSRM (serveur public)
- Hébergements via Overpass API (OpenStreetMap, avec miroirs de secours)
Les tarifs transport/hébergement sont des estimations réalistes.
Aucune donnée simulée n'est utilisée pour les hébergements.
"""

import math
import re
from dataclasses import dataclass
from typing import List, Tuple
from datetime import date, timedelta

import streamlit as st
import pandas as pd
import requests
import numpy as np

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
PHOTON_URL = "https://photon.komoot.io/api/"
OSRM_URL = "https://router.project-osrm.org/route/v1/driving"

# Liste de miroirs Overpass (essayés dans l'ordre)
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

# Remplacez par votre véritable adresse email (obligatoire pour Nominatim)
USER_AGENT = "TripPlanner/1.0 (monadresse@example.com)"
TIMEOUT = 20
OVERPASS_TIMEOUT = 30

# ----------------------------------------------------------------------
# Modèles de données
# ----------------------------------------------------------------------
@dataclass
class Location:
    name: str
    lat: float
    lon: float
    display_name: str = ""

@dataclass
class TransportOption:
    mode: str
    provider: str
    description: str
    duration_min: float
    price_eur: float
    co2_kg: float
    score: float = 0.0

@dataclass
class AccommodationOption:
    name: str
    type: str              # "hotel" ou "airbnb"
    stars: int             # 0 pour les non-hôtels
    price_per_night: float
    distance_km: float
    lat: float
    lon: float
    address: str
    source: str
    url: str = ""
    score: float = 0.0

@dataclass
class TripOption:
    transport: TransportOption
    accommodation: AccommodationOption
    total_price: float
    total_duration_min: float
    score: float
    reasons: List[str]

# ----------------------------------------------------------------------
# Géocodage (Nominatim + repli Photon)
# ----------------------------------------------------------------------
def _geocode_nominatim(city: str) -> Location:
    """Géocodage via Nominatim."""
    params = {
        "q": city,
        "format": "json",
        "limit": 1,
        "addressdetails": 1,
    }
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "fr",
    }
    resp = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise ValueError(f"Lieu introuvable : {city}")
    item = data[0]
    return Location(
        name=city,
        lat=float(item["lat"]),
        lon=float(item["lon"]),
        display_name=item.get("display_name", city),
    )

def _geocode_photon(city: str) -> Location:
    """Géocodage via Photon (repli si Nominatim échoue)."""
    params = {
        "q": city,
        "limit": 1,
    }
    resp = requests.get(PHOTON_URL, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    features = data.get("features", [])
    if not features:
        raise ValueError(f"Lieu introuvable : {city}")
    feature = features[0]
    coords = feature["geometry"]["coordinates"]
    props = feature["properties"]
    return Location(
        name=city,
        lat=coords[1],
        lon=coords[0],
        display_name=props.get("name", city) + ", " + props.get("country", ""),
    )

@st.cache_data(ttl=3600, show_spinner=False)
def geocode(city: str) -> Location:
    """Géolocalise une ville, en essayant d'abord Nominatim puis Photon."""
    try:
        return _geocode_nominatim(city)
    except Exception as e:
        try:
            return _geocode_photon(city)
        except Exception as e2:
            raise ValueError(
                f"Impossible de géolocaliser '{city}' avec les services disponibles. "
                f"Erreur Nominatim : {e} ; Erreur Photon : {e2}"
            )

# ----------------------------------------------------------------------
# Itinéraires routiers (OSRM)
# ----------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def get_route_km_duration(lat1: float, lon1: float, lat2: float, lon2: float):
    """Calcule distance routière (km) et durée (min) via OSRM."""
    coords = f"{lon1},{lat1};{lon2},{lat2}"
    params = {"overview": "false", "steps": "false", "alternatives": "false"}
    resp = requests.get(f"{OSRM_URL}/{coords}", params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok":
        raise ValueError("Impossible de calculer l'itinéraire routier.")
    route = data["routes"][0]
    return route["distance"] / 1000.0, route["duration"] / 60.0

# ----------------------------------------------------------------------
# Génération des options de transport (estimations)
# ----------------------------------------------------------------------
def generate_transport_options(origin: Location, destination: Location, roundtrip: bool) -> List[TransportOption]:
    """Retourne les options de transport estimées (bus, train, avion)."""
    distance_km, driving_min = get_route_km_duration(
        origin.lat, origin.lon, destination.lat, destination.lon
    )
    options = []

    def add_option(mode, provider, duration_min, price, co2_per_km, description):
        co2 = co2_per_km * distance_km
        if roundtrip:
            duration_min *= 2
            price *= 2
            co2 *= 2
            description += " (aller-retour)"
        options.append(
            TransportOption(
                mode=mode,
                provider=provider,
                description=description,
                duration_min=round(duration_min, 1),
                price_eur=round(price, 2),
                co2_kg=round(co2, 2),
            )
        )

    # Bus longue distance (jusqu'à 2000 km)
    if distance_km <= 2000:
        bus_duration = driving_min * 1.35
        bus_price = 5 + distance_km * 0.09
        add_option("Bus", "FlixBus (estimation)", bus_duration, bus_price, 0.03, "Bus longue distance")

    # Train grandes lignes (toujours)
    train_duration = driving_min * 0.85
    train_price = 10 + distance_km * 0.16
    add_option("Train", "SNCF (estimation)", train_duration, train_price, 0.014, "Train grandes lignes")

    # Avion (au-delà de 300 km)
    if distance_km > 300:
        plane_duration = 120 + (distance_km / 780) * 60
        plane_price = 70 + distance_km * 0.10
        add_option("Avion", "Compagnie aérienne (estimation)", plane_duration, plane_price, 0.18, "Vol commercial")

    # Sécurité : toujours au moins une option train
    if not options:
        train_duration = driving_min * 0.85
        train_price = 10 + distance_km * 0.16
        add_option("Train", "SNCF (estimation)", train_duration, train_price, 0.014, "Train grandes lignes")

    return options

# ----------------------------------------------------------------------
# Recherche d'hébergements via OpenStreetMap Overpass API
# ----------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Distance en km entre deux points GPS."""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def parse_stars(tags: dict) -> int:
    """Extrait le nombre d'étoiles d'un hôtel à partir des tags OSM."""
    raw = tags.get("stars") or tags.get("hotel:stars") or tags.get("class")
    if raw:
        match = re.search(r"\d+", str(raw))
        if match:
            stars = int(match.group())
            return max(1, min(5, stars))
    name = tags.get("name", "")
    for i in range(5, 0, -1):
        if str(i) in name:
            return i
    return 3  # défaut

def estimate_price(stars: int, acc_type: str, distance_km: float) -> float:
    """Estimation du prix par nuit en fonction du type et des étoiles."""
    if acc_type == "hotel":
        base = {1: 45, 2: 60, 3: 85, 4: 130, 5: 260}
        price = base.get(stars, 85)
        if distance_km < 2:
            price *= 1.10
        elif distance_km > 10:
            price *= 0.90
        return round(price, 2)
    else:  # type airbnb-like
        price = 45 + 10 * (1 / (distance_km + 1))
        return round(price, 2)

def _query_overpass(query: str, attempt: int = 0) -> dict:
    """Envoie la requête Overpass en essayant plusieurs miroirs."""
    if attempt >= len(OVERPASS_URLS):
        raise RuntimeError("Tous les miroirs Overpass ont échoué.")
    url = OVERPASS_URLS[attempt]
    try:
        # Utilisation de GET avec paramètre data pour éviter certains problèmes de proxy
        resp = requests.get(
            url,
            params={"data": query},
            headers={"User-Agent": USER_AGENT},
            timeout=OVERPASS_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.warning(f"Échec du miroir Overpass {url} : {e}. Tentative suivante...")
        return _query_overpass(query, attempt + 1)

@st.cache_data(ttl=600, show_spinner=False)
def fetch_osm_accommodations(
    lat: float,
    lon: float,
    radius_km: int,
    acc_types: Tuple[str, ...],
) -> List[AccommodationOption]:
    """
    Interroge l'API Overpass (OpenStreetMap) pour récupérer les hébergements réels.
    Retourne une liste vide si aucun résultat ou si le service échoue.
    """
    statements = []
    radius_m = radius_km * 1000

    if "hotel" in acc_types:
        statements.append(f'  node["tourism"="hotel"](around:{radius_m},{lat},{lon});')
        statements.append(f'  way["tourism"="hotel"](around:{radius_m},{lat},{lon});')
    if "airbnb" in acc_types:
        for tag in ["apartment", "guest_house", "chalet"]:
            statements.append(f'  node["tourism"="{tag}"](around:{radius_m},{lat},{lon});')
            statements.append(f'  way["tourism"="{tag}"](around:{radius_m},{lat},{lon});')

    union = "\n".join(statements)
    query = f"""
[out:json][timeout:25];
(
{union}
);
out center tags 50;
"""

    try:
        data = _query_overpass(query)
    except Exception as e:
        st.error(f"Erreur lors de la récupération des hébergements : {e}")
        return []

    elements = data.get("elements", [])
    results = []

    for el in elements:
        tags = el.get("tags", {})
        el_lat = el.get("lat") or (el.get("center", {}).get("lat") if el.get("center") else None)
        el_lon = el.get("lon") or (el.get("center", {}).get("lon") if el.get("center") else None)
        if el_lat is None or el_lon is None:
            continue

        acc_type = "hotel" if tags.get("tourism") == "hotel" else "airbnb"
        stars = parse_stars(tags) if acc_type == "hotel" else 0
        distance_km = haversine_km(lat, lon, el_lat, el_lon)
        if distance_km > radius_km:
            continue

        name = tags.get("name") or tags.get("addr:street") or tags.get("operator") or "Hébergement sans nom"
        price = estimate_price(stars, acc_type, distance_km)

        address = " ".join(filter(None, [
            tags.get("addr:housenumber", ""),
            tags.get("addr:street", ""),
            tags.get("addr:city", ""),
        ])).strip()
        if not address:
            address = "Adresse non disponible"

        url = f"https://www.openstreetmap.org/?mlat={el_lat}&mlon={el_lon}#map=17/{el_lat}/{el_lon}"
        source = "OpenStreetMap (données réelles)" if acc_type == "hotel" else "OpenStreetMap (appartements/meublés)"

        results.append(
            AccommodationOption(
                name=name,
                type=acc_type,
                stars=stars,
                price_per_night=price,
                distance_km=round(distance_km, 2),
                lat=el_lat,
                lon=el_lon,
                address=address,
                source=source,
                url=url,
            )
        )

    return results

# ----------------------------------------------------------------------
# Scoring et combinaisons
# ----------------------------------------------------------------------
def _normalize(value, min_v, max_v):
    if max_v > min_v:
        return (value - min_v) / (max_v - min_v)
    return 0.0

def transport_score(t: TransportOption, all_transports: List[TransportOption]) -> float:
    prices = [x.price_eur for x in all_transports]
    durations = [x.duration_min for x in all_transports]
    co2s = [x.co2_kg for x in all_transports]
    p = _normalize(t.price_eur, min(prices), max(prices))
    d = _normalize(t.duration_min, min(durations), max(durations))
    c = _normalize(t.co2_kg, min(co2s), max(co2s))
    return 100 * (1 - 0.5 * p - 0.3 * d - 0.2 * c)

def accommodation_score(a: AccommodationOption, all_accommodations: List[AccommodationOption], radius_km: int) -> float:
    prices = [x.price_per_night for x in all_accommodations]
    p = _normalize(a.price_per_night, min(prices), max(prices))
    if a.type == "hotel":
        comfort = a.stars / 5.0
    else:
        comfort = 0.65
    proximity = max(0.0, 1 - a.distance_km / radius_km)
    return 100 * (0.35 * comfort + 0.30 * proximity + 0.35 * (1 - p))

def build_trip_options(
    transports: List[TransportOption],
    accommodations: List[AccommodationOption],
    budget: float,
    tolerance: float,
    nights: int,
    radius_km: int,
) -> List[TripOption]:
    """Construit et classe les combinaisons respectant budget + tolérance."""
    trips = []
    for t in transports:
        t_score = transport_score(t, transports)
        for a in accommodations:
            total_price = t.price_eur + a.price_per_night * nights
            if total_price > budget + tolerance + 1e-6:
                continue
            a_score = accommodation_score(a, accommodations, radius_km)
            budget_score = 100 * max(0.0, 1 - abs(total_price - budget) / max(budget, 1))
            score = 0.3 * t_score + 0.3 * a_score + 0.4 * budget_score

            reasons = []
            if t.price_eur == min(x.price_eur for x in transports):
                reasons.append("Transport le plus économique")
            if t.duration_min == min(x.duration_min for x in transports):
                reasons.append("Trajet le plus rapide")
            if t.co2_kg == min(x.co2_kg for x in transports):
                reasons.append("Empreinte carbone la plus faible")
            if a.type == "hotel" and a.stars >= 4:
                reasons.append(f"Hôtel {a.stars}* confortable")
            if a.distance_km < 2:
                reasons.append("Hébergement à moins de 2 km du centre")
            if total_price <= budget:
                reasons.append("Prix total dans le budget")
            else:
                reasons.append("Prix total dans la marge de tolérance")
            if not reasons:
                reasons.append("Bon équilibre qualité/prix")

            trips.append(
                TripOption(
                    transport=t,
                    accommodation=a,
                    total_price=round(total_price, 2),
                    total_duration_min=round(t.duration_min, 1),
                    score=round(score, 1),
                    reasons=reasons,
                )
            )
    trips.sort(key=lambda x: x.score, reverse=True)
    return trips

# ----------------------------------------------------------------------
# Interface utilisateur Streamlit
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="TripPlanner – Meilleur plan de voyage",
    page_icon="✈️",
    layout="wide",
)

def display_trip_card(trip, nights, origin, destination):
    """Affiche une carte détaillée pour une combinaison."""
    t = trip.transport
    a = trip.accommodation

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mode de transport", f"{t.mode} – {t.provider}")
        st.write(f"**Durée :** {t.duration_min:.0f} min")
        st.write(f"**Prix :** {t.price_eur:.2f} €")
        st.write(f"**CO2 :** {t.co2_kg:.1f} kg")
    with col2:
        st.metric("Hébergement", a.name)
        stars = f"{a.stars} étoiles" if a.stars else "Non classé"
        st.write(f"**Type :** {a.type.capitalize()} | {stars}")
        st.write(f"**Prix/nuit :** {a.price_per_night:.2f} €")
        st.write(f"**Distance :** {a.distance_km:.1f} km")
        if a.url:
            st.markdown(f"[Voir sur OpenStreetMap]({a.url})")
    with col3:
        st.metric("Prix total", f"{trip.total_price:.2f} €")
        st.metric("Durée trajet", f"{trip.total_duration_min:.0f} min")
        st.metric("Score", f"{trip.score:.0f}/100")
    st.write("**Points forts :** " + ", ".join(trip.reasons))

st.title("✈️ TripPlanner – Meilleure combinaison transport + hébergement")
st.caption(
    "Recherche en temps réel sur Internet : géocodage (Nominatim/Photon), itinéraires OSRM, "
    "hébergements OpenStreetMap. Tarifs transport/hébergement **estimés** par modèles."
)

with st.form("search_form"):
    col1, col2 = st.columns(2)
    with col1:
        origin_input = st.text_input("Ville de départ *", value="Paris")
        date_depart = st.date_input("Date de départ", value=date.today() + timedelta(days=30))
    with col2:
        destination_input = st.text_input("Destination *", value="Lyon")
        nights = st.number_input("Durée du séjour (nuits) *", min_value=1, max_value=60, value=2, step=1)

    col3, col4 = st.columns(2)
    with col3:
        budget = st.number_input(
            "Budget cible total (€, par personne, transport + hébergement) *",
            min_value=10.0,
            max_value=10000.0,
            value=300.0,
            step=10.0,
        )
        tolerance = st.slider("Marge de tolérance (€)", min_value=0, max_value=500, value=50, step=10)
    with col4:
        radius = st.slider("Rayon de recherche hébergement (km)", min_value=1, max_value=50, value=10)
        acc_type = st.selectbox("Type d'hébergement", ["Hôtel", "Airbnb", "Les deux"])
        star_filter = st.selectbox(
            "Étoiles minimum (hôtels uniquement)",
            ["Toutes", 1, 2, 3, 4, 5],
            disabled=(acc_type == "Airbnb"),
        )

    roundtrip = st.checkbox("Inclure le trajet aller-retour", value=True)
    submitted = st.form_submit_button("🔍 Rechercher les meilleures options")

if submitted:
    if not origin_input.strip() or not destination_input.strip():
        st.error("Veuillez remplir les champs de départ et de destination.")
    else:
        try:
            with st.spinner("Géocodage des villes..."):
                origin = geocode(origin_input.strip())
                destination = geocode(destination_input.strip())

            with st.spinner("Calcul des itinéraires et collecte des hébergements..."):
                transport_options = generate_transport_options(origin, destination, roundtrip)

                acc_types = []
                if acc_type in ["Hôtel", "Les deux"]:
                    acc_types.append("hotel")
                if acc_type in ["Airbnb", "Les deux"]:
                    acc_types.append("airbnb")

                accommodations = fetch_osm_accommodations(
                    destination.lat,
                    destination.lon,
                    radius,
                    tuple(acc_types),
                )

                if acc_type != "Airbnb" and star_filter != "Toutes":
                    accommodations = [
                        a for a in accommodations
                        if a.type != "hotel" or a.stars >= int(star_filter)
                    ]

                if not accommodations:
                    st.warning(
                        "Aucun hébergement réel trouvé avec ces critères. "
                        "Élargissez le rayon ou modifiez le filtre d'étoiles."
                    )
                else:
                    if not transport_options:
                        st.error("Aucune option de transport estimée. Vérifiez la connexion Internet.")
                    else:
                        trips = build_trip_options(
                            transport_options,
                            accommodations,
                            budget,
                            tolerance,
                            nights,
                            radius,
                        )

                        if not trips:
                            st.error(
                                "Aucun résultat ne correspond à votre budget, même avec la marge de tolérance. "
                                "Essayez d'augmenter le budget ou la tolérance, ou de réduire le rayon/les étoiles."
                            )
                            # Suggestion de l'option la moins chère
                            all_trips = build_trip_options(
                                transport_options,
                                accommodations,
                                budget=10**9,
                                tolerance=0,
                                nights=nights,
                                radius_km=radius,
                            )
                            if all_trips:
                                cheapest = all_trips[0]
                                st.info(
                                    f"L'option la moins chère actuellement trouvée coûte "
                                    f"**{cheapest.total_price:.2f} €**. Ajustez votre budget pour la voir."
                                )
                        else:
                            st.success(f"{len(trips)} combinaison(s) trouvée(s) dans votre budget élargi.")

                            best = trips[0]
                            st.subheader("🏆 Meilleure recommandation")
                            display_trip_card(best, nights, origin, destination)

                            st.subheader("📊 Comparaison des options")
                            data = []
                            for i, trip in enumerate(trips[:15], 1):
                                data.append({
                                    "Option": f"#{i} {trip.transport.mode} + {trip.accommodation.name}",
                                    "Prix total (€)": trip.total_price,
                                    "Durée trajet (min)": trip.total_duration_min,
                                    "Transport (€)": trip.transport.price_eur,
                                    "Hébergement/nuit (€)": trip.accommodation.price_per_night,
                                    "Score": trip.score,
                                    "Type hébergement": trip.accommodation.type,
                                    "Distance (km)": trip.accommodation.distance_km,
                                })
                            df = pd.DataFrame(data).set_index("Option")
                            st.dataframe(df, use_container_width=True)

                            chart_df = df[["Prix total (€)"]]
                            st.bar_chart(chart_df)

                            st.subheader("📍 Localisation des hébergements")
                            map_data = pd.DataFrame([
                                {
                                    "lat": trip.accommodation.lat,
                                    "lon": trip.accommodation.lon,
                                }
                                for trip in trips[:20]
                            ])
                            st.map(map_data, zoom=11)

        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")
            st.info(
                "Vérifiez votre connexion Internet et que les noms de villes sont corrects. "
                "Si le problème persiste, les services gratuits utilisés peuvent être momentanément indisponibles."
            )

st.sidebar.header("ℹ️ À propos des données")
st.sidebar.markdown(
    """
- **Géocodage** : Nominatim (OpenStreetMap) avec repli Photon
- **Itinéraires** : OSRM (serveur public)
- **Hébergements** : Overpass API (OpenStreetMap, miroirs multiples)
- **Tarifs transport** : estimations basées sur des barèmes moyens
- **Tarifs hébergement** : estimations selon type/étoiles/proximité

Pour des prix et horaires réels, connectez des API partenaires :
SNCF, FlixBus, Amadeus, Booking, Airbnb.
"""
)
