# 🧭 Travel Plan Finder

Moteur de recherche qui compare et recommande les meilleures **combinaisons
transport (bus, train, avion) + hébergement (hôtel, Airbnb)** selon un départ,
une destination, **une date de départ** et un budget — sur **données réelles
récupérées sur internet**, sans aucune donnée simulée.

## 🔌 Sources de données réelles (gratuites, sans clé API)

| Donnée | Source | Détail |
|---|---|---|
| Géocodage villes | **Photon** (secours : Nominatim) | N'importe quelle ville du monde → coordonnées GPS réelles (OpenStreetMap) |
| Trains | **db-vendo** (v6.db.transport.rest) | Horaires et prix réels de l'API Deutsche Bahn, couvre la France et l'Europe (TGV, ICE, Thalys…) |
| Bus / route | **OSRM** (router.project-osrm.org) | Distance et durée réelles du réseau routier OpenStreetMap |
| Hôtels | **Overpass API** (overpass-api.de) | Hôtels réels : nom, étoiles déclarées, position GPS, site web |
| Vols | **Google Flights** | Pas d'API vols gratuite → lien de recherche réel pré-rempli (villes + date) |
| Airbnb | **airbnb.fr** | Pas d'API publique → lien de recherche réel pré-rempli (destination + dates + voyageurs) |

**Honnêteté produit** : ce qu'aucune API publique gratuite ne publie (tarifs
des vols, prix des chambres d'hôtel, annonces Airbnb) n'est **jamais inventé** :
soit un lien de réservation réel est fourni, soit la valeur est explicitement
marquée **« indicatif »** dans l'interface (calculée sur la distance réelle
OSRM pour le bus, sur les étoiles réelles pour les hôtels).

## Fonctionnalités

- Recherche par ville de départ / destination (géocodage réel, aucune liste figée)
- **Date de départ** : pilote les horaires de trains réels et les liens de réservation
- Budget avec **tolérance ±**, **rayon de recherche** (km, filtrage haversine sur
  positions GPS réelles), type d'hébergement Hôtel/Airbnb/les deux,
  **filtre étoiles** actif uniquement pour les hôtels
- **Score qualité/prix 0-100** : prix 45 %, temps porte-à-porte 25 %,
  confort 15 %, CO₂ 15 % (repondération élégante quand un critère est indisponible)
- Tableau comparatif + cartes détaillées avec badges (🏆 💰 ⚡ 🌱) et
  **boutons de réservation réels** (train, hôtel, vols, Airbnb)
- Panneau de **provenance des données** consultable dans l'interface
- Gestion d'erreurs claire (champs vides, ville introuvable, date passée,
  budget trop serré avec suggestion du prix minimum trouvé)

## Architecture

```
app.py                             Point d'entrée Streamlit
travel_planner/
├── models.py                      Dataclasses (offres, plans, résultat)
├── geo.py                         Géocodage réel Photon/Nominatim + haversine
├── engine.py                      Logique métier : recherche/filtrage/classement
├── scoring.py                     Score qualité/prix + badges
├── ui.py                          Interface Streamlit (aucune logique métier)
└── providers/
    ├── base.py                    Interfaces (TransportProvider, AccommodationProvider)
    ├── transport.py               db-vendo (trains) + OSRM (route) + Google Flights
    └── accommodation.py           Overpass (hôtels) + lien Airbnb
```

La logique métier ne dépend pas de Streamlit : elle est testable et réutilisable.

## Déploiement sur Streamlit Community Cloud

1. **Pousser sur GitHub** :
   ```bash
   git init && git add . && git commit -m "Travel Plan Finder"
   git remote add origin https://github.com/<vous>/<repo>.git
   git push -u origin main
   ```
2. Sur [share.streamlit.io](https://share.streamlit.io) : **New app** →
   dépôt, branche `main`, fichier principal `app.py` → **Deploy**.
   Aucun secret requis : toutes les API utilisées sont sans clé.

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Passer à des tarifs réservables en temps réel (optionnel)

Pour des prix de vols et d'hôtels réservables (au lieu des liens/indicatifs) :
créer une classe implémentant `TransportProvider` / `AccommodationProvider`
appelant une API partenaire payante (Booking Affiliate, LiteAPI, Kiwi Tequila…),
stocker les clés dans **Settings → Secrets** de l'app Streamlit
(`st.secrets["MA_CLE"]`), jamais dans Git. Aucun changement côté UI.
