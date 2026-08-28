# 🐎 Outil de pronostic hippique (galop plat)

Application Streamlit d'aide à la décision pour les courses de galop plat. L'utilisateur saisit 6 paramètres par cheval ; l'outil calcule deux scores complémentaires.

## Auto-remplissage optionnel (PMU.fr)

L'application peut tenter de pré-remplir le tableau des partants (nom, poids, cote, forme estimée à partir de la musique) en interrogeant l'API interne de PMU.fr, via le module `pmu_connector.py`.

⚠️ **Cette API n'est pas officielle ni documentée par PMU.** Elle peut changer ou devenir indisponible sans préavis, et son usage automatisé se situe dans une zone grise juridique (droit des bases de données). En conséquence :

- L'application ne dépend jamais de cette API pour fonctionner : en cas d'échec, un message d'erreur s'affiche et le tableau reste éditable manuellement.
- Réserver cette fonctionnalité à un usage personnel et ponctuel, avec des appels espacés — pas à un usage commercial ni à des requêtes en boucle.
- L'aptitude distance/terrain et le niveau jockey/entraîneur restent toujours à évaluer manuellement : l'API ne fournit pas ces jugements qualitatifs.
- Si l'API change de structure un jour, seul `pmu_connector.py` doit être adapté — le reste de l'application n'est pas affecté.

## Analyse de l'historique des courses (optionnel, à exécuter en local)

Deux scripts permettent de construire un jeu de données historique et d'en tirer des statistiques calibrées, en s'appuyant sur le projet open-pmu-api (API REST publique donnant les arrivées passées).

⚠️ **Ces scripts doivent être exécutés sur ta machine**, pas dans un environnement d'exécution restreint : ils font des centaines d'appels HTTP vers `open-pmu-api.vercel.app`, un domaine qui n'est pas forcément accessible depuis tous les environnements.

1. **`collecte_historique.py`** — parcourt jour par jour une période donnée (par défaut : janvier 2025 à juillet 2026) et construit `historique_courses.csv` (une ligne par cheval par course). Un délai est appliqué entre chaque appel pour rester courtois envers ce service gratuit et non officiel.
   ```bash
   pip install -r requirements-analyse.txt
   python collecte_historique.py
   ```
2. **`calibrer_poids.py`** — charge ce CSV et calcule :
   - le taux de podium (top 3) par jockey et par entraîneur sur la période collectée ;
   - une régression logistique estimant l'importance relative des critères disponibles dans cet historique (forme via la musique, corde, gains, âge, taux jockey, taux entraîneur) pour prédire une entrée dans le top 3.
   ```bash
   python calibrer_poids.py
   ```
   Résultat : `resultats_calibration.json`.

**Limite importante à connaître** : cet historique ne contient ni le poids porté, ni l'aptitude terrain/distance — deux des 5 critères du score intrinsèque de `scoring.py`. Ces deux-là ne peuvent pas être calibrés par cette source de données et resteront à évaluer manuellement, quelle que soit la quantité de données collectées.

Une fois `resultats_calibration.json` généré avec de vraies données, il peut servir à enrichir l'application (par exemple, remplacer l'évaluation manuelle "Fort/Moyen/Faible" du jockey/entraîneur par son taux de podium réel) — cette intégration n'a pas été câblée automatiquement dans `scoring.py`, faute de données réelles disponibles au moment de la conception de ces scripts.

## Fonctionnement

Pour chaque cheval du peloton, l'utilisateur renseigne (manuellement ou via l'auto-remplissage) :

1. **Forme récente** — nombre de podiums sur les 5 dernières courses
2. **Poids porté** (kg)
3. **Aptitude à la distance du jour** (Favorable / Neutre / Défavorable)
4. **Aptitude au terrain du jour** (Favorable / Neutre / Défavorable)
5. **Niveau jockey/entraîneur** (Fort / Moyen / Faible)
6. **Cote probable**

L'outil calcule ensuite :

- **Le score intrinsèque** : une estimation de la valeur du cheval basée uniquement sur les 5 premiers critères, indépendamment de la cote. Chaque critère est normalisé *relativement au peloton du jour* (les chevaux sont comparés entre eux, pas sur une échelle absolue).
- **Le score de valeur** : la comparaison entre la probabilité de performance déduite du score intrinsèque et la probabilité implicite de la cote (corrigée du surround). Un écart positif signale un cheval potentiellement sous-coté par le marché ; un écart négatif signale l'inverse.

Les poids attribués aux 5 critères du score intrinsèque sont réglables dans la barre latérale (valeurs par défaut : Forme 30%, Distance 20%, Terrain 20%, Poids 15%, Jockey/Entraîneur 15%).

## Structure du projet

```
app.py              → interface Streamlit
scoring.py           → logique de calcul (score intrinsèque + score de valeur)
models.py             → constantes et structure des données
data_logger.py        → préparation de l'export CSV
requirements.txt
```

## Installation locale

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Déploiement sur Streamlit Community Cloud

1. Pousser ce dossier sur un repo GitHub.
2. Se connecter sur [share.streamlit.io](https://share.streamlit.io) avec le compte GitHub.
3. Sélectionner le repo et indiquer `app.py` comme fichier principal.
4. Déploiement automatique à chaque `git push` sur la branche principale.

## Suivi et amélioration future

Chaque analyse peut être téléchargée en CSV (bouton en bas de page), avec une colonne "Résultat réel" à compléter manuellement après la course. En archivant ces fichiers au fil des courses, il devient possible, à terme, de recalibrer les poids du score intrinsèque de façon empirique plutôt qu'intuitive (ex. via une régression logistique sur les données accumulées), une fois un volume suffisant de courses réunies.

## Limites

Cet outil produit une estimation basée sur des paramètres saisis manuellement et des poids réglés de façon heuristique, pas un modèle validé statistiquement sur données historiques. Il est conçu comme une aide à la réflexion, pas comme une prédiction garantie.
