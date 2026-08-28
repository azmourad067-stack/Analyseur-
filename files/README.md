# 🐎 Outil de pronostic hippique (galop plat)

Application Streamlit d'aide à la décision pour les courses de galop plat. L'utilisateur saisit 6 paramètres par cheval ; l'outil calcule deux scores complémentaires.

## Fonctionnement

Pour chaque cheval du peloton, l'utilisateur renseigne :

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
