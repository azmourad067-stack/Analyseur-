# 🏇 HorseProno — Streamlit + probabilités

Application Streamlit modulaire pour classer les partants d'une course hippique par probabilité estimée de victoire et de placement.

## Architecture

```text
horse_prono_app/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   └── template_historique.csv
├── app_core/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   ├── pmu.py
│   └── io_utils.py
└── tests/
```

## Sources de données

### 1) PMU
Le connecteur `app_core/pmu.py` utilise l'endpoint public historiquement utilisé par des projets de collecte communautaires :
`https://online.turfinfo.api.pmu.fr/rest/client/1`

Endpoints utilisés :
- `/programme/{DDMMYYYY}`
- `/programme/{DDMMYYYY}/R{n}/C{n}/participants`

**Important :** cette interface n'est pas présentée ici comme une API officielle documentée par PMU. Le code est donc volontairement défensif et l'application prévoit un fallback CSV lorsque le schéma change ou que le service est indisponible.

### 2) Historique CSV
C'est la voie recommandée pour entraîner le modèle. Le fichier doit contenir une ligne par cheval engagé, avec `finish_position` renseigné pour les courses historiques.

Le dépôt fournit `data/template_historique.csv` comme exemple de structure, sans prétendre représenter des résultats réels.

### 3) France Galop / LeTrot
Ces plateformes proposent des statistiques riches sur les courses, jockeys/drivers et entraîneurs. Elles peuvent être utilisées pour construire un pipeline d'enrichissement séparé, en respectant leurs conditions d'accès et les droits associés aux données.

## Modèle

Deux régressions logistiques régularisées sont entraînées :
- `P(victoire) = P(finish_position = 1 | X)`
- `P(placé) = P(finish_position <= 3 | X)`

Une séparation temporelle 80/20 est utilisée lorsque les données permettent une validation suffisamment grande, afin de limiter le biais de fuite temporelle. Les métriques affichées sont log-loss et ROC-AUC sur la partie de validation lorsque cela est possible.

Les probabilités de victoire sont ensuite normalisées au niveau de la course pour obtenir une distribution qui somme à 100 %.

## Déploiement Streamlit Community Cloud

1. Crée un dépôt GitHub public ou privé.
2. Place tous les fichiers de ce projet à la racine du dépôt.
3. Va sur `https://share.streamlit.io/` puis **Create app**.
4. Choisis le repository, la branche et le fichier `app.py`.
5. Dans **Advanced settings**, choisis Python 3.12 (compatible avec l'environnement par défaut actuel de Community Cloud) et ajoute les secrets uniquement si tu branches un service externe nécessitant une clé.
6. Déploie.

Streamlit Community Cloud lit les dépendances depuis `requirements.txt`. Il est préférable d'en utiliser un seul et de pinner les versions pour rendre les déploiements reproductibles.

### Secrets
Le projet actuel n'en a pas besoin. Si une future source exige une clé :

```toml
[external_api]
api_key = "..."
```

Puis :

```python
import streamlit as st
key = st.secrets["external_api"]["api_key"]
```

Ne commit jamais `.streamlit/secrets.toml`.

## Lancer en local

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Rafraîchir / enrichir les données

- **Courses du jour** : le bouton de chargement du programme interroge le connecteur PMU. Le cache Streamlit réduit la fréquence des appels.
- **Historique** : préparer un CSV selon `data/template_historique.csv`, puis l'importer dans le bloc d'entraînement.
- **Pipeline industriel recommandé** : stocker l'historique dans SQLite/PostgreSQL/Supabase, puis entraîner automatiquement sur un job planifié et charger le modèle sérialisé dans l'application.

## Limites et bonnes pratiques

Le modèle est probabiliste. Une probabilité élevée ne signifie pas qu'un cheval va gagner. Il faut mesurer la qualité hors échantillon dans le temps, éviter les variables indisponibles avant le départ, surveiller les changements de distribution des courses et ne jamais présenter un score comme une garantie de gain.
