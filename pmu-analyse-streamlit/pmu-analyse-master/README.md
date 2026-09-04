# 🐎 PMU Analyse — Streamlit

Outil d'aide à la décision pour les paris hippiques : collecte des programmes PMU
(réunions, courses, participants) via l'API publique **turfinfo**, stockage en
**SQLite** et visualisation / analyse dans une interface **Streamlit**.

## Fonctionnalités

- **📥 Collecte des données** : récupère les programmes PMU entre deux dates
  (réunions, courses et participants) avec barre de progression.
- **🗂️ Explorer la base** : consultation des tables réunions, courses, participants.
- **📊 Analyses** : statistiques sur les participants (gains de carrière, victoires,
  top entraîneurs, top chevaux, répartition par discipline).

## Structure du projet

```
pmu-analyse-master/
├── streamlit_app.py          # Application Streamlit (point d'entrée)
├── requirements.txt
├── .streamlit/config.toml    # Thème & configuration
├── main.py                   # Ancien point d'entrée CLI (collecte simple)
├── scrapping/
│   └── scrapping.py          # Appels API PMU turfinfo
├── data_traitement/
│   └── traitement.py         # Prétraitement avant insertion
├── database/
│   ├── setup_database.py     # Modèles SQLAlchemy + moteur SQLite
│   ├── database.py           # Fonctions d'enregistrement
│   └── db/pmu_data.db        # Base SQLite (données d'exemple incluses)
├── logger/
│   └── logging_config.ini
└── example_response/         # Exemples de réponses API
```

## Démarrage en local

```bash
python -m venv .venv
source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Déploiement sur Streamlit Cloud (via GitHub)

1. Créez un dépôt sur [GitHub](https://github.com) (ex. `pmu-analyse`).
2. Poussez le contenu de ce dossier :
   ```bash
   git init
   git add .
   git commit -m "PMU Analyse Streamlit"
   git branch -M main
   git remote add origin https://github.com/<votre-compte>/pmu-analyse.git
   git push -u origin main
   ```
3. Rendez-vous sur [share.streamlit.io](https://share.streamlit.io) et connectez-vous
   avec votre compte GitHub.
4. Cliquez sur **New app** → sélectionnez le dépôt `pmu-analyse`, branche `main`,
   fichier principal `streamlit_app.py`.
5. Cliquez sur **Deploy**. Votre application est en ligne en quelques minutes.

> Le fichier `database/db/pmu_data.db` est ignoré par git (`.gitignore`) : en
> production, la base se remplit automatiquement via la page **Collecte des données**.

## Remarques

- L'API utilisée est publique et sans authentification ; un `user-agent` standard est envoyé.
- Les données sont fournies à titre informatif et ne constituent pas un conseil de jeu.
