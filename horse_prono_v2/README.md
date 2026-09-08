# 🏇 HorseProno — architecture data production

Cette version transforme l'application en plateforme de données :

```text
             ┌──────────────┐
             │ PMU / CSV    │
             └──────┬───────┘
                    ↓
              scripts/etl_*
                    ↓
        ┌────────────────────────┐
        │ Supabase / PostgreSQL  │
        │ races                  │
        │ participants           │
        │ ingestion_runs         │
        │ model_versions         │
        │ predictions            │
        └────────────┬───────────┘
                     ↓
              scripts/train_model.py
                     ↓
       validation temporelle 80/20
                     ↓
     artifact JSON (scaler + coefficients)
                     ↓
              Streamlit Cloud
```

## Pourquoi cette architecture ?

Le modèle et l'historique ne doivent plus vivre uniquement dans la mémoire d'une session Streamlit. Supabase devient le stockage persistant et permet de conserver l'historique des courses, les résultats, les versions du modèle et les prédictions.

Le modèle n'est **pas stocké en pickle** dans la base. Seuls les paramètres numériques nécessaires à la prédiction sont conservés dans `model_versions.artifact`. L'application reconstruit un classifieur numérique minimal à partir de JSON.

## 1. Créer Supabase

Crée un projet Supabase puis exécute `database/schema.sql` dans le SQL Editor. Le schéma crée les tables et les index nécessaires.

L'application Streamlit utilise une clé publique/anon et ne doit pas recevoir la clé `service_role`. Le job ETL/training, lui, utilise `SUPABASE_SERVICE_KEY` dans GitHub Actions ou dans un environnement serveur de confiance.

## 2. Secrets Streamlit

Localement, copie `.streamlit/secrets.toml.example` en `.streamlit/secrets.toml` puis renseigne :

```toml
SUPABASE_URL = "https://....supabase.co"
SUPABASE_KEY = "..."
```

**Ne commit jamais ce fichier.** Streamlit Community Cloud permet de coller ces valeurs dans les secrets de l'application. Voir la documentation officielle : https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management

## 3. Dépendances

Le projet est actuellement épinglé sur Streamlit 1.63.0 et Supabase Python 2.31.0. Les versions peuvent évoluer ; pinner les versions rend le déploiement reproductible.

## 4. Initialiser les données historiques

Le fichier attendu est une ligne par cheval et par course. Au minimum :

```text
race_id,race_date,discipline,hippodrome,distance,terrain,field_size,horse_number,horse_name,jockey,trainer,odds,draw,weight,recent_form,career_runs,career_wins,career_places,finish_position
```

Import :

```bash
python scripts/import_history.py ./mon_historique.csv
```

Pour cet import, définis `SUPABASE_URL` et `SUPABASE_SERVICE_KEY` dans l'environnement.

## 5. Ingestion PMU

Pour alimenter les programmes/partants :

```bash
python scripts/etl_pmu.py --date 2026-09-08
```

Le connecteur utilise :

```text
https://online.turfinfo.api.pmu.fr/rest/client/1
```

via `/programme/{DDMMYYYY}` et `/programme/{DDMMYYYY}/R{n}/C{n}/participants`.

**Important :** ce connecteur correspond à une interface utilisée par des projets communautaires et ne doit pas être considérée comme un contrat d'API publique officiellement garanti par PMU. Le code est défensif et l'architecture prévoit le CSV comme voie de secours.

## 6. Entraîner et publier une version

```bash
python scripts/train_model.py
```

Le script :

1. lit les courses terminées depuis Supabase ;
2. trie les observations dans le temps ;
3. réserve la partie la plus récente pour la validation ;
4. entraîne deux régressions logistiques L2 : victoire et placé ;
5. calcule log-loss et AUC quand possible ;
6. crée un artefact JSON ;
7. publie une nouvelle entrée dans `model_versions` ;
8. active cette version.

## 7. GitHub Actions

`.github/workflows/etl.yml` exécute quotidiennement :

```text
ingestion PMU → Supabase → entraînement → nouvelle version active
```

Ajoute dans les **GitHub Actions Secrets** :

```text
SUPABASE_URL
SUPABASE_SERVICE_KEY
```

La clé service-role ne doit **jamais** être mise dans le code ou dans les secrets de l'application publique Streamlit.

## 8. Déployer Streamlit Community Cloud

Le dépôt doit contenir `app.py` à la racine et `requirements.txt`. Dans Community Cloud :

```text
Create app
→ dépôt GitHub
→ branche main
→ fichier app.py
→ Advanced settings
→ secrets Supabase
→ Deploy
```

Community Cloud lit `requirements.txt` et permet de définir les secrets depuis les paramètres de l'application. Voir :

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management

## Sources de données et limites juridiques

Les sources web peuvent modifier leurs formats, conditions d'utilisation ou mécanismes d'accès. Le projet ne contourne pas de contrôle d'accès et n'utilise pas de scraping agressif. Pour enrichir les statistiques jockey/entraîneur, privilégie les interfaces/API autorisées ou des exports sous licence.

## Prochaine étape recommandée

La prochaine amélioration quantitative serait de remplacer/compléter la double logistique par un modèle de classement de course (Bradley-Terry/Plackett-Luce ou learning-to-rank), puis d'ajouter une vraie évaluation de **calibration probabiliste**, Brier score et simulation de valeur attendue. Cela donnera une base beaucoup plus solide pour comparer modèle, marché et incertitude.
