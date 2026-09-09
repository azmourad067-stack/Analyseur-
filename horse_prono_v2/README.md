# 🏇 HorseProno V3 — architecture data + modèle quantitatif

HorseProno V3 transforme l'application en **plateforme de données et de modélisation** : les courses et résultats persistent dans PostgreSQL/Supabase, les features sont reconstruites **strictement avant chaque course**, les modèles sont évalués dans le temps, puis une version du modèle est publiée pour Streamlit.

```text
                     ┌──────────────────┐
                     │ PMU / CSV / APIs │
                     └────────┬─────────┘
                              │
                              ▼
                        scripts/etl_*
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Supabase / PostgreSQL   │
                 │ races                   │
                 │ participants            │
                 │ market_snapshots        │
                 │ ingestion_runs          │
                 │ model_versions          │
                 │ backtest_runs           │
                 │ predictions             │
                 └────────────┬────────────┘
                              │
                              ▼
                     features temporelles
                              │
                   ┌──────────┼───────────┐
                   ▼          ▼           ▼
             Bradley-Terry  Logistique  Gradient Boosting
                   │          │           │
                   └──────────┴───────────┘
                              │
                       calibration
                              │
                       backtest temporel
                              │
                       modèle actif JSON
                              │
                              ▼
                         Streamlit Cloud
```

## Ce que V3 améliore

### 1. Features sans fuite temporelle

Les statistiques cheval/jockey/entraîneur sont calculées à partir des résultats **antérieurs à la date de la course**. Pour chaque entité, le calcul utilise un `shift` logique (résultat courant exclu), ce qui évite un biais classique : connaître le résultat d'une course au moment où l'on construit les variables censées le prédire.

La feature de forme récente est une moyenne exponentiellement pondérée des cinq dernières performances connues.

### 2. Bradley-Terry

Le problème hippique est d'abord un problème de **classement intra-course**. Bradley-Terry transforme chaque course historique en comparaisons deux à deux :

```text
P(A bat B) = sigmoid(strength(A) - strength(B))
```

Les probabilités de victoire sont ensuite obtenues par une softmax des forces latentes dans chaque course.

### 3. Régression logistique

Une seconde composante modélise directement `P(victoire)` et `P(placé)`. Elle reste interprétable et régularisée, donc adaptée à des données bruitées.

### 4. Gradient Boosting = challenger offline

Un `HistGradientBoostingClassifier` est entraîné avec la même séparation temporelle. Il sert de **challenger non linéaire** et entre dans l'évaluation du modèle V3 hors ligne.

Le modèle publié dans Supabase est volontairement un artefact JSON portable : il contient Bradley-Terry + logistique + calibration, sans `pickle`. Cela simplifie la sécurité et la reproductibilité du déploiement.

### 5. Calibration

La calibration utilise un jeu chronologique séparé (Platt scaling sur le logit de la probabilité brute). Les probabilités finales sont ainsi évaluées sur une période future qui n'a pas participé à l'ajustement des paramètres de calibration.

### 6. Backtest

Le pipeline calcule notamment :

- log-loss victoire/placé ;
- Brier score ;
- ROC-AUC lorsque défini ;
- précision Top-1 ;
- couverture Top-3 ;
- probabilité moyenne du n°1 du modèle vs marché ;
- edge moyen modèle - marché ;
- ROI simulé à mise fixe de 1 unité sur le Top-1.

Le ROI est un **outil d'évaluation historique**, pas une promesse de performance future.

## 1. Installer Supabase

Créer un projet Supabase puis exécuter :

```text
 database/schema.sql
```

dans le SQL Editor.

Le schéma crée les tables, index, RLS et la fonction :

```text
get_training_history_until(p_race_date, p_limit)
```

Cette fonction permet au serveur de demander uniquement l'historique antérieur à une date cible.

## 2. Secrets

### Streamlit Community Cloud

Dans les secrets de l'application :

```toml
SUPABASE_URL = "https://xxxxx.supabase.co"
SUPABASE_KEY = "ey..."
```

Utiliser une clé publique/anon pour l'application. Ne jamais publier une `service_role` dans GitHub ni dans les secrets d'un front public.

### GitHub Actions

Ajouter les secrets :

```text
SUPABASE_URL
SUPABASE_SERVICE_KEY
```

Le workflow quotidien utilise la clé de service pour les écritures ETL et la publication des versions de modèle.

## 3. Amorcer l'historique

Le modèle a besoin d'un historique réel de courses terminées. Le format minimal :

```text
race_id,race_date,discipline,hippodrome,distance,terrain,field_size,horse_number,horse_name,jockey,trainer,odds,draw,weight,recent_form,finish_position
```

Puis :

```bash
python scripts/import_history.py historique.csv
```

Pour un modèle réellement utile, vise plusieurs dizaines de milliers de partants, idéalement sur plusieurs années et plusieurs disciplines.

## 4. Lancer l'entraînement

```bash
python scripts/train_model.py
```

Le script :

```text
historique terminé
→ tri temporel par course
→ 70 % train
→ 15 % calibration
→ 15 % test futur
→ BT + logistique + challenger GB
→ calibration
→ métriques
→ artefact JSON
→ nouvelle version Supabase
→ activation
```

Une nouvelle version n'est publiée que si son artefact est différent du modèle actif précédent.

## 5. ETL PMU

Le connecteur actuel cible une interface PMU utilisée par des projets communautaires :

```text
https://online.turfinfo.api.pmu.fr/rest/client/1
```

Exemple :

```bash
python scripts/etl_pmu.py --date 2026-09-08
```

**Limite importante :** cette interface n'est pas traitée dans le projet comme un contrat d'API publique documenté et garanti. Le code reste défensif et le CSV demeure une voie de secours. Les formats, conditions d'utilisation ou disponibilités peuvent évoluer.

Le projet ne contourne aucun contrôle d'accès et n'effectue pas de scraping agressif.

## 6. Workflow automatique

`.github/workflows/etl.yml` exécute chaque jour :

```text
PMU veille + PMU jour
        ↓
Supabase
        ↓
entraînement V3
        ↓
backtest
        ↓
modèle actif
```

Pour un historique initial volumineux, utilise d'abord `import_history.py`. Le workflow quotidien sert ensuite de maintenance incrémentale.

## 7. Streamlit Community Cloud

À la racine :

```text
app.py
requirements.txt
.streamlit/
```

Puis dans Community Cloud :

```text
Create app
→ GitHub repository
→ branch main
→ app.py
→ Advanced settings
→ Python 3.12
→ Secrets
→ Deploy
```

Community Cloud exige que `requirements.txt` soit détectable depuis la racine ou le dossier de l'entrypoint. La documentation actuelle indique également que Python 3.12 est le défaut de Community Cloud. Voir :

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## 8. Philosophie du modèle

Le marché des cotes est un benchmark, pas une vérité absolue. Le moteur expose donc côte, probabilité implicite normalisée, probabilité modèle et :

```text
EV théorique = P(modèle) × cote - 1
```

Cela permet de rechercher les divergences entre le modèle et le marché tout en affichant explicitement l'incertitude.

Une cote élevée ne signifie pas qu'un cheval est intéressant ; une probabilité modèle supérieure au marché n'implique pas non plus un avantage réel. Toute décision doit être testée sur un historique indépendant et suffisamment long.

## 9. Limites connues

- Le résultat dépend fortement de la qualité et de la profondeur de l'historique.
- Les identifiants cheval/jockey/entraîneur doivent idéalement être normalisés ; les noms textuels peuvent créer des doublons.
- Les cotes doivent correspondre à une observation réellement disponible **avant le départ** pour éviter une fuite de marché.
- Les effets spécifiques hippodrome × distance × terrain × corde sont encore perfectibles.
- Le placement dépend de la réglementation du pari et du nombre de places payées ; V3 fixe par défaut `Top 3` pour le label statistique.
- Les probabilités ne garantissent aucun gain.

## Prochaine évolution quantitative

Une V4 pertinente serait d'ajouter :

```text
Plackett-Luce / learning-to-rank
+ statistiques par hippodrome-distance-terrain
+ snapshots de cotes intraday
+ modèles spécifiques Plat / Trot / obstacle
+ calibration par discipline
+ simulation Monte Carlo des classements
+ recherche d'hyperparamètres temporelle
+ monitoring de dérive des données
+ comparaison systématique modèle / marché
```
