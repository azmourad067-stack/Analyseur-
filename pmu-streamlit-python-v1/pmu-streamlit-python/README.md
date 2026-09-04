# 🏇 PMU Elite Punter — Python / Streamlit

Migration complète du projet Node.js/Express/SQLite/TensorFlow.js vers Python/Streamlit/Supabase.

## Architecture

- `app.py` : interface Streamlit
- `src/api/pmu_api.py` : récupération PMU
- `src/core/processor.py` : transformation des données
- `src/core/intelligence.py` : moteur heuristique porté du projet original
- `src/core/bankroll.py` : Kelly Criterion
- `src/database/supabase.py` : accès PostgreSQL/Supabase
- `supabase/schema.sql` : schéma de base

## Déploiement

1. Créer le projet Supabase et exécuter `supabase/schema.sql`.
2. Ajouter `SUPABASE_URL` et `SUPABASE_KEY` dans les secrets Streamlit.
3. Déployer `app.py` sur Streamlit Community Cloud.

## Important

Le modèle TensorFlow.js original n'est pas présent dans l'archive fournie. La V1 porte donc le moteur heuristique et prépare l'architecture. Le portage exact du réseau ML sera fait après validation du pipeline de données et du dataset.
