# Import historique HorseProno V3

## 1. GitHub Actions

Créer dans le dépôt GitHub les secrets :

- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`

Puis ouvrir **Actions → HorseProno import historique → Run workflow** et choisir les dates.

Par défaut : 2026-01-01 → 2026-06-30.

## 2. Import local

```bash
python scripts/import_history.py api --start 2026-01-01 --end 2026-06-30 --delay 1
```

Ou CSV :

```bash
python scripts/import_history.py csv data/template_historique.csv
```

## 3. Important

L'API historique `open-pmu-api.vercel.app` est une API communautaire/non documentée : elle peut changer de format ou être indisponible. Le script journalise les jours en erreur et peut être relancé sans recréer les mêmes couples course/numéro de cheval.

La clé `SUPABASE_SERVICE_KEY` ne doit jamais être placée dans Streamlit Cloud ni commitée dans GitHub. Streamlit utilise uniquement `SUPABASE_URL` + `SUPABASE_KEY` côté lecture.
