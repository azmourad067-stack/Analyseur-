# Déploiement HorseProno V3

## Streamlit Cloud

Dans **Settings → Secrets**, coller :

```toml
SUPABASE_URL = "https://gwnpqsbtkubtumrfnspd.supabase.co"
SUPABASE_KEY = "VOTRE_CLE_ANON_OU_PUBLISHABLE"
```

Ne jamais mettre `SUPABASE_SERVICE_KEY` dans Streamlit.

## GitHub Actions

Dans **Settings → Secrets and variables → Actions**, ajouter :

- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`

Le service key sert aux écritures ETL/import et ne doit rester que dans GitHub Actions.
