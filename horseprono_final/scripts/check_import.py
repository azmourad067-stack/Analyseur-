import os
from supabase import create_client
u=os.getenv('SUPABASE_URL'); k=os.getenv('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_KEY')
if not u or not k: raise SystemExit('SUPABASE_URL et SUPABASE_SERVICE_KEY/SUPABASE_KEY requis')
sb=create_client(u,k)
for table in ['races','participants','ingestion_runs','model_versions','backtest_runs','predictions']:
    try:
        rows=sb.table(table).select('*',count='exact').limit(1).execute()
        print(f'{table}: {rows.count if rows.count is not None else "?"}')
    except Exception as e: print(f'{table}: ERREUR {e}')
