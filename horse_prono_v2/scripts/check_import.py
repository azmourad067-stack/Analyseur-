from __future__ import annotations

import os
from supabase import create_client

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
if not url or not key:
    raise SystemExit("SUPABASE_URL / SUPABASE_SERVICE_KEY manquants")

c = create_client(url, key)
for table in ("races", "participants", "ingestion_runs", "model_versions"):
    r = c.table(table).select("*", count="exact").limit(1).execute()
    print(f"{table}: {r.count or 0}")
