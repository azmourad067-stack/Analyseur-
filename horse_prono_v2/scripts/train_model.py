from __future__ import annotations

import json
import os
import sys

# Allow `python scripts/train_model.py` from repository root.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app_core.db import get_latest_active_model, get_supabase_client, get_training_history, save_model_version
from app_core.model import HorseRacingModel
from app_core.model_store import build_model_record


def set_active(model_id: int) -> None:
    client = get_supabase_client()
    if client is None:
        raise RuntimeError("Supabase non configuré")
    client.table("model_versions").update({"is_active": False}).neq("id", model_id).execute()
    client.table("model_versions").update({"is_active": True}).eq("id", model_id).execute()


def main() -> None:
    history = get_training_history()
    if history.empty:
        raise SystemExit("Aucun historique terminé dans Supabase.")

    model = HorseRacingModel()
    metrics = model.fit(history)
    record = build_model_record(model.win_model, model.place_model, metrics, model.config)

    # Avoid inserting an identical artifact again.
    previous = get_latest_active_model()
    if previous and previous.get("artifact_hash") == record["artifact_hash"]:
        print("Modèle inchangé : aucun nouveau modèle enregistré.")
        print(json.dumps(metrics, indent=2))
        return

    created = save_model_version(record)
    set_active(int(created["id"]))
    print(f"Modèle actif: {created['id']}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
