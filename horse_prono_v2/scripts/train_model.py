from __future__ import annotations

import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app_core.db import activate_model, get_latest_active_model, get_supabase_client, get_training_history, save_backtest_run, save_model_version
from app_core.model import HorseRacingModel
from app_core.model_store import build_model_record


def main() -> None:
    history = get_training_history()
    if history.empty:
        raise SystemExit("Aucun historique terminé dans Supabase.")

    model = HorseRacingModel()
    metrics = model.fit(history)
    record = build_model_record(model, metrics)

    previous = get_latest_active_model()
    if previous and previous.get("artifact_hash") == record["artifact_hash"]:
        print("Modèle inchangé : pas de nouvelle version.")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return

    created = save_model_version(record)
    model_id = int(created["id"])
    activate_model(model_id)
    save_backtest_run({
        "model_version_id": model_id,
        "period_start": metrics.get("test_start"),
        "period_end": metrics.get("test_end"),
        "races": metrics.get("backtest_races", metrics.get("test_races", 0)),
        "metrics": metrics,
    })
    print(f"Modèle actif : {model_id}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
