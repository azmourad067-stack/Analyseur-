from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app_core.db import get_training_history
from app_core.model import HorseRacingModel

if __name__ == "__main__":
    history = get_training_history()
    model = HorseRacingModel()
    metrics = model.fit(history)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
