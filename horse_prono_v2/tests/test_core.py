import pandas as pd

from app_core.data import example_race, normalize_history
from app_core.model import HorseRacingModel


def test_demo_prediction_sums_to_one():
    race = example_race()
    model = HorseRacingModel()
    result = model.predict(race)
    assert len(result) == len(race)
    assert abs(float(result["win_probability"].sum()) - 1.0) < 1e-8
    assert result["rank"].tolist() == list(range(1, len(result) + 1))


def test_history_normalization():
    df = pd.DataFrame({
        "race_id": ["R1"], "race_date": ["2026-01-01"], "recent_form": ["2 1 4 5"],
        "career_runs": [10], "career_wins": [2], "career_places": [5], "finish_position": [2],
    })
    out = normalize_history(df)
    assert out.loc[0, "recent_form_score"] > 0
    assert out.loc[0, "career_win_rate"] == 0.2
