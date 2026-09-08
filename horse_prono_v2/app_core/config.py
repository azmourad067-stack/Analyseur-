from dataclasses import dataclass

PMU_BASE_URL = "https://online.turfinfo.api.pmu.fr/rest/client/1"
OPEN_PMU_API_URL = "https://open-pmu-api.vercel.app/api/arrivees"
REQUEST_TIMEOUT = 15
CACHE_TTL_SECONDS = 600

REQUIRED_HISTORY_COLUMNS = [
    "race_id", "race_date", "discipline", "hippodrome", "distance", "terrain",
    "field_size", "horse_number", "horse_name", "jockey", "trainer", "odds",
    "draw", "weight", "recent_form", "career_runs", "career_wins",
    "career_places", "finish_position"
]

FEATURE_COLUMNS = [
    "log_odds", "odds_implied", "draw_norm", "weight_norm", "recent_form_score",
    "career_win_rate", "career_place_rate", "field_size_norm", "distance_norm"
]

@dataclass(frozen=True)
class ModelConfig:
    min_train_rows: int = 80
    random_state: int = 42
    c: float = 1.0
    place_cutoff: int = 3
