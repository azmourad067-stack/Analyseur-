from dataclasses import dataclass

PMU_BASE_URL = "https://online.turfinfo.api.pmu.fr/rest/client/1"
REQUEST_TIMEOUT = 20
CACHE_TTL_SECONDS = 600

HISTORY_COLUMNS = [
    "race_id", "race_date", "discipline", "hippodrome", "distance", "terrain",
    "field_size", "horse_number", "horse_name", "jockey", "trainer", "odds",
    "draw", "weight", "recent_form", "finish_position"
]

# All features are intended to be knowable before the race starts.
FEATURE_COLUMNS = [
    "log_odds", "market_prob", "draw_norm", "draw_low_advantage", "weight_rel",
    "distance_norm", "field_size_norm", "horse_win_rate", "horse_place_rate",
    "jockey_win_rate", "jockey_place_rate", "trainer_win_rate", "trainer_place_rate",
    "recent_form_score", "recent_consistency", "horse_starts_prior",
    "jockey_starts_prior", "trainer_starts_prior",
]

@dataclass(frozen=True)
class ModelConfig:
    min_train_rows: int = 500
    random_state: int = 42
    logit_c: float = 0.35
    bt_c: float = 0.50
    gb_max_depth: int = 3
    gb_learning_rate: float = 0.035
    gb_n_estimators: int = 180
    place_cutoff: int = 3
    train_fraction: float = 0.70
    calibration_fraction: float = 0.15
    bt_max_pairs_per_race: int = 180
