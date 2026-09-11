from dataclasses import dataclass


PMU_BASE_URL = (
    "https://online.turfinfo.api.pmu.fr/rest/client/1"
)

REQUEST_TIMEOUT = 20

CACHE_TTL_SECONDS = 600


HISTORY_COLUMNS = [
    "race_id",
    "race_date",
    "discipline",
    "hippodrome",
    "distance",
    "terrain",
    "field_size",
    "horse_number",
    "horse_name",
    "jockey",
    "trainer",
    "odds",
    "draw",
    "weight",
    "age",
    "sex",
    "recent_form",
    "finish_position",
]


# ============================================================
# HORSEPRONO V4.1 FEATURES
# ============================================================

FEATURE_COLUMNS = [

    # Marché
    "log_odds",
    "market_prob",

    # Course / participant
    "draw_norm",
    "draw_low_advantage",
    "weight_rel",
    "distance_norm",
    "field_size_norm",

    # Historique temporel
    "horse_win_rate",
    "horse_place_rate",

    "jockey_win_rate",
    "jockey_place_rate",

    "trainer_win_rate",
    "trainer_place_rate",

    "recent_form_score",
    "recent_consistency",

    "horse_starts_prior",
    "jockey_starts_prior",
    "trainer_starts_prior",

    # --------------------------------------------------------
    # AGE / SEXE V4.1
    # --------------------------------------------------------

    "age_norm",
    "age_relative",
    "age_missing",

    # Mâle = catégorie de référence.
    "sex_female",
    "sex_gelding",
    "sex_known",

    # --------------------------------------------------------
    # Discipline
    # --------------------------------------------------------

    "discipline_plat",
    "discipline_autostart",
    "discipline_volte",
    "discipline_monte",
    "discipline_obstacle",

    # --------------------------------------------------------
    # PLAT
    # --------------------------------------------------------

    "plat_weight_rel",
    "plat_draw_norm",
    "plat_distance_norm",
    "plat_age_norm",

    # --------------------------------------------------------
    # AUTOSTART
    # --------------------------------------------------------

    "autostart_draw_norm",
    "autostart_distance_norm",
    "autostart_field_size_norm",
    "autostart_age_norm",

    # --------------------------------------------------------
    # VOLTE
    # --------------------------------------------------------

    "volte_distance_norm",
    "volte_field_size_norm",
    "volte_age_norm",

    # --------------------------------------------------------
    # MONTE
    # --------------------------------------------------------

    "monte_weight_rel",
    "monte_distance_norm",
    "monte_age_norm",

    # --------------------------------------------------------
    # OBSTACLE
    # --------------------------------------------------------

    "obstacle_weight_rel",
    "obstacle_distance_norm",
    "obstacle_age_norm",
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
