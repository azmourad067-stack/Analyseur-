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
    "recent_form",
    "finish_position",
]


# ============================================================
# HORSEPRONO V4 FEATURES
# ============================================================
#
# Toutes les variables doivent être connues avant le départ.
#
# Les nouvelles features V4 permettent au modèle de ne plus
# traiter de la même façon :
#
# - le Plat
# - l'Attelé Autostart
# - l'Attelé Volte
# - le Trot Monté
# - l'Obstacle
#
# Les interactions sont particulièrement importantes :
# par exemple le poids n'a pas le même sens en Plat qu'en
# Attelé, et la corde est particulièrement intéressante
# en Autostart.
# ============================================================

FEATURE_COLUMNS = [

    # --------------------------------------------------------
    # Marché
    # --------------------------------------------------------

    "log_odds",
    "market_prob",

    # --------------------------------------------------------
    # Course / participant
    # --------------------------------------------------------

    "draw_norm",
    "draw_low_advantage",
    "weight_rel",
    "distance_norm",
    "field_size_norm",

    # --------------------------------------------------------
    # Historique temporel
    # --------------------------------------------------------

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
    # Discipline V4
    # --------------------------------------------------------

    "discipline_plat",
    "discipline_autostart",
    "discipline_volte",
    "discipline_monte",
    "discipline_obstacle",

    # --------------------------------------------------------
    # Interactions PLAT
    # --------------------------------------------------------

    "plat_weight_rel",
    "plat_draw_norm",
    "plat_distance_norm",

    # --------------------------------------------------------
    # Interactions AUTOSTART
    # --------------------------------------------------------

    "autostart_draw_norm",
    "autostart_distance_norm",
    "autostart_field_size_norm",

    # --------------------------------------------------------
    # Interactions VOLTE
    # --------------------------------------------------------

    "volte_distance_norm",
    "volte_field_size_norm",

    # --------------------------------------------------------
    # Interactions TROT MONTE
    # --------------------------------------------------------

    "monte_weight_rel",
    "monte_distance_norm",

    # --------------------------------------------------------
    # Interactions OBSTACLE
    # --------------------------------------------------------

    "obstacle_weight_rel",
    "obstacle_distance_norm",
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
