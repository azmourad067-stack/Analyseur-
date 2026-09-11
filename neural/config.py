from pathlib import Path

# ============================================================
# HORSEPRONO NEURAL V1
# Configuration générale
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models" / "neural"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Reproductibilité
# ------------------------------------------------------------

RANDOM_SEED = 42

# ------------------------------------------------------------
# Target principale
# ------------------------------------------------------------

TARGET_COLUMN = "target_top3"

# ------------------------------------------------------------
# Split temporel
# ------------------------------------------------------------

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# IMPORTANT :
# aucun shuffle temporel avant la séparation train/valid/test.

# ------------------------------------------------------------
# Architecture Neural V1
# ------------------------------------------------------------

HIDDEN_LAYERS = [128, 64, 32]

DROPOUT_1 = 0.25
DROPOUT_2 = 0.20

# ------------------------------------------------------------
# Entraînement
# ------------------------------------------------------------

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

BATCH_SIZE = 256
MAX_EPOCHS = 100

EARLY_STOPPING_PATIENCE = 12

# ------------------------------------------------------------
# Fichiers modèle
# ------------------------------------------------------------

MODEL_PATH = MODEL_DIR / "horseprono_neural_v1.pt"
SCALER_PATH = MODEL_DIR / "horseprono_neural_v1_scaler.joblib"
FEATURES_PATH = MODEL_DIR / "horseprono_neural_v1_features.json"

# ------------------------------------------------------------
# Colonnes interdites
# ------------------------------------------------------------

# Variables connues seulement APRÈS la course.
# Elles ne doivent jamais entrer dans X.

LEAKAGE_COLUMNS = [
    "position",
    "rang",
    "classement",
    "arrivee",
    "resultat",
    "is_winner",
    "target_win",
    "target_top3",
    "target_top5",
]
