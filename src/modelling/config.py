from pathlib import Path

# ---- Model ----
MODEL_TYPE = "RandomForest"  # options: "LinearRegression", "Ridge", "RandomForest", "GradientBoosting"
MODEL_VERSION = "0.1.0"

# ---- Directories ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "assets" / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_PATH = MODELS_DIR / f"pipeline__v{MODEL_VERSION}.pkl"

# ---- Dataset / Columns ----
TARGET = "Age"
CATEGORICAL_COLS = ["Sex"]
NUMERIC_COLS = [
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
]

# ---- Data Split ----
TEST_SIZE = 0.2
RANDOM_STATE = 42
