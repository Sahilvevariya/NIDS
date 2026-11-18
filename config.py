"""Configuration constants for the Network Intrusion Detection project."""
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

# Model artifact paths
MODELS_DIR = PROJECT_ROOT / "models"
STAGE1_DIR = MODELS_DIR / "stage1"
STAGE2_DIR = MODELS_DIR / "stage2"

# Stage 1 artifacts
STAGE1_MODEL_PATH = STAGE1_DIR / "binary_model.joblib"
STAGE1_SCALER_PATH = STAGE1_DIR / "scaler.joblib"
STAGE1_PCA_PATH = STAGE1_DIR / "pca.joblib"
STAGE1_FEATURES_PATH = STAGE1_DIR / "feature_columns.json"
STAGE1_LABELS_PATH = STAGE1_DIR / "label_mapping.json"

# Stage 2 artifacts
STAGE2_MODEL_PATH = STAGE2_DIR / "attack_model.joblib"
STAGE2_SCALER_PATH = STAGE2_DIR / "scaler.joblib"
STAGE2_FEATURES_PATH = STAGE2_DIR / "feature_columns.json"
STAGE2_LABELS_PATH = STAGE2_DIR / "label_mapping.json"

# Metadata
METADATA_PATH = MODELS_DIR / "metadata.json"

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DEMO_SAMPLE_PATH = DATA_DIR / "demo_sample.csv"

# Results
RESULTS_DIR = PROJECT_ROOT / "results"

# Stage 1 constants
STAGE1_PCA_COMPONENTS = 25
STAGE1_EXPECTED_VARIANCE = 0.8689  # From synthetic data training

# Stage 2 constants
STAGE2_MAX_SAMPLES_PER_CLASS = 10000
STAGE2_RANDOM_STATE = 42
STAGE2_TEST_SIZE = 0.20

# Columns to exclude from features (both stages)
LEAKAGE_COLUMNS = [
    "Label",
    "Attack",
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
]

# Columns to exclude from Stage 2 only (in addition to above)
STAGE2_EXTRA_LEAKAGE = []

# Numeric replacement values
INF_REPLACEMENT = 0.0
NAN_REPLACEMENT = 0.0

# Replay defaults
DEFAULT_BATCH_SIZE = 50
DEFAULT_INTERVAL_SECONDS = 3
MIN_BATCH_SIZE = 10
MAX_BATCH_SIZE = 500
MIN_INTERVAL = 2
MAX_INTERVAL = 5

# UI constants
APP_TITLE = "Network Intrusion Detection & Attack Classification System"
APP_SUBTITLE = "Machine Learning-Based Network Security Analysis Using CIC-IDS2017"
PROTOTYPE_BANNER = "Dataset-based replay prototype. Not a live NIDS."

# Color scheme
COLORS = {
    "benign": "#2ECC71",      # Green
    "attack": "#E74C3C",      # Red
    "warning": "#F39C12",     # Amber
    "info": "#3498DB",        # Blue
    "background": "#1E1E2E",  # Dark navy
    "surface": "#282A36",     # Charcoal
    "text": "#F8F8F2",        # Off-white
}