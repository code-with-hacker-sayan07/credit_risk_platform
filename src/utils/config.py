import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SQL_DIR = BASE_DIR / "sql"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
DOCUMENTS_DIR = BASE_DIR / "documents"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, SQL_DIR, NOTEBOOKS_DIR, DOCUMENTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "credit_risk.db"
MODEL_PATH = MODELS_DIR / "credit_model.joblib"
METADATA_PATH = MODELS_DIR / "feature_metadata.json"

# API keys and configs
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")
