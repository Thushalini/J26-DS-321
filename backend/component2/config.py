from pathlib import Path

# backend/component2/  (the folder this file is in)
COMPONENT_DIR = Path(__file__).resolve().parent

DATA_DIR = COMPONENT_DIR / "data"                    # NOT pushed to GitHub
CANDIDATES_FILE = DATA_DIR / "candidate_repos.json"  # made by search_repos.py
CLONE_DIR = DATA_DIR / "clones"                      # temporary clones
RAW_DATA_DIR = DATA_DIR / "raw_data"                 # made by extract_repos.py
DB_PATH = DATA_DIR / "c2.db"                         # our database

# Secret word mixed into member codes so nobody can reverse them
ANON_SALT = "change-this-to-any-secret-word"

# The 7 DS lifecycle stages Component 2 will classify into (used from Step 2)
STAGES = [
    "data_cleaning",
    "eda",
    "feature_engineering",
    "model_development",
    "model_evaluation",
    "deployment_mlops",
    "documentation",
]