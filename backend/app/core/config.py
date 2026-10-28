"""Core Configuration for CineAI FastAPI Backend."""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from pathlib import Path
from typing import List
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseModel):
    PROJECT_NAME: str = "CineAI - Movie Recommendation API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Paths - CineAI (Movies)
    REPO_ROOT_PATH: Path = REPO_ROOT
    DATA_PROCESSED_PATH: Path = REPO_ROOT / "data" / "processed"
    CHECKPOINTS_PATH: Path = REPO_ROOT / "models" / "checkpoints"
    RESULTS_PATH: Path = REPO_ROOT / "results"
    
    # Paths - MusicAI (Songs)
    DATA_MUSIC_PATH: Path = REPO_ROOT / "data" / "music" / "processed"
    MUSIC_CHECKPOINTS_PATH: Path = REPO_ROOT / "models" / "checkpoints" / "music"
    DEFAULT_MUSIC_MODEL: str = "item_cf"
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8501",  # Streamlit fallback origin
    ]


settings = Settings()
