"""Model Registry and Inference Service for FastAPI Backend."""

from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import joblib
import pandas as pd

from backend.app.core.config import settings
from backend.app.schemas.model import ModelInfo
from backend.app.schemas.recommendation import RecommendedMovie, RecommendationResponse
from backend.app.services.movie_service import movie_service
from backend.app.services.poster_service import generate_svg_poster

# Ensure repository root is on sys.path
REPO_ROOT = settings.REPO_ROOT_PATH
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.model_loader import MODEL_REGISTRY, load_or_train_model, CHECKPOINTS_DIR


class ModelService:
    """Manages AI recommendation models, checkpoint lazy-loading, and real-time inference."""

    def __init__(self) -> None:
        self.registry = MODEL_REGISTRY
        self.cached_models: Dict[str, Any] = {}
        
        # Load train dataframe to extract user training history
        train_path = settings.DATA_PROCESSED_PATH / "train.csv"
        self.train_df = pd.read_csv(train_path)
        
        # Precompute user seen movie IDs from Train
        self.user_seen_movies: Dict[int, set] = (
            self.train_df.groupby("user_id")["movie_id"].apply(set).to_dict()
        )
        self.all_movie_ids = list(movie_service.movies_by_id.keys())

    def get_model_info_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        for key, meta in self.registry.items():
            if meta["id"] == model_id:
                return {"display_key": key, **meta}
        return None

    def list_models(self) -> List[ModelInfo]:
        models = []
        for meta in self.registry.values():
            models.append(ModelInfo(
                id=meta["id"],
                name=meta["name"],
                family=meta["family"],
                score_type=meta["score_type"],
                score_range=meta["score_range"],
                score_format=meta["score_format"],
                description=meta["description"],
                complexity=meta["complexity"],
            ))
        return models

    def get_loaded_model(self, model_id: str) -> Any:
        info = self.get_model_info_by_id(model_id)
        if not info:
            raise ValueError(f"Unknown model_id: '{model_id}'")

        display_key = info["display_key"]
        if display_key in self.cached_models:
            return self.cached_models[display_key]

        # Load from checkpoint
        ckpt_path = CHECKPOINTS_DIR / f"{model_id}.pkl"
        if ckpt_path.exists():
            model = joblib.load(ckpt_path)
            model.is_fitted = True
        else:
            movies_df = pd.read_csv(settings.DATA_PROCESSED_PATH / "movies_clean.csv")
            model = load_or_train_model(display_key, train_df=self.train_df, movies_df=movies_df)

        self.cached_models[display_key] = model
        return model

    def get_recommendations(
        self,
        user_id: int,
        model_id: str = "neural_cf",
        top_n: int = 10,
    ) -> RecommendationResponse:
        info = self.get_model_info_by_id(model_id)
        if not info:
            raise ValueError(f"Invalid model_id: '{model_id}'")

        t_start = time.perf_counter()
        model = self.get_loaded_model(model_id)

        # Guarantee candidates strictly exclude user's Train interactions
        seen_train = self.user_seen_movies.get(user_id, set())
        candidate_ids = [mid for mid in self.all_movie_ids if mid not in seen_train]

        rec_ids = model.recommend(user_id=user_id, top_n=top_n, candidate_items=candidate_ids)
        inference_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

        fmt = info.get("score_format", "{:.2f}")
        stype = info.get("score_type", "Score")

        recommended_movies: List[RecommendedMovie] = []
        for rank, mid in enumerate(rec_ids, start=1):
            meta = movie_service.movies_by_id.get(mid, {})
            title = str(meta.get("title", f"Movie #{mid}"))
            year = int(meta["year"]) if pd.notna(meta.get("year")) else None
            genres_str = str(meta.get("genres_str", ""))
            genres = genres_str.split("|") if genres_str and genres_str != "nan" else []

            raw_score = 0.0
            try:
                raw_score = float(model.predict(user_id, mid))
            except Exception:
                raw_score = 0.0

            try:
                formatted_score = fmt.format(raw_score)
            except Exception:
                formatted_score = f"{raw_score:.2f}"

            poster_url = generate_svg_poster(title, year, genres, mid)

            recommended_movies.append(RecommendedMovie(
                movie_id=mid,
                title=title,
                year=year,
                genres=genres,
                genres_str=genres_str,
                rank=rank,
                raw_score=raw_score,
                formatted_score=formatted_score,
                score_type=stype,
                poster_url=poster_url,
            ))

        return RecommendationResponse(
            user_id=user_id,
            model_id=model_id,
            model_name=info["name"],
            top_n=top_n,
            count=len(recommended_movies),
            inference_time_ms=inference_time_ms,
            recommendations=recommended_movies,
        )


model_service = ModelService()
