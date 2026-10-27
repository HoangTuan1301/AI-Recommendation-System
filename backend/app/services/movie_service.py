"""Movie Metadata and Catalog Service."""

from typing import Dict, List, Optional
import pandas as pd
from backend.app.core.config import settings
from backend.app.schemas.movie import MovieDetail, MoviePagination, MovieSummary
from backend.app.services.poster_service import generate_svg_poster


class MovieService:
    """Manages MovieLens 100K catalog, search, and details."""

    def __init__(self) -> None:
        movies_path = settings.DATA_PROCESSED_PATH / "movies_clean.csv"
        train_path = settings.DATA_PROCESSED_PATH / "train.csv"
        
        self.movies_df = pd.read_csv(movies_path)
        self.train_df = pd.read_csv(train_path)
        
        # Precompute rating statistics per movie from Train history strictly
        stats = self.train_df.groupby("movie_id")["rating"].agg(["count", "mean"]).reset_index()
        stats.columns = ["movie_id", "train_rating_count", "train_avg_rating"]
        self.movies_df = self.movies_df.merge(stats, on="movie_id", how="left")
        self.movies_df["train_rating_count"] = self.movies_df["train_rating_count"].fillna(0).astype(int)
        self.movies_df["train_avg_rating"] = self.movies_df["train_avg_rating"].fillna(0.0).round(2)

        # Index by ID
        self.movies_by_id: Dict[int, dict] = self.movies_df.set_index("movie_id").to_dict("index")
        
        # Cache genre lists
        self.all_genres = [
            "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
            "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ]

    def _to_summary(self, mid: int) -> Optional[MovieSummary]:
        meta = self.movies_by_id.get(mid)
        if not meta:
            return None
        
        genres_str = str(meta.get("genres_str", ""))
        genres = genres_str.split("|") if genres_str and genres_str != "nan" else []
        year = int(meta["year"]) if pd.notna(meta.get("year")) else None
        title = str(meta.get("title", f"Movie #{mid}"))
        poster_url = generate_svg_poster(title, year, genres, mid)

        return MovieSummary(
            movie_id=mid,
            title=title,
            year=year,
            release_date=str(meta.get("release_date", "")) if pd.notna(meta.get("release_date")) else None,
            genres=genres,
            genres_str=genres_str,
            avg_rating=float(meta.get("train_avg_rating", 0.0)),
            rating_count=int(meta.get("train_rating_count", 0)),
            poster_url=poster_url,
        )

    def get_movies(
        self,
        page: int = 1,
        limit: int = 20,
        query: Optional[str] = None,
        genre: Optional[str] = None,
        sort_by: str = "popular",  # 'popular', 'rating', 'title', 'year'
    ) -> MoviePagination:
        df = self.movies_df.copy()

        if query:
            clean_q = query.strip().lower()
            df = df[df["title"].str.lower().str.contains(clean_q, na=False)]

        if genre and genre != "All":
            df = df[df[genre] == 1] if genre in df.columns else df[df["genres_str"].str.contains(genre, na=False)]

        if sort_by == "popular":
            df = df.sort_values(by=["train_rating_count", "train_avg_rating"], ascending=[False, False])
        elif sort_by == "rating":
            df = df.sort_values(by=["train_avg_rating", "train_rating_count"], ascending=[False, False])
        elif sort_by == "year":
            df = df.sort_values(by="year", ascending=False)
        elif sort_by == "title":
            df = df.sort_values(by="title", ascending=True)

        total = len(df)
        start = (page - 1) * limit
        end = start + limit
        page_df = df.iloc[start:end]

        items = [self._to_summary(int(row["movie_id"])) for _, row in page_df.iterrows() if self._to_summary(int(row["movie_id"]))]
        return MoviePagination(total=total, page=page, limit=limit, items=[it for it in items if it is not None])

    def get_movie_detail(self, movie_id: int) -> Optional[MovieDetail]:
        base = self._to_summary(movie_id)
        if not base:
            return None
        
        meta = self.movies_by_id[movie_id]
        
        # Rating distribution from Train
        m_ratings = self.train_df[self.train_df["movie_id"] == movie_id]["rating"].value_counts().to_dict()
        dist = {int(r): int(m_ratings.get(r, 0)) for r in range(1, 6)}
        
        # Similar movies in the same primary genre
        similar: List[MovieSummary] = []
        if base.genres:
            primary_g = base.genres[0]
            if primary_g in self.movies_df.columns:
                sim_df = self.movies_df[
                    (self.movies_df[primary_g] == 1) & (self.movies_df["movie_id"] != movie_id)
                ].sort_values(by="train_rating_count", ascending=False).head(6)
                for _, s_row in sim_df.iterrows():
                    sm = self._to_summary(int(s_row["movie_id"]))
                    if sm:
                        similar.append(sm)

        return MovieDetail(
            **base.model_dump(),
            imdb_url=str(meta.get("imdb_url", "")) if pd.notna(meta.get("imdb_url")) else None,
            rating_distribution=dist,
            similar_movies=similar,
        )


movie_service = MovieService()
