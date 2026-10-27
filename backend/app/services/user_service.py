"""User Profile Service for FastAPI Backend."""

from typing import Dict, List, Optional
import pandas as pd
from backend.app.core.config import settings
from backend.app.schemas.user import GenreAffinity, UserProfileResponse, UserSummary, UsersListResponse
from backend.app.services.movie_service import movie_service


class UserService:
    """Provides user statistics and profile exploration strictly derived from train.csv."""

    def __init__(self) -> None:
        train_path = settings.DATA_PROCESSED_PATH / "train.csv"
        self.train_df = pd.read_csv(train_path)
        
        # User summaries
        user_grouped = self.train_df.groupby("user_id")
        self.user_stats: Dict[int, dict] = {}
        
        for uid, group in user_grouped:
            cnt = len(group)
            mean_r = round(float(group["rating"].mean()), 2)
            high_cnt = int((group["rating"] >= 4.0).sum())
            self.user_stats[uid] = {
                "user_id": uid,
                "ratings_count": cnt,
                "avg_rating": mean_r,
                "high_rated_count": high_cnt,
            }

    def list_users(self, page: int = 1, limit: int = 50) -> UsersListResponse:
        all_users = sorted(self.user_stats.values(), key=lambda u: u["user_id"])
        total = len(all_users)
        start = (page - 1) * limit
        end = start + limit
        items = [UserSummary(**u) for u in all_users[start:end]]
        return UsersListResponse(total=total, users=items)

    def get_user_profile(self, user_id: int) -> Optional[UserProfileResponse]:
        stats = self.user_stats.get(user_id)
        if not stats:
            return None

        u_rows = self.train_df[self.train_df["user_id"] == user_id]

        # Rating distribution
        r_counts = u_rows["rating"].value_counts().to_dict()
        dist = {int(r): int(r_counts.get(r, 0)) for r in range(1, 6)}

        # Genre affinities
        genre_counter: Dict[str, int] = {}
        for mid in u_rows["movie_id"]:
            meta = movie_service.movies_by_id.get(mid)
            if meta:
                g_str = str(meta.get("genres_str", ""))
                if g_str and g_str != "nan":
                    for g in g_str.split("|"):
                        genre_counter[g] = genre_counter.get(g, 0) + 1

        total_genre_hits = sum(genre_counter.values()) or 1
        fav_genres: List[GenreAffinity] = []
        for g, count in sorted(genre_counter.items(), key=lambda x: x[1], reverse=True)[:8]:
            fav_genres.append(GenreAffinity(
                genre=g,
                count=count,
                percentage=round((count / total_genre_hits) * 100, 1),
            ))

        # Top rated movies from Train
        top_mids = (
            u_rows[u_rows["rating"] >= 4.0]
            .sort_values(by=["rating", "timestamp"], ascending=[False, False])["movie_id"]
            .head(10)
            .tolist()
        )
        top_movies = [movie_service._to_summary(mid) for mid in top_mids if movie_service._to_summary(mid)]

        return UserProfileResponse(
            user_id=user_id,
            ratings_count=stats["ratings_count"],
            avg_rating=stats["avg_rating"],
            high_rated_count=stats["high_rated_count"],
            rating_distribution=dist,
            favorite_genres=fav_genres,
            top_rated_movies=[m for m in top_movies if m is not None],
        )


user_service = UserService()
