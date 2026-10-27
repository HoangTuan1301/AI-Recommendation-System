"""Module for loading and validating MovieLens 100K dataset.

This module provides clean, robust data ingestion routines for MovieLens 100K,
including schema validation, foreign key checks, and data integrity verification.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GENRE_NAMES: List[str] = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

RATINGS_COLUMNS: List[str] = ["user_id", "movie_id", "rating", "timestamp"]
USERS_COLUMNS: List[str] = ["user_id", "age", "gender", "occupation", "zip_code"]
MOVIES_BASE_COLUMNS: List[str] = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
]


def load_ratings(file_path: Union[str, Path] = "data/raw/ml-100k/u.data") -> pd.DataFrame:
    """Load ratings data from u.data.

    Args:
        file_path: Path to the u.data file.

    Returns:
        pd.DataFrame: DataFrame containing [user_id, movie_id, rating, timestamp].

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file is malformed.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Ratings file not found at: {path.resolve()}")

    try:
        ratings_df = pd.read_csv(
            path,
            sep="\t",
            names=RATINGS_COLUMNS,
            header=None,
            dtype={
                "user_id": int,
                "movie_id": int,
                "rating": int,
                "timestamp": int,
            },
            engine="c",
        )
        logger.info("Successfully loaded ratings: %d rows.", len(ratings_df))
        return ratings_df
    except Exception as exc:
        logger.error("Failed to read ratings file: %s", exc)
        raise ValueError(f"Error parsing ratings file {path}: {exc}") from exc


def load_movies(file_path: Union[str, Path] = "data/raw/ml-100k/u.item") -> pd.DataFrame:
    """Load movie metadata from u.item.

    Args:
        file_path: Path to the u.item file.

    Returns:
        pd.DataFrame: DataFrame containing movie details and 19 binary genre columns.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file is malformed.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Movies file not found at: {path.resolve()}")

    column_names = MOVIES_BASE_COLUMNS + GENRE_NAMES
    try:
        movies_df = pd.read_csv(
            path,
            sep="|",
            names=column_names,
            header=None,
            encoding="ISO-8859-1",
            dtype={"movie_id": int, **{g: int for g in GENRE_NAMES}},
            engine="c",
        )
        logger.info("Successfully loaded movies: %d items.", len(movies_df))
        return movies_df
    except Exception as exc:
        logger.error("Failed to read movies file: %s", exc)
        raise ValueError(f"Error parsing movies file {path}: {exc}") from exc


def load_users(file_path: Union[str, Path] = "data/raw/ml-100k/u.user") -> pd.DataFrame:
    """Load user demographics from u.user.

    Args:
        file_path: Path to the u.user file.

    Returns:
        pd.DataFrame: DataFrame containing user demographics.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file is malformed.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Users file not found at: {path.resolve()}")

    try:
        users_df = pd.read_csv(
            path,
            sep="|",
            names=USERS_COLUMNS,
            header=None,
            dtype={"user_id": int, "age": int, "gender": str, "occupation": str, "zip_code": str},
            engine="c",
        )
        logger.info("Successfully loaded users: %d users.", len(users_df))
        return users_df
    except Exception as exc:
        logger.error("Failed to read users file: %s", exc)
        raise ValueError(f"Error parsing users file {path}: {exc}") from exc


def load_genres(file_path: Union[str, Path] = "data/raw/ml-100k/u.genre") -> Dict[int, str]:
    """Load genre id to name mapping from u.genre.

    Args:
        file_path: Path to the u.genre file.

    Returns:
        Dict[int, str]: Mapping from genre index to genre name.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Genre file not found at: {path.resolve()}")

    genres_df = pd.read_csv(path, sep="|", header=None, names=["genre_name", "genre_id"])
    genres_df = genres_df.dropna()
    return dict(zip(genres_df["genre_id"].astype(int), genres_df["genre_name"].astype(str)))


def load_occupations(file_path: Union[str, Path] = "data/raw/ml-100k/u.occupation") -> List[str]:
    """Load list of occupations from u.occupation.

    Args:
        file_path: Path to the u.occupation file.

    Returns:
        List[str]: List of occupation names.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Occupation file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        occupations = [line.strip() for line in f if line.strip()]
    return occupations


def load_movielens_100k(
    data_dir: Union[str, Path] = "data/raw/ml-100k",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience function to load ratings, movies, and users all together.

    Args:
        data_dir: Base directory containing uncompressed MovieLens 100K files.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (ratings, movies, users)
    """
    base_dir = Path(data_dir)
    ratings = load_ratings(base_dir / "u.data")
    movies = load_movies(base_dir / "u.item")
    users = load_users(base_dir / "u.user")
    return ratings, movies, users


def validate_movielens_data(
    ratings: pd.DataFrame, movies: pd.DataFrame, users: pd.DataFrame
) -> Dict[str, Any]:
    """Perform integrity, schema, and relationship checks on MovieLens DataFrames.

    Args:
        ratings: Ratings DataFrame.
        movies: Movies DataFrame.
        users: Users DataFrame.

    Returns:
        Dict[str, Any]: Validation summary report.

    Raises:
        ValueError: If critical integrity constraints are violated.
    """
    # 1. Null check on critical columns
    rating_nulls = int(ratings[["user_id", "movie_id", "rating"]].isnull().sum().sum())
    user_nulls = int(users["user_id"].isnull().sum())
    movie_nulls = int(movies["movie_id"].isnull().sum())
    if rating_nulls > 0 or user_nulls > 0 or movie_nulls > 0:
        raise ValueError(
            f"Null values detected in primary keys: ratings={rating_nulls}, users={user_nulls}, movies={movie_nulls}"
        )

    # 2. Duplicate interaction check
    duplicated_interactions = int(ratings.duplicated(subset=["user_id", "movie_id"]).sum())
    if duplicated_interactions > 0:
        raise ValueError(f"Found {duplicated_interactions} duplicate user-movie rating interactions.")

    # 3. Rating range validation [1, 5]
    min_rating = float(ratings["rating"].min())
    max_rating = float(ratings["rating"].max())
    if min_rating < 1 or max_rating > 5:
        raise ValueError(f"Rating out of bounds [1, 5]: found [{min_rating}, {max_rating}].")

    # 4. Foreign key integrity
    user_ids_in_ratings = set(ratings["user_id"])
    user_ids_in_users = set(users["user_id"])
    unmatched_users = user_ids_in_ratings - user_ids_in_users
    if unmatched_users:
        raise ValueError(f"Found {len(unmatched_users)} user_ids in ratings not present in users table.")

    movie_ids_in_ratings = set(ratings["movie_id"])
    movie_ids_in_movies = set(movies["movie_id"])
    unmatched_movies = movie_ids_in_ratings - movie_ids_in_movies
    if unmatched_movies:
        raise ValueError(f"Found {len(unmatched_movies)} movie_ids in ratings not present in movies table.")

    num_users = len(user_ids_in_ratings)
    num_movies = len(movie_ids_in_ratings)
    num_ratings = len(ratings)
    sparsity = 1.0 - (num_ratings / (num_users * num_movies))

    report = {
        "is_valid": True,
        "num_ratings": num_ratings,
        "num_users": num_users,
        "num_movies": num_movies,
        "min_rating": min_rating,
        "max_rating": max_rating,
        "duplicated_interactions": duplicated_interactions,
        "user_nulls": user_nulls,
        "movie_nulls": movie_nulls,
        "rating_nulls": rating_nulls,
        "matrix_sparsity_percent": round(sparsity * 100, 4),
    }
    logger.info("Validation passed successfully: %s", report)
    return report


if __name__ == "__main__":
    r, m, u = load_movielens_100k()
    val_report = validate_movielens_data(r, m, u)
    print("\n--- Validation Report Summary ---")
    for k, v in val_report.items():
        print(f"{k}: {v}")
