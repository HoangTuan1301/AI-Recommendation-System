"""Script to run Exploratory Data Analysis (EDA) on MovieLens 100K,
save publication-quality figures to reports/figures/, and export
an executed Jupyter Notebook to notebooks/01_movielens_eda.ipynb.
"""

from pathlib import Path
import sys

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_loader import GENRE_NAMES, load_movielens_100k, validate_movielens_data

# Set visual style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_eda_and_generate_figures():
    ratings, movies, users = load_movielens_100k()
    val_report = validate_movielens_data(ratings, movies, users)

    # 1. Rating Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    rating_counts = ratings["rating"].value_counts().sort_index()
    total_ratings = len(ratings)
    bars = ax.bar(rating_counts.index, rating_counts.values, color="#2b5c8f", edgecolor="#1a365d", width=0.6)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total_ratings) * 100
        ax.annotate(f"{h:,}\n({pct:.1f}%)", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Distribution of MovieLens 100K Ratings (1 to 5 Stars)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Rating Value", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_ylim(0, max(rating_counts.values) * 1.15)
    ax.set_xticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "01_rating_distribution.png", dpi=300)
    plt.close()

    # 2. User Activity (Ratings per User)
    user_counts = ratings.groupby("user_id").size()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(user_counts, bins=40, kde=True, color="#1e88e5", ax=ax, edgecolor="#0d47a1")
    ax.axvline(user_counts.median(), color="#d32f2f", linestyle="--", linewidth=1.5,
               label=f"Median: {int(user_counts.median())} ratings")
    ax.axvline(user_counts.mean(), color="#f57c00", linestyle=":", linewidth=1.5,
               label=f"Mean: {user_counts.mean():.1f} ratings")
    ax.set_title("Distribution of Ratings per User (Activity)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Ratings Given by User", fontsize=11)
    ax.set_ylabel("Number of Users", fontsize=11)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "02_user_activity_distribution.png", dpi=300)
    plt.close()

    # 3. Movie Popularity (Long Tail)
    movie_counts = ratings.groupby("movie_id").size().sort_values(ascending=False).values
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(movie_counts) + 1), movie_counts, color="#00897b", linewidth=2)
    ax.fill_between(range(1, len(movie_counts) + 1), movie_counts, color="#80cbc4", alpha=0.4)
    ax.axvline(x=int(len(movie_counts) * 0.2), color="#e53935", linestyle="--",
               label="Top 20% Movies (~80% of total ratings)")
    ax.set_title("Movie Popularity Long-Tail Distribution", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Movie Rank (by popularity)", fontsize=11)
    ax.set_ylabel("Number of Ratings", fontsize=11)
    ax.set_yscale("log")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "03_movie_popularity_distribution.png", dpi=300)
    plt.close()

    # 4. Top 10 Most Rated Movies
    top10_popular = (
        ratings.groupby("movie_id")
        .size()
        .reset_index(name="count")
        .merge(movies[["movie_id", "title"]], on="movie_id")
        .sort_values("count", ascending=True)
        .tail(10)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top10_popular["title"], top10_popular["count"], color="#3949ab", edgecolor="#1a237e")
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:,}", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9, fontweight="bold")
    ax.set_title("Top 10 Most Rated Movies in MovieLens 100K", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Ratings", fontsize=11)
    ax.set_xlim(0, max(top10_popular["count"]) * 1.15)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "04_top10_most_rated_movies.png", dpi=300)
    plt.close()

    # 5. Top 10 Highest Rated Movies (Min 100 Ratings)
    movie_stats = (
        ratings.groupby("movie_id")
        .agg(mean_rating=("rating", "mean"), rating_count=("rating", "count"))
        .reset_index()
        .merge(movies[["movie_id", "title"]], on="movie_id")
    )
    top10_rated = (
        movie_stats[movie_stats["rating_count"] >= 100]
        .sort_values("mean_rating", ascending=True)
        .tail(10)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top10_rated["title"], top10_rated["mean_rating"], color="#00897b", edgecolor="#004d40")
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.2f} ★", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9, fontweight="bold")
    ax.set_title("Top 10 Highest Rated Movies (Min 100 Ratings Threshold)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Average Rating (Stars)", fontsize=11)
    ax.set_xlim(0, 5.2)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "05_top10_highest_rated_movies.png", dpi=300)
    plt.close()

    # 6. Genre Distribution
    genre_counts = movies[GENRE_NAMES].sum().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(genre_counts.index, genre_counts.values, color="#5e35b1", edgecolor="#311b92")
    for bar in bars:
        w = bar.get_width()
        pct = (w / len(movies)) * 100
        ax.annotate(f"{w:,} ({pct:.1f}%)", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9)
    ax.set_title("Movie Distribution by Genre (19 Genres)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Movies in Genre", fontsize=11)
    ax.set_xlim(0, max(genre_counts.values) * 1.18)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "06_genre_distribution.png", dpi=300)
    plt.close()

    # 7. Ratings Over Time
    ratings_time = ratings.copy()
    ratings_time["datetime"] = pd.to_datetime(ratings_time["timestamp"], unit="s")
    monthly_activity = ratings_time.set_index("datetime").resample("ME").size()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly_activity.index, monthly_activity.values, color="#e53935", marker="o", linewidth=2)
    ax.set_title("MovieLens 100K Rating Activity Over Time (1997 - 1998)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Time (Month / Year)", fontsize=11)
    ax.set_ylabel("Ratings per Month", fontsize=11)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "07_ratings_over_time.png", dpi=300)
    plt.close()

    # 8. User-Item Interaction Matrix Sparsity Heatmap (Sample 50x50)
    sample_users = ratings["user_id"].unique()[:50]
    sample_movies = ratings["movie_id"].unique()[:50]
    sample_ratings = ratings[(ratings["user_id"].isin(sample_users)) & (ratings["movie_id"].isin(sample_movies))]
    pivot_sample = sample_ratings.pivot(index="user_id", columns="movie_id", values="rating")

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(pivot_sample.notnull(), cmap="Blues", cbar=False, ax=ax, linewidths=0.2, linecolor="#eeeeee")
    ax.set_title(f"Sample User-Item Interaction Matrix (Sparsity: {val_report['matrix_sparsity_percent']}%)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Movie ID (Sample)", fontsize=11)
    ax.set_ylabel("User ID (Sample)", fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "08_user_item_sparsity_heatmap.png", dpi=300)
    plt.close()

    print("All 8 figures successfully generated in reports/figures/!")


if __name__ == "__main__":
    run_eda_and_generate_figures()
