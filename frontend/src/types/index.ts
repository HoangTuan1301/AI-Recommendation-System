/**
 * TypeScript Data Models for CineAI Frontend
 */

export interface MovieSummary {
  movie_id: number;
  title: string;
  year?: number | null;
  release_date?: string | null;
  genres: string[];
  genres_str: string;
  avg_rating?: number | null;
  rating_count: number;
  poster_url?: string | null;
}

export interface MovieDetail extends MovieSummary {
  imdb_url?: string | null;
  rating_distribution?: Record<number, number> | null;
  similar_movies: MovieSummary[];
}

export interface MoviePagination {
  total: number;
  page: number;
  limit: number;
  items: MovieSummary[];
}

export interface ModelInfo {
  id: string;
  name: string;
  family: string;
  score_type: string;
  score_range: string;
  score_format: string;
  description: string;
  complexity: string;
}

export interface RecommendedMovie {
  movie_id: number;
  title: string;
  year?: number | null;
  genres: string[];
  genres_str: string;
  rank: number;
  raw_score?: number | null;
  formatted_score: string;
  score_type: string;
  avg_rating?: number | null;
  rating_count?: number;
  poster_url?: string | null;
}

export interface RecommendationResponse {
  user_id: number;
  model_id: string;
  model_name: string;
  top_n: number;
  count: number;
  inference_time_ms: number;
  recommendations: RecommendedMovie[];
}

export interface UserSummary {
  user_id: number;
  ratings_count: number;
  avg_rating: number;
  high_rated_count: number;
}

export interface GenreAffinity {
  genre: string;
  count: number;
  percentage: number;
}

export interface UserProfileResponse {
  user_id: number;
  ratings_count: number;
  avg_rating: number;
  high_rated_count: number;
  rating_distribution: Record<number, number>;
  favorite_genres: GenreAffinity[];
  top_rated_movies: MovieSummary[];
}

export interface ModelMetric {
  model: string;
  model_id: string;
  precision_at_10: number;
  recall_at_10: number;
  f1_at_10: number;
  ndcg_at_10: number;
  hit_rate_at_10: number;
  evaluation_users: number;
  fit_time_seconds: number;
  evaluate_time_seconds: number;
  total_time_seconds: number;
}

export interface MetricsSummaryResponse {
  total_models: number;
  best_hit_rate_model: string;
  best_ndcg_model: string;
  best_precision_model: string;
  metrics: ModelMetric[];
}

export interface CompareResponse {
  user_id: number;
  model_a_id: string;
  model_a_name: string;
  model_b_id: string;
  model_b_name: string;
  top_n: number;
  jaccard_similarity: number;
  overlap_count: number;
  recommendations_a: RecommendedMovie[];
  recommendations_b: RecommendedMovie[];
  shared_movie_ids: number[];
}
