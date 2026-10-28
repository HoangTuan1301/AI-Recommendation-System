/**
 * TypeScript Data Models for MusicAI Frontend
 */

export interface MusicModelInfo {
  id: string;
  name: string;
  family: string;
  description: string;
  is_personalized: boolean;
  is_default: boolean;
  is_available: boolean;
  is_loaded: boolean;
  checkpoint_file: string;
  checkpoint_size_kb?: number | null;
  latency_profile?: string | null;
}

export interface MusicModelsListResponse {
  total: number;
  default_model: string;
  models: MusicModelInfo[];
}

export interface RecommendedSong {
  rank: number;
  song_id: string;
  title: string;
  artist_name: string;
  track_id?: string | null;
  score: number;
}

export interface MusicRecommendationResponse {
  user_id: string;
  model_requested: string;
  model_used: string;
  fallback_used: boolean;
  fallback_reason?: string | null;
  top_k: number;
  count: number;
  inference_time_ms: number;
  recommendations: RecommendedSong[];
}

export interface SongDetailResponse {
  song_id: string;
  track_id?: string | null;
  title: string;
  artist_name: string;
  total_play_count?: number | null;
  unique_listeners?: number | null;
}

export interface UserHistorySong {
  song_id: string;
  title: string;
  artist_name: string;
  track_id?: string | null;
  play_count: number;
}

export interface UserHistoryResponse {
  user_id: string;
  total_plays: number;
  total_songs: number;
  history: UserHistorySong[];
}

export interface MusicHealthResponse {
  status: string;
  service: string;
  domain: string;
  catalog_songs: number;
  total_users: number;
  total_train_interactions: number;
  available_models: number;
  loaded_models: string[];
  default_model: string;
}
