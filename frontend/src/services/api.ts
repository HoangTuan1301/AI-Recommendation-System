/**
 * API Service Client for CineAI Backend
 */

import {
  CompareResponse,
  MetricsSummaryResponse,
  ModelInfo,
  MovieDetail,
  MoviePagination,
  RecommendationResponse,
  UserProfileResponse,
  UserSummary,
} from '../types';

const RAW_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');
const API_BASE = RAW_BASE ? `${RAW_BASE}/api/v1` : '/api/v1';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  try {
    const response = await fetch(`${API_BASE}${url}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    });

    if (!response.ok) {
      let errorMsg = `HTTP Error ${response.status}: ${response.statusText}`;
      try {
        const errorJson = await response.json();
        if (errorJson.detail) {
          errorMsg = errorJson.detail;
        }
      } catch {
        // Keep default errorMsg
      }
      throw new Error(errorMsg);
    }

    return response.json();
  } catch (err: any) {
    if (err.name === 'TypeError' && String(err.message).toLowerCase().includes('fetch')) {
      throw new Error(
        'Không thể kết nối tới máy chủ Backend. Máy chủ có thể đang khởi động lại (Render Cold Start mất khoảng 50-60s), vui lòng đợi giây lát và thử lại!'
      );
    }
    throw err;
  }
}

export const api = {
  // Health
  getHealth: () => fetchJson<{ status: string; catalog_movies: number; available_models: number }>('/health'),

  // Models
  getModels: async (): Promise<ModelInfo[]> => {
    const data = await fetchJson<{ total: number; models: ModelInfo[] }>('/models');
    return data.models;
  },

  // Movies
  getMovies: (params: { page?: number; limit?: number; q?: string; genre?: string; sort_by?: string }) => {
    const qp = new URLSearchParams();
    if (params.page) qp.append('page', params.page.toString());
    if (params.limit) qp.append('limit', params.limit.toString());
    if (params.q) qp.append('q', params.q);
    if (params.genre && params.genre !== 'All') qp.append('genre', params.genre);
    if (params.sort_by) qp.append('sort_by', params.sort_by);
    return fetchJson<MoviePagination>(`/movies?${qp.toString()}`);
  },

  getGenres: () => fetchJson<string[]>('/movies/genres'),

  getMovieDetail: (movieId: number) => fetchJson<MovieDetail>(`/movies/${movieId}`),

  // Users
  getUsers: async (page = 1, limit = 50): Promise<UserSummary[]> => {
    const data = await fetchJson<{ total: number; users: UserSummary[] }>(`/users?page=${page}&limit=${limit}`);
    return data.users;
  },

  getUserProfile: (userId: number) => fetchJson<UserProfileResponse>(`/users/${userId}/profile`),

  // Recommendations
  getRecommendations: (userId: number, modelId: string, topN = 10) =>
    fetchJson<RecommendationResponse>('/recommendations', {
      method: 'POST',
      body: JSON.stringify({ user_id: userId, model_id: modelId, top_n: topN }),
    }),

  // Metrics
  getMetrics: () => fetchJson<MetricsSummaryResponse>('/metrics'),

  // 1v1 Model Compare
  compareModels: (userId: number, modelAId: string, modelBId: string, topN = 10) =>
    fetchJson<CompareResponse>('/compare', {
      method: 'POST',
      body: JSON.stringify({ user_id: userId, model_a_id: modelAId, model_b_id: modelBId, top_n: topN }),
    }),
};
