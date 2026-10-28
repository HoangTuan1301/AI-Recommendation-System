/**
 * API Service Client for MusicAI Backend
 */

import {
  MusicHealthResponse,
  MusicModelInfo,
  MusicModelsListResponse,
  MusicRecommendationResponse,
  SongDetailResponse,
  UserHistoryResponse,
} from '../types/music';

const RAW_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');
const MUSIC_API_BASE = RAW_BASE ? `${RAW_BASE}/api/music` : '/api/music';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  try {
    const response = await fetch(`${MUSIC_API_BASE}${url}`, {
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
        // Use fallback errorMsg
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

export const musicApi = {
  // Health & diagnostics
  getHealth: (): Promise<MusicHealthResponse> => fetchJson<MusicHealthResponse>('/health'),

  // Available models
  getModels: async (): Promise<MusicModelInfo[]> => {
    const data = await fetchJson<MusicModelsListResponse>('/models');
    return data.models;
  },

  // Song detail
  getSongDetail: (songId: string): Promise<SongDetailResponse> => {
    return fetchJson<SongDetailResponse>(`/songs/${encodeURIComponent(songId.trim())}`);
  },

  // User listening history strictly from training split
  getUserHistory: (userId: string, limit = 50): Promise<UserHistoryResponse> => {
    const params = new URLSearchParams({ limit: String(limit) });
    return fetchJson<UserHistoryResponse>(
      `/users/${encodeURIComponent(userId.trim())}/history?${params.toString()}`
    );
  },

  // Recommendation inference (GET /api/music/recommendations?user_id=...&model=...&top_k=...)
  getRecommendations: (
    userId: string,
    model = 'item_cf',
    topK = 10
  ): Promise<MusicRecommendationResponse> => {
    const params = new URLSearchParams({
      user_id: userId.trim(),
      model: model.trim(),
      top_k: String(topK),
    });
    return fetchJson<MusicRecommendationResponse>(`/recommendations?${params.toString()}`);
  },
};
