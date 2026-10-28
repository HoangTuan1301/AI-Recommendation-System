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

  // Song catalog
  getCatalog: (limit = 20, offset = 0) => {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    return fetchJson<{ total: number; limit: number; offset: number; songs: any[] }>(`/songs/catalog?${params.toString()}`);
  },

  // Song detail
  getSongDetail: (songId: string): Promise<SongDetailResponse> => {
    return fetchJson<SongDetailResponse>(`/songs/${encodeURIComponent(songId)}`);
  },

  // Top songs
  getTopSongs: (limit = 10) => {
    return fetchJson<{ total: number; limit: number; songs: any[] }>(`/songs/top?limit=${limit}`);
  },

  // Sample users
  getSampleUsers: () => {
    return fetchJson<{ count: number; sample_users: any[] }>('/users/sample_users');
  },

  // User listening history
  getUserHistory: (userId: string): Promise<UserHistoryResponse> => {
    return fetchJson<UserHistoryResponse>(`/users/sample_users/${encodeURIComponent(userId)}/history`);
  },

  // Recommendation inference
  getRecommendations: (
    userId: string,
    model = 'item_cf',
    topK = 10,
    includeDetails = true
  ): Promise<MusicRecommendationResponse> => {
    return fetchJson<MusicRecommendationResponse>('/recommend', {
      method: 'POST',
      body: JSON.stringify({
        user_id: userId,
        model: model,
        top_k: topK,
        include_details: includeDetails,
      }),
    });
  },
};
