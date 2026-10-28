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

const MUSIC_API_BASE = '/api/music';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
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
}

export const musicApi = {
  // Health & diagnostics
  getHealth: (): Promise<MusicHealthResponse> => fetchJson<MusicHealthResponse>('/health'),

  // Available models
  getModels: async (): Promise<MusicModelInfo[]> => {
    const data = await fetchJson<MusicModelsListResponse>('/models');
    return data.models;
  },

  // Song Recommendations
  getRecommendations: (
    userId: string,
    modelId: string = 'item_cf',
    topK: number = 10,
  ): Promise<MusicRecommendationResponse> => {
    const qp = new URLSearchParams({
      user_id: userId,
      model: modelId,
      top_k: topK.toString(),
    });
    return fetchJson<MusicRecommendationResponse>(`/recommendations?${qp.toString()}`);
  },

  // Song Details
  getSongDetail: (songId: string): Promise<SongDetailResponse> =>
    fetchJson<SongDetailResponse>(`/songs/${encodeURIComponent(songId)}`),

  // User Training History
  getUserHistory: (userId: string, limit: number = 50): Promise<UserHistoryResponse> =>
    fetchJson<UserHistoryResponse>(`/users/${encodeURIComponent(userId)}/history?limit=${limit}`),
};
