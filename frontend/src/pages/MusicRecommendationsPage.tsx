import React, { useEffect, useState } from 'react';
import {
  Sparkles,
  Music,
  User,
  History,
  AlertTriangle,
  RotateCw,
  Clock,
  CheckCircle2,
} from 'lucide-react';
import { useDomain, SAMPLE_MUSIC_USERS } from '../context/DomainContext';
import { musicApi } from '../services/musicApi';
import { MusicModelInfo, MusicRecommendationResponse } from '../types/music';
import { MusicModelSelector } from '../components/music/MusicModelSelector';
import { SongCard } from '../components/music/SongCard';

export const MusicRecommendationsPage: React.FC = () => {
  const { musicUserId, setMusicUserId, setHistoryModalOpen } = useDomain();

  const [models, setModels] = useState<MusicModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('item_cf');
  const [topK, setTopK] = useState<number>(10);
  const [customUserId, setCustomUserId] = useState<string>(musicUserId);

  const [recommendations, setRecommendations] = useState<MusicRecommendationResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Load available models on mount
  useEffect(() => {
    musicApi
      .getModels()
      .then((data: MusicModelInfo[]) => {
        setModels(data);
        const def = data.find((m: MusicModelInfo) => m.is_default);
        if (def) setSelectedModel(def.id);
      })
      .catch((err: Error) => {
        setError('Could not load AI models from backend: ' + err.message);
      });
  }, []);

  // Fetch recommendations function
  const fetchRecommendations = (targetUser: string, modelId: string, k: number) => {
    setLoading(true);
    setError(null);

    musicApi
      .getRecommendations(targetUser, modelId, k)
      .then((res: MusicRecommendationResponse) => {
        setRecommendations(res);
        setLoading(false);
      })
      .catch((err: Error) => {
        setError(err.message || 'Failed to fetch song recommendations.');
        setLoading(false);
      });
  };

  // Initial load
  useEffect(() => {
    fetchRecommendations(musicUserId, selectedModel, topK);
  }, []);

  const handleSelectSampleUser = (userId: string) => {
    setMusicUserId(userId);
    setCustomUserId(userId);
    fetchRecommendations(userId, selectedModel, topK);
  };

  const handleApplyForm = (e: React.FormEvent) => {
    e.preventDefault();
    if (!customUserId.trim()) return;
    setMusicUserId(customUserId.trim());
    fetchRecommendations(customUserId.trim(), selectedModel, topK);
  };

  const handleModelChange = (newModelId: string) => {
    setSelectedModel(newModelId);
    fetchRecommendations(musicUserId, newModelId, topK);
  };

  const handleTopKChange = (newK: number) => {
    setTopK(newK);
    fetchRecommendations(musicUserId, selectedModel, newK);
  };

  return (
    <div className="min-h-screen pt-24 pb-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-8 animate-fadeIn">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-800 pb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-black uppercase tracking-wider bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 flex items-center gap-1.5">
              <Music className="w-3.5 h-3.5" />
              MusicAI Domain
            </span>
            <span className="text-xs text-zinc-400">Million Song Dataset 10-Core</span>
          </div>

          <h1 className="text-3xl sm:text-4xl font-black text-white tracking-tight">
            AI Song Recommendations
          </h1>
          <p className="text-sm text-zinc-400 mt-1 max-w-2xl">
            Explore personalized tracks generated across 6 collaborative filtering algorithms.
            Evaluated on real implicit feedback listening behavior.
          </p>
        </div>

        {/* View History Button */}
        <button
          onClick={() => setHistoryModalOpen(true)}
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl font-bold text-xs bg-zinc-800/90 hover:bg-emerald-600 text-zinc-200 hover:text-white border border-zinc-700 hover:border-emerald-500 transition-all shadow-md self-start md:self-center"
        >
          <History className="w-4 h-4 text-emerald-400" />
          View Training Listening History
        </button>
      </div>

      {/* Control Panel Card */}
      <div className="bg-cine-surface border border-cine-border rounded-2xl p-6 shadow-xl space-y-6">
        {/* Sample User Selector Pills */}
        <div>
          <label className="text-xs font-bold uppercase tracking-wider text-zinc-400 mb-2 flex items-center gap-1.5">
            <User className="w-3.5 h-3.5 text-emerald-400" />
            Quick Demo Users:
          </label>
          <div className="flex flex-wrap gap-2">
            {SAMPLE_MUSIC_USERS.map((u) => {
              const isSelected = musicUserId === u.id;
              return (
                <button
                  key={u.id}
                  onClick={() => handleSelectSampleUser(u.id)}
                  className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all flex items-center gap-2 border ${
                    isSelected
                      ? 'bg-emerald-600 text-white border-emerald-400 shadow-md shadow-emerald-950/50'
                      : 'bg-zinc-800/80 text-zinc-300 border-zinc-700 hover:bg-zinc-700 hover:text-white'
                  }`}
                >
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  {u.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* User ID & Top-K Form */}
        <form onSubmit={handleApplyForm} className="grid grid-cols-1 md:grid-cols-12 gap-4 items-end">
          <div className="md:col-span-8">
            <label className="block text-xs font-bold uppercase tracking-wider text-zinc-400 mb-1">
              Custom User ID (MSD raw identifier):
            </label>
            <input
              type="text"
              value={customUserId}
              onChange={(e) => setCustomUserId(e.target.value)}
              placeholder="e.g. 0023d50b3c99c4352585408b1b21f3762655f434"
              className="w-full px-4 py-2.5 rounded-xl bg-zinc-950 border border-zinc-700/80 text-white font-mono text-xs focus:outline-none focus:border-emerald-500 transition-colors shadow-inner"
            />
          </div>

          <div className="md:col-span-2">
            <label className="block text-xs font-bold uppercase tracking-wider text-zinc-400 mb-1">
              Top-K Songs:
            </label>
            <div className="flex items-center gap-1">
              {[5, 10, 20].map((k) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => handleTopKChange(k)}
                  className={`flex-1 py-2 rounded-xl text-xs font-bold border transition-all ${
                    topK === k
                      ? 'bg-emerald-600 text-white border-emerald-400'
                      : 'bg-zinc-900 text-zinc-400 border-zinc-800 hover:text-white'
                  }`}
                >
                  {k}
                </button>
              ))}
            </div>
          </div>

          <div className="md:col-span-2">
            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 rounded-xl font-bold text-xs bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-950/60 transition-all flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {loading ? (
                <>
                  <RotateCw className="w-4 h-4 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4 fill-white" />
                  Generate
                </>
              )}
            </button>
          </div>
        </form>

        {/* AI Model Selector */}
        {models.length > 0 && (
          <div className="pt-4 border-t border-zinc-800">
            <MusicModelSelector
              models={models}
              selectedModelId={selectedModel}
              onSelectModel={handleModelChange}
              disabled={loading}
            />
          </div>
        )}
      </div>

      {/* Results Header Banner */}
      {recommendations && (
        <div className="space-y-4">
          <div className="p-4 rounded-2xl bg-zinc-900 border border-zinc-800 flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-black">
                <CheckCircle2 className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  Generated Top-{recommendations.count} Recommendations
                </h3>
                <p className="text-xs text-zinc-400 font-mono">
                  Engine: <span className="text-emerald-400 font-bold">{recommendations.model_used}</span>
                  {recommendations.model_requested !== recommendations.model_used && (
                    <span className="text-amber-400"> (Requested: {recommendations.model_requested})</span>
                  )}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-4 text-xs font-semibold text-zinc-400">
              <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-zinc-800 border border-zinc-700">
                <Clock className="w-3.5 h-3.5 text-emerald-400" />
                Inference: {recommendations.inference_time_ms.toFixed(1)} ms
              </span>
              <span className="px-3 py-1 rounded-full bg-zinc-800 border border-zinc-700">
                User: #{musicUserId.slice(0, 8)}...
              </span>
            </div>
          </div>

          {/* Cold-Start Fallback Warning Banner */}
          {recommendations.fallback_used && (
            <div className="p-4 rounded-xl bg-amber-950/40 border border-amber-600/50 flex items-start gap-3 text-amber-200">
              <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
              <div className="text-xs space-y-1">
                <p className="font-bold text-amber-300">Cold-Start Fallback Active</p>
                <p className="text-zinc-300 leading-relaxed">
                  {recommendations.fallback_reason ||
                    'User has no training interactions recorded. Automatically recommended highest global popularity songs.'}
                </p>
              </div>
            </div>
          )}

          {/* Recommendations Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
            {recommendations.recommendations.map((song) => (
              <SongCard key={song.song_id} song={song} />
            ))}
          </div>
        </div>
      )}

      {/* Error View */}
      {error && (
        <div className="p-6 rounded-2xl bg-rose-950/40 border border-rose-600/50 text-center space-y-3">
          <AlertTriangle className="w-8 h-8 text-rose-400 mx-auto" />
          <p className="text-sm font-bold text-rose-300">{error}</p>
          <button
            onClick={() => fetchRecommendations(musicUserId, selectedModel, topK)}
            className="px-4 py-2 rounded-xl bg-rose-600 hover:bg-rose-500 text-xs font-bold text-white transition-all shadow-md"
          >
            Retry Recommendation
          </button>
        </div>
      )}
    </div>
  );
};
