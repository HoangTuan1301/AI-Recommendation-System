import React, { useEffect, useState } from 'react';
import { Sparkles, Info, Clock, CheckCircle2, AlertCircle } from 'lucide-react';
import { api } from '../services/api';
import { ModelInfo, RecommendedMovie, RecommendationResponse } from '../types';
import { ModelSelector } from '../components/recommendations/ModelSelector';
import { RecommendationControls } from '../components/recommendations/RecommendationControls';
import { MovieCard } from '../components/movies/MovieCard';
import { MovieCardSkeleton } from '../components/ui/LoadingSkeleton';
import { useUser } from '../context/UserContext';

export const RecommendationsPage: React.FC = () => {
  const { currentUserId } = useUser();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('neural_cf');
  const [topN, setTopN] = useState<number>(10);
  const [recommendations, setRecommendations] = useState<RecommendedMovie[]>([]);
  const [responseMeta, setResponseMeta] = useState<RecommendationResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch model list
  useEffect(() => {
    api.getModels()
      .then((data) => setModels(data))
      .catch((err) => console.error('Failed to load models list:', err));
  }, []);

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getRecommendations(currentUserId, selectedModelId, topN);
      setRecommendations(data.recommendations);
      setResponseMeta(data);
    } catch (err: any) {
      setError(err?.message || 'Failed to generate recommendations.');
    } finally {
      setLoading(false);
    }
  };

  // Initial recommendation run
  useEffect(() => {
    handleGenerate();
  }, [currentUserId, selectedModelId]);

  const activeModel = models.find((m) => m.id === selectedModelId);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 pb-16 space-y-8">
      {/* Page Header */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <span className="px-2.5 py-1 rounded-md bg-cine-red text-white text-[11px] font-black uppercase tracking-wider">
            AI ENGINE
          </span>
          <span className="text-xs text-cine-muted font-bold">
            Real Model Checkpoint Inference
          </span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-black text-white tracking-tight">
          Personalized AI Movie Recommendations
        </h1>
        <p className="text-sm text-cine-muted mt-1 max-w-3xl">
          Synthesize personalized recommendations for any MovieLens 100K user using your choice of 10 machine learning and deep learning models.
        </p>
      </div>

      {/* Recommendation Controls */}
      <RecommendationControls
        topN={topN}
        setTopN={setTopN}
        onGenerate={handleGenerate}
        loading={loading}
        modelName={activeModel?.name || 'Selected Model'}
      />

      {/* Model Selector Cards */}
      {models.length > 0 && (
        <ModelSelector
          models={models}
          selectedModelId={selectedModelId}
          onSelectModel={(id) => setSelectedModelId(id)}
        />
      )}

      {/* Model Semantics Banner */}
      {activeModel && (
        <div className="p-4 rounded-xl bg-cine-surface border border-cine-border flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-2.5">
            <Info className="w-4 h-4 text-cine-red flex-shrink-0" />
            <span className="text-zinc-300">
              <b>{activeModel.name}:</b> {activeModel.description}
            </span>
          </div>
          <div className="flex items-center gap-3 font-mono text-[11px] text-cine-muted flex-shrink-0">
            <span>Score Scale: <b className="text-emerald-400">{activeModel.score_range}</b></span>
            <span>Complexity: <b className="text-cyan-400">{activeModel.complexity}</b></span>
          </div>
        </div>
      )}

      {/* Error Notice */}
      {error && (
        <div className="p-4 rounded-xl bg-red-950/30 border border-red-500/50 flex items-center gap-3 text-red-200 text-sm">
          <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Results Header */}
      {responseMeta && (
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-cine-border pb-3">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-cine-red" />
            <span>Top {recommendations.length} Recommendations for User #{currentUserId}</span>
          </h2>
          <div className="flex items-center gap-4 text-xs text-cine-muted">
            <span className="flex items-center gap-1 text-emerald-400 font-bold">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>{responseMeta.model_name}</span>
            </span>
            <span className="flex items-center gap-1 font-mono">
              <Clock className="w-3.5 h-3.5" />
              <span>{responseMeta.inference_time_ms} ms</span>
            </span>
          </div>
        </div>
      )}

      {/* Recommendation Grid */}
      {loading ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
          {Array.from({ length: topN }).map((_, i) => (
            <MovieCardSkeleton key={i} />
          ))}
        </div>
      ) : recommendations.length > 0 ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
          {recommendations.map((rec) => (
            <div key={rec.movie_id} className="flex justify-center">
              <MovieCard movie={rec} priorityRank={rec.rank} />
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-12 text-cine-muted">
          No recommendations generated yet. Click "Generate Recommendations".
        </div>
      )}
    </div>
  );
};
