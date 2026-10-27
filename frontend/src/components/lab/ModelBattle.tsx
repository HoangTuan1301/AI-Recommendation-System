import React, { useState } from 'react';
import { Swords, Sparkles, Star } from 'lucide-react';
import { ModelInfo, CompareResponse } from '../../types';
import { api } from '../../services/api';
import { useUser } from '../../context/UserContext';

interface ModelBattleProps {
  models: ModelInfo[];
}

export const ModelBattle: React.FC<ModelBattleProps> = ({ models }) => {
  const { currentUserId } = useUser();
  const [modelA, setModelA] = useState<string>('neural_cf');
  const [modelB, setModelB] = useState<string>('user_cf');
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<CompareResponse | null>(null);

  const handleBattle = async () => {
    setLoading(true);
    try {
      const data = await api.compareModels(currentUserId, modelA, modelB, 10);
      setResult(data);
    } catch (err) {
      console.error('Battle comparison failed:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-cine-surface border border-cine-border rounded-2xl p-6 shadow-xl space-y-6">
      <div className="flex items-center gap-2 border-b border-cine-border pb-4">
        <Swords className="w-5 h-5 text-cine-red" />
        <div>
          <h3 className="text-base font-bold text-white">
            1v1 Model Battle Comparison
          </h3>
          <p className="text-xs text-cine-muted">
            Direct head-to-head comparison between any two models for Target User #{currentUserId}.
          </p>
        </div>
      </div>

      {/* Selectors Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
        <div className="space-y-1.5">
          <label className="text-xs font-bold text-cine-muted uppercase">Model A (Champion)</label>
          <select
            value={modelA}
            onChange={(e) => setModelA(e.target.value)}
            className="w-full px-3 py-2.5 rounded-xl bg-cine-bg border border-cine-border text-white text-xs font-semibold focus:outline-none focus:border-cine-red"
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} ({m.family})
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-1.5">
          <label className="text-xs font-bold text-cine-muted uppercase">Model B (Challenger)</label>
          <select
            value={modelB}
            onChange={(e) => setModelB(e.target.value)}
            className="w-full px-3 py-2.5 rounded-xl bg-cine-bg border border-cine-border text-white text-xs font-semibold focus:outline-none focus:border-cine-red"
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} ({m.family})
              </option>
            ))}
          </select>
        </div>

        <button
          onClick={handleBattle}
          disabled={loading || modelA === modelB}
          className="w-full py-2.5 rounded-xl bg-cine-red hover:bg-cine-red-hover text-white text-xs font-bold transition-all shadow-glow-red disabled:opacity-50"
        >
          {loading ? 'Simulating Battle...' : 'Compare Recommendations'}
        </button>
      </div>

      {/* Battle Results */}
      {result && (
        <div className="pt-4 space-y-6 animate-fadeIn">
          {/* Overlap Summary Card */}
          <div className="p-4 rounded-xl bg-gradient-to-r from-cine-bg via-cine-surface to-cine-bg border border-cine-border flex flex-col sm:flex-row items-center justify-between gap-4">
            <div>
              <span className="text-xs text-cine-muted font-bold uppercase block">
                Jaccard Similarity Index
              </span>
              <span className="text-2xl font-black text-cyan-400 font-mono">
                {(result.jaccard_similarity * 100).toFixed(1)}%
              </span>
            </div>

            <div className="text-center sm:text-right">
              <span className="text-xs text-cine-muted font-bold uppercase block">
                Shared Recommendations
              </span>
              <span className="text-xl font-bold text-emerald-400">
                {result.overlap_count} / {result.top_n} movies overlap
              </span>
            </div>
          </div>

          {/* Side-by-side Top-10 lists */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* List A */}
            <div className="p-4 bg-cine-bg rounded-xl border border-cine-border">
              <h4 className="text-sm font-bold text-white mb-3 flex items-center justify-between">
                <span>{result.model_a_name}</span>
                <span className="text-xs text-cine-muted">Top 10</span>
              </h4>
              <div className="space-y-2">
                {result.recommendations_a.map((rec) => {
                  const isShared = result.shared_movie_ids.includes(rec.movie_id);
                  return (
                    <div
                      key={rec.movie_id}
                      className={`p-2.5 rounded-lg border text-xs flex items-center justify-between transition-colors ${
                        isShared
                          ? 'bg-emerald-950/30 border-emerald-500/50 text-emerald-200'
                          : 'bg-cine-surface/60 border-cine-border text-white'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span className="w-5 text-cine-muted font-mono font-bold">
                          #{rec.rank}
                        </span>
                        <div>
                          <div className="font-semibold line-clamp-1">{rec.title}</div>
                          <div className="text-[10px] text-cine-muted">{rec.genres.slice(0, 2).join(', ')}</div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {isShared && (
                          <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                            Shared
                          </span>
                        )}
                        <span className="font-mono text-zinc-400">{rec.formatted_score}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* List B */}
            <div className="p-4 bg-cine-bg rounded-xl border border-cine-border">
              <h4 className="text-sm font-bold text-white mb-3 flex items-center justify-between">
                <span>{result.model_b_name}</span>
                <span className="text-xs text-cine-muted">Top 10</span>
              </h4>
              <div className="space-y-2">
                {result.recommendations_b.map((rec) => {
                  const isShared = result.shared_movie_ids.includes(rec.movie_id);
                  return (
                    <div
                      key={rec.movie_id}
                      className={`p-2.5 rounded-lg border text-xs flex items-center justify-between transition-colors ${
                        isShared
                          ? 'bg-emerald-950/30 border-emerald-500/50 text-emerald-200'
                          : 'bg-cine-surface/60 border-cine-border text-white'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span className="w-5 text-cine-muted font-mono font-bold">
                          #{rec.rank}
                        </span>
                        <div>
                          <div className="font-semibold line-clamp-1">{rec.title}</div>
                          <div className="text-[10px] text-cine-muted">{rec.genres.slice(0, 2).join(', ')}</div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {isShared && (
                          <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                            Shared
                          </span>
                        )}
                        <span className="font-mono text-zinc-400">{rec.formatted_score}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
