import React, { useEffect, useState } from 'react';
import { User, Star, Film, Award, Heart, BarChart3, Swords } from 'lucide-react';
import { api } from '../services/api';
import { UserProfileResponse, ModelInfo, CompareResponse } from '../types';
import { useUser } from '../context/UserContext';
import { MovieCard } from '../components/movies/MovieCard';
import { TableSkeleton } from '../components/ui/LoadingSkeleton';

export const ProfilePage: React.FC = () => {
  const { currentUserId, setCurrentUserId } = useUser();
  const [profile, setProfile] = useState<UserProfileResponse | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);

  // Model comparison states
  const [compareModelA, setCompareModelA] = useState('neural_cf');
  const [compareModelB, setCompareModelB] = useState('hybrid');
  const [battleData, setBattleData] = useState<CompareResponse | null>(null);
  const [battleLoading, setBattleLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      api.getUserProfile(currentUserId),
      api.getModels(),
    ])
      .then(([profData, modelsData]) => {
        setProfile(profData);
        setModels(modelsData);
      })
      .catch((err) => console.error('Failed to load profile:', err))
      .finally(() => setLoading(false));
  }, [currentUserId]);

  const handleRunBattle = async () => {
    setBattleLoading(true);
    try {
      const data = await api.compareModels(currentUserId, compareModelA, compareModelB, 10);
      setBattleData(data);
    } catch (err) {
      console.error('Failed to compare models:', err);
    } finally {
      setBattleLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 pb-16 space-y-8">
      {/* User Header */}
      <div className="p-6 sm:p-8 rounded-2xl bg-gradient-to-r from-cine-surface via-cine-hover to-cine-surface border border-cine-border shadow-xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6">
        <div className="flex items-center gap-5">
          <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-2xl bg-gradient-to-br from-cine-red to-purple-600 flex items-center justify-center text-2xl sm:text-3xl font-black text-white shadow-glow-red">
            #{currentUserId}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-white/10 text-white uppercase tracking-wider">
                Active CineAI Viewer
              </span>
              <span className="text-xs text-cine-muted font-semibold">MovieLens 100K</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-black text-white mt-1">
              Viewer #{currentUserId} Profile
            </h1>
            <p className="text-xs text-cine-muted mt-0.5">
              Taste profile, viewing history, and ratings synthesized strictly from training interactions.
            </p>
          </div>
        </div>

        {/* Quick User Switcher */}
        <div className="flex items-center gap-2">
          <label className="text-xs text-cine-muted font-bold uppercase">Switch User:</label>
          <select
            value={currentUserId}
            onChange={(e) => setCurrentUserId(parseInt(e.target.value, 10))}
            className="px-3 py-2 rounded-xl bg-cine-bg border border-cine-border text-white text-xs font-bold focus:outline-none focus:border-cine-red"
          >
            {[1, 196, 42, 100, 250, 450, 700, 943].map((uid) => (
              <option key={uid} value={uid}>
                User #{uid}
              </option>
            ))}
          </select>
        </div>
      </div>

      {loading || !profile ? (
        <TableSkeleton />
      ) : (
        <div className="space-y-8">
          {/* KPI Statistics */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="p-5 rounded-xl bg-cine-surface border border-cine-border text-center">
              <span className="text-xs text-cine-muted uppercase font-bold block mb-1">
                Total Movies Rated (Train)
              </span>
              <span className="text-3xl font-black text-white font-mono">
                {profile.ratings_count}
              </span>
              <span className="text-xs text-zinc-500 block mt-1">Verified interactions</span>
            </div>

            <div className="p-5 rounded-xl bg-cine-surface border border-cine-border text-center">
              <span className="text-xs text-cine-muted uppercase font-bold block mb-1">
                Average Rating Given
              </span>
              <span className="text-3xl font-black text-amber-400 font-mono flex items-center justify-center gap-1.5">
                <Star className="w-6 h-6 fill-current" />
                <span>{profile.avg_rating.toFixed(2)}</span>
              </span>
              <span className="text-xs text-zinc-500 block mt-1">Scale: 1.0 to 5.0 stars</span>
            </div>

            <div className="p-5 rounded-xl bg-cine-surface border border-cine-border text-center">
              <span className="text-xs text-cine-muted uppercase font-bold block mb-1">
                Loved Movies (Rating ≥ 4.0)
              </span>
              <span className="text-3xl font-black text-emerald-400 font-mono">
                {profile.high_rated_count}
              </span>
              <span className="text-xs text-zinc-500 block mt-1">Positive affinity interactions</span>
            </div>
          </div>

          {/* Rating Breakdown & Favorite Genres */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Rating distribution histogram */}
            <div className="p-6 rounded-2xl bg-cine-surface border border-cine-border space-y-4">
              <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-cine-red" />
                <span>Rating Distribution in History</span>
              </h3>
              <div className="space-y-2 bg-cine-bg p-4 rounded-xl border border-cine-border">
                {[5, 4, 3, 2, 1].map((stars) => {
                  const count = profile.rating_distribution[stars] || 0;
                  const pct = profile.ratings_count > 0 ? Math.round((count / profile.ratings_count) * 100) : 0;
                  return (
                    <div key={stars} className="flex items-center gap-3 text-xs">
                      <span className="w-12 font-bold text-zinc-300 flex items-center gap-1">
                        {stars} <Star className="w-3 h-3 fill-amber-400 text-amber-400" />
                      </span>
                      <div className="flex-grow h-3 bg-cine-hover rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-cine-red to-amber-500 rounded-full transition-all duration-500"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="w-14 text-right text-cine-muted font-mono">
                        {count} ({pct}%)
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Favorite Genres Affinities */}
            <div className="p-6 rounded-2xl bg-cine-surface border border-cine-border space-y-4">
              <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Heart className="w-4 h-4 text-cine-red fill-current" />
                <span>Top Genre Affinities</span>
              </h3>
              <div className="space-y-2 bg-cine-bg p-4 rounded-xl border border-cine-border max-h-56 overflow-y-auto">
                {profile.favorite_genres.map((g) => (
                  <div key={g.genre} className="flex items-center justify-between p-2 rounded-lg bg-cine-surface text-xs">
                    <span className="font-semibold text-white">{g.genre}</span>
                    <div className="flex items-center gap-3">
                      <span className="text-cine-muted">{g.count} titles</span>
                      <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-cine-red/20 text-cine-red border border-cine-red/30">
                        {g.percentage}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Top Rated Movies from Train History */}
          <div className="space-y-4">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Award className="w-5 h-5 text-amber-400" />
              <span>Highest Rated Movies from User's Training History (4.0 & 5.0 Stars)</span>
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
              {profile.top_rated_movies.map((m) => (
                <MovieCard key={m.movie_id} movie={m} />
              ))}
            </div>
          </div>

          {/* Compare Recommendations for this User */}
          <div className="p-6 rounded-2xl bg-cine-surface border border-cine-border space-y-6">
            <div className="flex items-center gap-2 border-b border-cine-border pb-3">
              <Swords className="w-5 h-5 text-cine-red" />
              <h3 className="text-base font-bold text-white">
                Compare Top-10 Recommendations for User #{currentUserId}
              </h3>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 items-end">
              <div>
                <label className="text-xs font-bold text-cine-muted uppercase block mb-1">Model A</label>
                <select
                  value={compareModelA}
                  onChange={(e) => setCompareModelA(e.target.value)}
                  className="w-full px-3 py-2 rounded-xl bg-cine-bg border border-cine-border text-white text-xs font-semibold focus:outline-none focus:border-cine-red"
                >
                  {models.map((m) => (
                    <option key={m.id} value={m.id}>{m.name}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="text-xs font-bold text-cine-muted uppercase block mb-1">Model B</label>
                <select
                  value={compareModelB}
                  onChange={(e) => setCompareModelB(e.target.value)}
                  className="w-full px-3 py-2 rounded-xl bg-cine-bg border border-cine-border text-white text-xs font-semibold focus:outline-none focus:border-cine-red"
                >
                  {models.map((m) => (
                    <option key={m.id} value={m.id}>{m.name}</option>
                  ))}
                </select>
              </div>

              <button
                onClick={handleRunBattle}
                disabled={battleLoading || compareModelA === compareModelB}
                className="w-full py-2 rounded-xl bg-cine-red hover:bg-cine-red-hover text-white text-xs font-bold transition-all disabled:opacity-50"
              >
                {battleLoading ? 'Comparing...' : 'Compare Top-10'}
              </button>
            </div>

            {battleData && (
              <div className="pt-2 animate-fadeIn space-y-4">
                <div className="p-3 rounded-lg bg-cine-bg border border-cine-border flex items-center justify-between text-xs">
                  <span>Jaccard Overlap: <b className="text-cyan-400 font-mono">{(battleData.jaccard_similarity * 100).toFixed(1)}%</b></span>
                  <span>Shared Picks: <b className="text-emerald-400">{battleData.overlap_count} / 10 movies</b></span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-3 rounded-lg bg-cine-bg border border-cine-border">
                    <h5 className="text-xs font-bold text-white mb-2">{battleData.model_a_name} Top 10</h5>
                    <div className="space-y-1">
                      {battleData.recommendations_a.map((m) => (
                        <div key={m.movie_id} className="text-xs py-1 px-2 rounded bg-cine-surface flex items-center justify-between">
                          <span className="line-clamp-1">#{m.rank}. {m.title}</span>
                          <span className="font-mono text-[10px] text-zinc-400">{m.formatted_score}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-cine-bg border border-cine-border">
                    <h5 className="text-xs font-bold text-white mb-2">{battleData.model_b_name} Top 10</h5>
                    <div className="space-y-1">
                      {battleData.recommendations_b.map((m) => (
                        <div key={m.movie_id} className="text-xs py-1 px-2 rounded bg-cine-surface flex items-center justify-between">
                          <span className="line-clamp-1">#{m.rank}. {m.title}</span>
                          <span className="font-mono text-[10px] text-zinc-400">{m.formatted_score}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
