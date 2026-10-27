import React, { useEffect, useState } from 'react';
import { X, Star, Calendar, ExternalLink, Film, Sparkles } from 'lucide-react';
import { useUser } from '../../context/UserContext';
import { api } from '../../services/api';
import { MovieDetail } from '../../types';
import { MovieCard } from './MovieCard';

export const MovieDetailModal: React.FC = () => {
  const { selectedMovieId, setSelectedMovieId } = useUser();
  const [detail, setDetail] = useState<MovieDetail | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (selectedMovieId !== null) {
      setLoading(true);
      api.getMovieDetail(selectedMovieId)
        .then((data) => setDetail(data))
        .catch((err) => console.error("Failed to load movie detail:", err))
        .finally(() => setLoading(false));
    } else {
      setDetail(null);
    }
  }, [selectedMovieId]);

  if (selectedMovieId === null) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/80 backdrop-blur-md animate-fadeIn">
      {/* Click outside to close */}
      <div
        className="absolute inset-0"
        onClick={() => setSelectedMovieId(null)}
      />

      <div className="relative w-full max-w-4xl max-h-[90vh] bg-cine-surface border border-cine-border rounded-2xl overflow-y-auto shadow-2xl z-10 text-white">
        {/* Close Button */}
        <button
          onClick={() => setSelectedMovieId(null)}
          className="absolute top-4 right-4 z-20 w-9 h-9 rounded-full bg-black/60 hover:bg-cine-red text-white flex items-center justify-center transition-all"
        >
          <X className="w-5 h-5" />
        </button>

        {loading || !detail ? (
          <div className="p-16 flex flex-col items-center justify-center text-cine-muted">
            <Film className="w-10 h-10 animate-spin text-cine-red mb-3" />
            <p>Loading movie information...</p>
          </div>
        ) : (
          <div>
            {/* Top Backdrop Header */}
            <div className="relative h-48 sm:h-64 w-full bg-gradient-to-r from-cine-bg via-cine-surface to-cine-bg flex items-end p-6">
              <div className="absolute inset-0 opacity-40 overflow-hidden">
                <img
                  src={detail.poster_url || ''}
                  alt={detail.title}
                  className="w-full h-full object-cover filter blur-sm scale-110"
                />
              </div>
              <div className="absolute inset-0 bg-gradient-to-t from-cine-surface via-cine-surface/70 to-transparent" />

              <div className="relative z-10 flex items-end gap-6">
                <div className="hidden sm:block w-32 aspect-[2/3] rounded-lg overflow-hidden shadow-2xl border border-cine-border bg-cine-bg flex-shrink-0">
                  <img
                    src={detail.poster_url || ''}
                    alt={detail.title}
                    className="w-full h-full object-cover"
                  />
                </div>
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cine-red text-white">
                      ID #{detail.movie_id}
                    </span>
                    {detail.year && (
                      <span className="text-xs text-cine-muted font-semibold">
                        {detail.year}
                      </span>
                    )}
                  </div>
                  <h2 className="text-2xl sm:text-3xl font-black text-white">
                    {detail.title}
                  </h2>
                  <div className="flex flex-wrap gap-1.5 mt-2">
                    {detail.genres.map((g) => (
                      <span
                        key={g}
                        className="px-2 py-0.5 rounded text-xs bg-white/10 text-zinc-200 border border-white/10"
                      >
                        {g}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Modal Body */}
            <div className="p-6 space-y-6">
              {/* Stats Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="p-3 bg-cine-bg rounded-xl border border-cine-border text-center">
                  <div className="text-[11px] font-bold text-cine-muted uppercase">Train Rating Count</div>
                  <div className="text-xl font-bold text-white mt-1">{detail.rating_count} ratings</div>
                </div>
                <div className="p-3 bg-cine-bg rounded-xl border border-cine-border text-center">
                  <div className="text-[11px] font-bold text-cine-muted uppercase">Mean Rating</div>
                  <div className="text-xl font-bold text-amber-400 mt-1 flex items-center justify-center gap-1">
                    <Star className="w-4 h-4 fill-current" />
                    <span>{detail.avg_rating?.toFixed(1) || 'N/A'}</span>
                  </div>
                </div>
                <div className="p-3 bg-cine-bg rounded-xl border border-cine-border text-center">
                  <div className="text-[11px] font-bold text-cine-muted uppercase">Release Date</div>
                  <div className="text-sm font-semibold text-white mt-2">{detail.release_date || 'N/A'}</div>
                </div>
                <div className="p-3 bg-cine-bg rounded-xl border border-cine-border text-center">
                  <div className="text-[11px] font-bold text-cine-muted uppercase">External Reference</div>
                  <div className="mt-2">
                    {detail.imdb_url ? (
                      <a
                        href={detail.imdb_url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-xs font-semibold text-cine-red hover:underline flex items-center justify-center gap-1"
                      >
                        <span>IMDb Link</span>
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    ) : (
                      <span className="text-xs text-cine-muted">No link</span>
                    )}
                  </div>
                </div>
              </div>

              {/* Rating Distribution Breakdown */}
              {detail.rating_distribution && (
                <div>
                  <h4 className="text-sm font-bold text-white uppercase tracking-wider mb-3">
                    Rating Distribution in Training History
                  </h4>
                  <div className="space-y-1.5 bg-cine-bg p-4 rounded-xl border border-cine-border">
                    {[5, 4, 3, 2, 1].map((stars) => {
                      const count = detail.rating_distribution?.[stars] || 0;
                      const total = detail.rating_count || 1;
                      const pct = Math.round((count / total) * 100);
                      return (
                        <div key={stars} className="flex items-center gap-3 text-xs">
                          <span className="w-12 font-semibold text-zinc-300 flex items-center gap-1">
                            {stars} <Star className="w-3 h-3 fill-amber-400 text-amber-400" />
                          </span>
                          <div className="flex-grow h-2.5 bg-cine-hover rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-amber-500 to-amber-300 rounded-full transition-all duration-500"
                              style={{ width: `${pct}%` }}
                            />
                          </div>
                          <span className="w-16 text-right text-cine-muted">
                            {count} ({pct}%)
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Similar Movies Section */}
              {detail.similar_movies && detail.similar_movies.length > 0 && (
                <div>
                  <h4 className="text-sm font-bold text-white uppercase tracking-wider mb-3 flex items-center gap-1.5">
                    <Sparkles className="w-4 h-4 text-cine-red" />
                    <span>Similar Movies in Same Genre</span>
                  </h4>
                  <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
                    {detail.similar_movies.map((sim) => (
                      <div
                        key={sim.movie_id}
                        onClick={() => setSelectedMovieId(sim.movie_id)}
                        className="p-2 rounded-lg bg-cine-bg border border-cine-border hover:border-cine-red cursor-pointer transition-all hover:scale-105"
                      >
                        <div className="aspect-[2/3] rounded overflow-hidden mb-1.5 bg-cine-hover">
                          <img
                            src={sim.poster_url || ''}
                            alt={sim.title}
                            className="w-full h-full object-cover"
                          />
                        </div>
                        <div className="text-[11px] font-bold text-white line-clamp-1">
                          {sim.title}
                        </div>
                        <div className="text-[10px] text-cine-muted">
                          {sim.year || ''}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
