import React from 'react';
import { Star, Sparkles } from 'lucide-react';
import { MovieSummary, RecommendedMovie } from '../../types';
import { useUser } from '../../context/UserContext';

interface MovieCardProps {
  movie: MovieSummary | RecommendedMovie;
  priorityRank?: number;
}

export const MovieCard: React.FC<MovieCardProps> = ({ movie, priorityRank }) => {
  const { setSelectedMovieId } = useUser();
  const recMovie = movie as RecommendedMovie;
  const isRecommended = typeof recMovie.rank === 'number' || typeof priorityRank === 'number';
  const displayRank = recMovie.rank || priorityRank;

  return (
    <div
      onClick={() => setSelectedMovieId(movie.movie_id)}
      className="group relative w-full rounded-xl bg-cine-surface hover:bg-cine-hover border border-cine-border/70 hover:border-cine-red/80 overflow-hidden cursor-pointer transition-all duration-300 hover:scale-105 hover:shadow-card-hover hover:z-20 flex flex-col"
    >
      {/* Poster Image Container (2:3 aspect ratio) */}
      <div className="relative w-full aspect-[2/3] overflow-hidden bg-cine-secondary">
        {movie.poster_url ? (
          <img
            src={movie.poster_url}
            alt={movie.title}
            className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
            loading="lazy"
          />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center p-4 text-center bg-gradient-to-br from-cine-surface to-cine-bg">
            <span className="text-xs font-bold text-white line-clamp-2">{movie.title}</span>
          </div>
        )}

        {/* Top-N Rank Badge */}
        {isRecommended && displayRank && (
          <div className="absolute top-2 left-2 px-2 py-0.5 rounded-md bg-gradient-to-r from-cine-red to-purple-600 text-white font-black text-xs shadow-lg tracking-wider flex items-center gap-1 z-10">
            <Sparkles className="w-3 h-3 fill-white" />
            <span>#{displayRank}</span>
          </div>
        )}

        {/* Top Right Release Year */}
        {movie.year && (
          <div className="absolute top-2 right-2 px-1.5 py-0.5 rounded bg-black/75 backdrop-blur-md text-[10px] font-bold text-zinc-300 border border-white/10 z-10">
            {movie.year}
          </div>
        )}

        {/* Hover Action Overlay */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col justify-end p-3 z-10">
          <button className="w-full py-2 rounded-lg bg-cine-red text-white text-xs font-bold hover:bg-cine-red-hover transition-colors shadow-glow-red">
            View Details
          </button>
        </div>
      </div>

      {/* Card Body Metadata Section */}
      <div className="p-3.5 flex flex-col justify-between flex-grow gap-2">
        <div>
          {/* Movie Title */}
          <h4
            title={movie.title}
            className="text-xs sm:text-sm font-bold text-white line-clamp-2 leading-snug group-hover:text-cine-red transition-colors"
          >
            {movie.title}
          </h4>

          {/* Genre Badges */}
          <div className="flex flex-wrap gap-1 mt-1.5">
            {movie.genres && movie.genres.slice(0, 2).map((g) => (
              <span
                key={g}
                className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-cine-elevated border border-cine-border text-cine-muted"
              >
                {g}
              </span>
            ))}
          </div>
        </div>

        {/* Score or Rating Footer */}
        <div className="pt-2 border-t border-cine-border/60 flex items-center justify-between text-xs">
          {recMovie.formatted_score ? (
            <div className="flex items-center justify-between w-full">
              <span className="text-[10px] text-cine-muted uppercase font-semibold truncate max-w-[90px]">
                {recMovie.score_type || 'AI Score'}
              </span>
              <span className="font-extrabold text-emerald-400 text-xs">
                {recMovie.formatted_score}
              </span>
            </div>
          ) : movie.avg_rating && movie.avg_rating > 0 ? (
            <div className="flex items-center justify-between w-full">
              <div className="flex items-center gap-1 text-amber-400 font-bold text-xs">
                <Star className="w-3.5 h-3.5 fill-current" />
                <span>{movie.avg_rating.toFixed(1)}</span>
              </div>
              <span className="text-[10px] text-cine-muted">
                {movie.rating_count} reviews
              </span>
            </div>
          ) : (
            <span className="text-[10px] text-zinc-500">Unrated</span>
          )}
        </div>
      </div>
    </div>
  );
};
