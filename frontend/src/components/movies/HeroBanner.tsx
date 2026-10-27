import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Sparkles, Info, Star, Film, ExternalLink } from 'lucide-react';
import { MovieSummary } from '../../types';
import { useUser } from '../../context/UserContext';

interface HeroBannerProps {
  movie?: MovieSummary;
}

export const HeroBanner: React.FC<HeroBannerProps> = ({ movie }) => {
  const navigate = useNavigate();
  const { setSelectedMovieId } = useUser();

  if (!movie) {
    return (
      <div className="relative h-[65vh] w-full bg-cine-surface flex items-center justify-center">
        <div className="flex items-center gap-3 text-cine-muted animate-pulse">
          <Film className="w-6 h-6 animate-spin text-cine-red" />
          <span className="text-sm font-semibold tracking-wide">Loading Featured Masterpiece...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="relative w-full min-h-[70vh] lg:h-[78vh] flex items-center overflow-hidden bg-cine-bg pt-20 pb-12 lg:py-0">
      {/* Dynamic Cinematic Atmospheric Backdrop with Ambient Glow */}
      <div className="absolute inset-0 pointer-events-none z-0">
        {/* Subtle radial spotlight focused on right poster area */}
        <div className="absolute top-0 right-0 w-3/4 h-full bg-[radial-gradient(ellipse_at_80%_30%,rgba(229,9,20,0.18),transparent_65%)]" />
        <div className="absolute bottom-0 left-0 w-1/2 h-full bg-[radial-gradient(circle_at_20%_80%,rgba(99,102,241,0.12),transparent_60%)]" />
        
        {/* Deep edge vignettes to blend seamlessly into background #08090D */}
        <div className="absolute inset-0 bg-gradient-to-t from-cine-bg via-transparent to-black/60" />
        <div className="absolute inset-0 bg-gradient-to-r from-cine-bg via-cine-bg/60 to-transparent" />
      </div>

      {/* Main Content Grid: 2 Columns on Desktop */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full flex flex-col lg:flex-row items-center justify-between gap-8 lg:gap-12">
        {/* Left Column: Hero Text & Call to Actions */}
        <div className="w-full lg:max-w-2xl space-y-5 text-left">
          {/* Top Badges */}
          <div className="flex flex-wrap items-center gap-2.5">
            <span className="px-3 py-1 rounded-md bg-cine-red text-white text-[11px] font-black uppercase tracking-widest shadow-glow-red">
              FEATURED MASTERPIECE
            </span>
            {movie.year && (
              <span className="px-2.5 py-0.5 rounded bg-cine-elevated text-zinc-300 font-bold text-xs border border-cine-border">
                {movie.year}
              </span>
            )}
            {movie.avg_rating && (
              <div className="flex items-center gap-1.5 px-2.5 py-0.5 rounded bg-amber-500/15 border border-amber-500/30 text-amber-400 font-bold text-xs backdrop-blur-md">
                <Star className="w-3.5 h-3.5 fill-current" />
                <span>{movie.avg_rating.toFixed(1)} / 5.0</span>
                {movie.rating_count && (
                  <span className="text-[10px] text-zinc-400 font-normal">({movie.rating_count} reviews)</span>
                )}
              </div>
            )}
          </div>

          {/* Huge Display Heading */}
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black text-white tracking-tight leading-[1.08] drop-shadow-2xl">
            {movie.title}
          </h1>

          {/* Genres Badges */}
          <div className="flex flex-wrap gap-2 pt-1">
            {movie.genres && movie.genres.map((genre) => (
              <span
                key={genre}
                className="px-3 py-1 rounded-full text-xs font-semibold bg-white/10 hover:bg-white/15 text-zinc-200 border border-white/15 backdrop-blur-md transition-colors"
              >
                {genre}
              </span>
            ))}
          </div>

          {/* Synopsis & AI Pitch */}
          <p className="text-sm sm:text-base text-zinc-300 leading-relaxed max-w-xl drop-shadow line-clamp-3">
            Experience the pinnacle of cinema and artificial intelligence recommendations on MovieLens 100K. Evaluated across 10 modern machine learning algorithms including Deep Neural Matrix Factorization and Weighted Hybrid Ensembles.
          </p>

          {/* Action CTAs */}
          <div className="flex flex-wrap items-center gap-3.5 pt-2">
            <button
              onClick={() => navigate('/recommendations')}
              className="px-7 py-3.5 rounded-xl bg-cine-red hover:bg-cine-red-hover text-white font-bold text-sm flex items-center gap-2.5 transition-all shadow-glow-red hover:scale-105 active:scale-95 cursor-pointer"
            >
              <Sparkles className="w-4 h-4 fill-white" />
              <span>Explore AI Recommendations</span>
            </button>

            <button
              onClick={() => setSelectedMovieId(movie.movie_id)}
              className="px-6 py-3.5 rounded-xl bg-cine-elevated hover:bg-cine-hover text-white font-semibold text-sm backdrop-blur-md flex items-center gap-2 transition-all border border-cine-border hover:border-zinc-500 hover:scale-105 active:scale-95 cursor-pointer"
            >
              <Info className="w-4 h-4 text-zinc-300" />
              <span>Movie Details & Stats</span>
            </button>
          </div>
        </div>

        {/* Right Column: Hero Floating Cinematic Poster Showcase */}
        <div className="w-full lg:w-auto flex justify-center lg:justify-end">
          <div
            onClick={() => setSelectedMovieId(movie.movie_id)}
            className="group relative cursor-pointer"
          >
            {/* Ambient Red Glow Backdrop behind Poster */}
            <div className="absolute -inset-4 bg-gradient-to-r from-cine-red/30 to-purple-600/30 rounded-2xl blur-2xl opacity-70 group-hover:opacity-100 transition-opacity duration-500" />

            {/* Poster Card */}
            <div className="relative w-56 sm:w-64 md:w-72 aspect-[2/3] rounded-2xl overflow-hidden shadow-2xl border-2 border-white/10 group-hover:border-cine-red/80 transition-all duration-500 group-hover:scale-105 group-hover:-rotate-1">
              <img
                src={movie.poster_url || undefined}
                alt={movie.title}
                className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
              />

              {/* Top Film Badge */}
              <div className="absolute top-3 left-3 px-2.5 py-1 rounded-md bg-black/75 backdrop-blur-md border border-white/20 text-[10px] font-extrabold uppercase tracking-widest text-white shadow-lg">
                CINEAI TOP PICK
              </div>

              {/* Hover Quick Action */}
              <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/30 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col justify-end p-4">
                <span className="text-xs font-bold text-white mb-1">{movie.title}</span>
                <span className="text-[11px] text-cine-red font-semibold flex items-center gap-1">
                  Click to inspect model scores & details &rarr;
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
