import React, { useRef } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { MovieSummary, RecommendedMovie } from '../../types';
import { MovieCard } from './MovieCard';

interface MovieCarouselProps {
  title: string;
  subtitle?: string;
  movies: (MovieSummary | RecommendedMovie)[];
  priorityRankStart?: number;
}

export const MovieCarousel: React.FC<MovieCarouselProps> = ({
  title,
  subtitle,
  movies,
  priorityRankStart,
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  const scroll = (direction: 'left' | 'right') => {
    if (scrollRef.current) {
      const scrollAmount = 600;
      scrollRef.current.scrollBy({
        left: direction === 'left' ? -scrollAmount : scrollAmount,
        behavior: 'smooth',
      });
    }
  };

  if (!movies || movies.length === 0) return null;

  return (
    <section className="my-8 relative group/carousel">
      {/* Row Header */}
      <div className="flex items-baseline justify-between mb-3 px-4 sm:px-6 lg:px-8">
        <div>
          <h3 className="text-lg sm:text-xl font-bold text-white tracking-wide">
            {title}
          </h3>
          {subtitle && (
            <p className="text-xs text-cine-muted mt-0.5">{subtitle}</p>
          )}
        </div>
        <span className="text-xs text-cine-muted font-medium">
          {movies.length} titles
        </span>
      </div>

      {/* Carousel Container */}
      <div className="relative">
        {/* Scroll Left Button */}
        <button
          onClick={() => scroll('left')}
          className="absolute left-1 top-1/2 -translate-y-1/2 z-30 w-10 h-10 rounded-full bg-black/70 hover:bg-cine-red text-white flex items-center justify-center opacity-0 group-hover/carousel:opacity-100 transition-all duration-200 shadow-xl backdrop-blur-sm"
          aria-label="Scroll left"
        >
          <ChevronLeft className="w-6 h-6" />
        </button>

        {/* Scrollable Track */}
        <div
          ref={scrollRef}
          className="flex gap-4 overflow-x-auto no-scrollbar px-4 sm:px-6 lg:px-8 py-4 scroll-smooth"
        >
          {movies.map((movie, index) => (
            <div key={`${movie.movie_id}-${index}`} className="flex-shrink-0 w-36 sm:w-44 md:w-48">
              <MovieCard
                movie={movie}
                priorityRank={priorityRankStart ? priorityRankStart + index : undefined}
              />
            </div>
          ))}
        </div>

        {/* Scroll Right Button */}
        <button
          onClick={() => scroll('right')}
          className="absolute right-1 top-1/2 -translate-y-1/2 z-30 w-10 h-10 rounded-full bg-black/70 hover:bg-cine-red text-white flex items-center justify-center opacity-0 group-hover/carousel:opacity-100 transition-all duration-200 shadow-xl backdrop-blur-sm"
          aria-label="Scroll right"
        >
          <ChevronRight className="w-6 h-6" />
        </button>
      </div>
    </section>
  );
};
