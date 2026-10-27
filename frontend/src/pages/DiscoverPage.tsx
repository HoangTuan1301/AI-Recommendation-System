import React, { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Search, Filter, ArrowUpDown } from 'lucide-react';
import { api } from '../services/api';
import { MovieSummary } from '../types';
import { MovieCard } from '../components/movies/MovieCard';
import { MovieCardSkeleton } from '../components/ui/LoadingSkeleton';
import { EmptyState } from '../components/ui/EmptyState';

export const DiscoverPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const initialQuery = searchParams.get('q') || '';

  const [movies, setMovies] = useState<MovieSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState(initialQuery);
  const [debouncedQuery, setDebouncedQuery] = useState(initialQuery);
  const [selectedGenre, setSelectedGenre] = useState('All');
  const [sortBy, setSortBy] = useState('popular');
  const [loading, setLoading] = useState(true);

  const genres = [
    'All', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
  ];

  // Debounce search input
  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedQuery(searchQuery);
      setPage(1);
    }, 350);
    return () => clearTimeout(handler);
  }, [searchQuery]);

  // Load movies on filters change
  useEffect(() => {
    const fetchCatalog = async () => {
      setLoading(true);
      try {
        const data = await api.getMovies({
          page,
          limit: 24,
          q: debouncedQuery,
          genre: selectedGenre,
          sort_by: sortBy,
        });
        setMovies(data.items);
        setTotal(data.total);
      } catch (err) {
        console.error('Failed to fetch movies catalog:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchCatalog();
  }, [page, debouncedQuery, selectedGenre, sortBy]);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 pb-16 space-y-8">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl sm:text-4xl font-black text-white tracking-tight">
          Discover Movie Catalog
        </h1>
        <p className="text-sm text-cine-muted mt-1">
          Explore all 1,682 films in MovieLens 100K with live filtering, full-text search, and genre tags.
        </p>
      </div>

      {/* Search & Sort Controls */}
      <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4">
        {/* Search Bar */}
        <div className="relative flex-1">
          <Search className="w-5 h-5 text-cine-muted absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search movie titles (e.g. Star Wars, Toy Story, Pulp Fiction)..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-11 pr-4 py-3 rounded-xl bg-cine-surface border border-cine-border text-white placeholder-cine-muted text-sm focus:outline-none focus:border-cine-red transition-all"
          />
        </div>

        {/* Sort Selector */}
        <div className="flex items-center gap-2">
          <ArrowUpDown className="w-4 h-4 text-cine-muted" />
          <select
            value={sortBy}
            onChange={(e) => {
              setSortBy(e.target.value);
              setPage(1);
            }}
            className="px-4 py-3 rounded-xl bg-cine-surface border border-cine-border text-white text-sm font-semibold focus:outline-none focus:border-cine-red"
          >
            <option value="popular">Most Popular</option>
            <option value="rating">Highest Rated</option>
            <option value="year">Release Year (Newest)</option>
            <option value="title">Title (A - Z)</option>
          </select>
        </div>
      </div>

      {/* Genre Filter Chips */}
      <div className="flex gap-2 overflow-x-auto no-scrollbar pb-1">
        {genres.map((g) => (
          <button
            key={g}
            onClick={() => {
              setSelectedGenre(g);
              setPage(1);
            }}
            className={`px-3.5 py-1.5 rounded-full text-xs font-semibold whitespace-nowrap transition-all ${
              selectedGenre === g
                ? 'bg-cine-red text-white shadow-glow-red'
                : 'bg-cine-surface text-cine-muted hover:text-white hover:bg-cine-hover border border-cine-border'
            }`}
          >
            {g}
          </button>
        ))}
      </div>

      {/* Results Count */}
      <div className="text-xs text-cine-muted font-medium">
        Showing <b>{movies.length}</b> of <b>{total}</b> movies
      </div>

      {/* Movie Grid */}
      {loading ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
          {Array.from({ length: 18 }).map((_, i) => (
            <MovieCardSkeleton key={i} />
          ))}
        </div>
      ) : movies.length === 0 ? (
        <EmptyState
          title="No matching movies found"
          description={`No movies found matching "${debouncedQuery}" in genre "${selectedGenre}".`}
          actionText="Reset Filters"
          onAction={() => {
            setSearchQuery('');
            setSelectedGenre('All');
            setSortBy('popular');
          }}
        />
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
          {movies.map((movie) => (
            <div key={movie.movie_id} className="flex justify-center">
              <MovieCard movie={movie} />
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      {total > 24 && (
        <div className="flex items-center justify-center gap-3 pt-8">
          <button
            onClick={() => setPage((p) => Math.max(p - 1, 1))}
            disabled={page === 1}
            className="px-4 py-2 rounded-lg bg-cine-surface border border-cine-border text-sm font-semibold text-white disabled:opacity-40 disabled:cursor-not-allowed hover:bg-cine-hover transition-colors"
          >
            Previous
          </button>
          <span className="text-xs text-cine-muted font-mono font-bold">
            Page {page} of {Math.ceil(total / 24)}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(p + 1, Math.ceil(total / 24)))}
            disabled={page >= Math.ceil(total / 24)}
            className="px-4 py-2 rounded-lg bg-cine-surface border border-cine-border text-sm font-semibold text-white disabled:opacity-40 disabled:cursor-not-allowed hover:bg-cine-hover transition-colors"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
};
