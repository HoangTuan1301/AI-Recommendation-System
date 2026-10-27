import React, { useEffect, useState } from 'react';
import { api } from '../services/api';
import { MovieSummary, RecommendedMovie } from '../types';
import { HeroBanner } from '../components/movies/HeroBanner';
import { MovieCarousel } from '../components/movies/MovieCarousel';
import { CarouselSkeleton } from '../components/ui/LoadingSkeleton';
import { useUser } from '../context/UserContext';

export const HomePage: React.FC = () => {
  const { currentUserId } = useUser();
  const [featuredMovie, setFeaturedMovie] = useState<MovieSummary | undefined>(undefined);
  const [userRecs, setUserRecs] = useState<RecommendedMovie[]>([]);
  const [popularMovies, setPopularMovies] = useState<MovieSummary[]>([]);
  const [topRatedMovies, setTopRatedMovies] = useState<MovieSummary[]>([]);
  const [actionMovies, setActionMovies] = useState<MovieSummary[]>([]);
  const [comedyMovies, setComedyMovies] = useState<MovieSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadHomeData = async () => {
      setLoading(true);
      try {
        // Parallel data fetch
        const [popRes, topRes, actRes, comRes, recRes] = await Promise.all([
          api.getMovies({ sort_by: 'popular', limit: 12 }),
          api.getMovies({ sort_by: 'rating', limit: 12 }),
          api.getMovies({ genre: 'Action', limit: 12 }),
          api.getMovies({ genre: 'Comedy', limit: 12 }),
          api.getRecommendations(currentUserId, 'neural_cf', 10).catch(() => null),
        ]);

        setPopularMovies(popRes.items);
        setTopRatedMovies(topRes.items);
        setActionMovies(actRes.items);
        setComedyMovies(comRes.items);

        // Featured classic movie (Star Wars or first popular item)
        const starWars = popRes.items.find((m) => m.title.includes('Star Wars')) || popRes.items[0];
        setFeaturedMovie(starWars);

        if (recRes && recRes.recommendations) {
          setUserRecs(recRes.recommendations);
        }
      } catch (err) {
        console.error('Failed to load home page data:', err);
      } finally {
        setLoading(false);
      }
    };

    loadHomeData();
  }, [currentUserId]);

  return (
    <div className="min-h-screen pb-16">
      {/* Cinematic Hero */}
      <HeroBanner movie={featuredMovie} />

      {/* Movie Carousels */}
      <div className="relative z-20 max-w-7xl mx-auto px-2 sm:px-4 space-y-6 sm:space-y-8 mt-2">
        {/* Recommended for current user */}
        {loading ? (
          <CarouselSkeleton count={6} />
        ) : (
          userRecs.length > 0 && (
            <MovieCarousel
              title={`Top AI Picks for You (User #${currentUserId})`}
              subtitle="Personalized recommendations synthesized by Neural Collaborative Filtering (NeuMF)"
              movies={userRecs}
            />
          )
        )}

        {/* Popular Movies */}
        {loading ? (
          <CarouselSkeleton count={6} />
        ) : (
          <MovieCarousel
            title="Popular on CineAI"
            subtitle="Most evaluated cinematic classics in MovieLens 100K"
            movies={popularMovies}
          />
        )}

        {/* Highest Rated */}
        {loading ? (
          <CarouselSkeleton count={6} />
        ) : (
          <MovieCarousel
            title="Highest Rated Masterpieces"
            subtitle="Top average audience rating from training records"
            movies={topRatedMovies}
          />
        )}

        {/* Action Movies */}
        {loading ? (
          <CarouselSkeleton count={6} />
        ) : (
          <MovieCarousel
            title="Action & Sci-Fi Adventures"
            subtitle="Explosive and futuristic storytelling"
            movies={actionMovies}
          />
        )}

        {/* Comedy Movies */}
        {loading ? (
          <CarouselSkeleton count={6} />
        ) : (
          <MovieCarousel
            title="Comedy & Lighthearted Gems"
            subtitle="Acclaimed humor and feel-good stories"
            movies={comedyMovies}
          />
        )}
      </div>
    </div>
  );
};
