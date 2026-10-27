import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { UserProvider } from '../context/UserContext';

import { Navbar } from '../components/layout/Navbar';
import { MovieCard } from '../components/movies/MovieCard';
import { MetricsExplainer } from '../components/lab/MetricsExplainer';
import { EmptyState } from '../components/ui/EmptyState';
import { ModelSelector } from '../components/recommendations/ModelSelector';
import { ModelInfo } from '../types';

const mockMovie = {
  movie_id: 1,
  title: 'Toy Story (1995)',
  year: 1995,
  genres: ['Animation', 'Comedy'],
  genres_str: 'Animation|Comedy',
  avg_rating: 4.15,
  rating_count: 452,
  poster_url: 'data:image/svg+xml;utf8,dummy',
};

const mockRecMovie = {
  ...mockMovie,
  rank: 1,
  formatted_score: '0.942 prob',
  score_type: 'Interaction Likelihood',
};

const mockModels: ModelInfo[] = [
  {
    id: 'neural_cf',
    name: 'Neural CF (NeuMF)',
    family: 'Deep Learning',
    score_type: 'Interaction Likelihood',
    score_range: '0.0 - 1.0',
    score_format: '{:.3f} prob',
    description: 'Neural Matrix Factorization combining GMF and MLP.',
    complexity: 'Deep Forward Pass',
  },
  {
    id: 'user_cf',
    name: 'User-Based CF',
    family: 'Neighborhood CF',
    score_type: 'Predicted Rating',
    score_range: '1.0 - 5.0 sao',
    score_format: '{:.2f} ★',
    description: 'Pearson correlation with shrinkage.',
    complexity: 'O(K) inference',
  },
];

describe('CineAI Component Unit Tests', () => {
  it('renders Navbar brand logo CineAI and nav links', () => {
    render(
      <UserProvider>
        <BrowserRouter>
          <Navbar />
        </BrowserRouter>
      </UserProvider>
    );
    expect(screen.getByText('CINE')).toBeInTheDocument();
    expect(screen.getByText('AI')).toBeInTheDocument();
    expect(screen.getByText('Discover')).toBeInTheDocument();
    expect(screen.getByText('AI Recommendations')).toBeInTheDocument();
    expect(screen.getByText('AI Lab & Benchmark')).toBeInTheDocument();
  });

  it('renders MovieCard with title, year, and genre badges', () => {
    render(
      <UserProvider>
        <MovieCard movie={mockMovie} />
      </UserProvider>
    );
    expect(screen.getByText('Toy Story (1995)')).toBeInTheDocument();
    expect(screen.getByText('1995')).toBeInTheDocument();
    expect(screen.getByText('Animation')).toBeInTheDocument();
    expect(screen.getByText('Comedy')).toBeInTheDocument();
    expect(screen.getByText('4.2')).toBeInTheDocument();
  });

  it('renders recommended MovieCard with priority rank and formatted score', () => {
    render(
      <UserProvider>
        <MovieCard movie={mockRecMovie} />
      </UserProvider>
    );
    expect(screen.getByText('#1')).toBeInTheDocument();
    expect(screen.getByText('0.942 prob')).toBeInTheDocument();
    expect(screen.getByText('Interaction Likelihood')).toBeInTheDocument();
  });

  it('renders MetricsExplainer with accurate Hit Rate@10 clarification', () => {
    render(<MetricsExplainer />);
    expect(screen.getByText(/Crucial Scientific Clarification on Hit Rate@10 = 82.03%:/i)).toBeInTheDocument();
    expect(screen.getByText(/Precision@10/i)).toBeInTheDocument();
    expect(screen.getByText(/Recall@10/i)).toBeInTheDocument();
    expect(screen.getByText(/NDCG@10/i)).toBeInTheDocument();
    expect(screen.getByText(/Hit Rate@10 \(HR@10\)/i)).toBeInTheDocument();
  });

  it('renders ModelSelector with 10 models and selects properly', () => {
    const handleSelect = vi.fn();
    render(
      <ModelSelector
        models={mockModels}
        selectedModelId="neural_cf"
        onSelectModel={handleSelect}
      />
    );
    expect(screen.getByText('Neural CF (NeuMF)')).toBeInTheDocument();
    expect(screen.getByText('User-Based CF')).toBeInTheDocument();
  });

  it('renders EmptyState when no results exist', () => {
    render(
      <EmptyState
        title="Custom Empty Title"
        description="Custom empty description"
      />
    );
    expect(screen.getByText('Custom Empty Title')).toBeInTheDocument();
    expect(screen.getByText('Custom empty description')).toBeInTheDocument();
  });
});
