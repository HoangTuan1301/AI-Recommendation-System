import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { DomainProvider, useDomain } from '../context/DomainContext';
import { DomainSwitcher } from '../components/layout/DomainSwitcher';
import { SongCard } from '../components/music/SongCard';
import { MusicModelSelector } from '../components/music/MusicModelSelector';
import { SongDetailModal } from '../components/music/SongDetailModal';
import { MusicUserHistoryModal } from '../components/music/MusicUserHistoryModal';
import { MusicModelsPage } from '../pages/MusicModelsPage';
import { MusicModelInfo, RecommendedSong } from '../types/music';

const mockSong: RecommendedSong = {
  song_id: 'SOFRQTD12A81C233C0',
  track_id: 'TRWKWJY128F427CF18',
  title: 'Sehr kosmisch',
  artist_name: 'Harmonia',
  score: 0.942,
  rank: 1,
};

const mockModels: MusicModelInfo[] = [
  {
    id: 'item_cf',
    name: 'Item-Based CF',
    family: 'Neighborhood CF',
    description: 'Cosine similarity with Top-50 neighbors.',
    is_personalized: true,
    is_default: true,
    is_available: true,
    is_loaded: true,
    checkpoint_file: 'item_cf.pkl',
    latency_profile: '9.3 ms',
  },
  {
    id: 'user_cf',
    name: 'User-Based CF',
    family: 'Neighborhood CF',
    description: 'User similarity based on listening vectors.',
    is_personalized: true,
    is_default: false,
    is_available: true,
    is_loaded: true,
    checkpoint_file: 'user_cf.pkl',
    latency_profile: '1.3 ms',
  },
  {
    id: 'ials',
    name: 'Implicit ALS',
    family: 'Matrix Factorization',
    description: 'Hu-Koren-Volinsky implicit alternating least squares.',
    is_personalized: true,
    is_default: false,
    is_available: true,
    is_loaded: true,
    checkpoint_file: 'ials.pkl',
    latency_profile: '4.9 ms',
  },
  {
    id: 'bpr_mf',
    name: 'BPR-MF',
    family: 'Pairwise Ranking',
    description: 'Bayesian Personalized Ranking Matrix Factorization.',
    is_personalized: true,
    is_default: false,
    is_available: true,
    is_loaded: true,
    checkpoint_file: 'bpr_mf.pkl',
    latency_profile: '5.1 ms',
  },
  {
    id: 'neumf',
    name: 'Neural CF (NeuMF)',
    family: 'Deep Learning',
    description: 'Combines Generalized Matrix Factorization and MLP.',
    is_personalized: true,
    is_default: false,
    is_available: true,
    is_loaded: true,
    checkpoint_file: 'neumf.pt',
    latency_profile: '28.4 ms',
  },
  {
    id: 'popularity',
    name: 'Popularity Baseline',
    family: 'Baseline',
    description: 'Frequency and listener count ranking.',
    is_personalized: false,
    is_default: false,
    is_available: true,
    is_loaded: true,
    checkpoint_file: 'popularity.pkl',
    latency_profile: '0.8 ms',
  },
];

describe('MusicAI Component Unit Tests', () => {
  it('renders DomainSwitcher with Movies and Music options', () => {
    render(
      <DomainProvider>
        <BrowserRouter>
          <DomainSwitcher />
        </BrowserRouter>
      </DomainProvider>
    );

    expect(screen.getByText('Movies')).toBeInTheDocument();
    expect(screen.getByText('Music')).toBeInTheDocument();
  });

  it('switches domain when clicking Music in DomainSwitcher', () => {
    const TestConsumer = () => {
      const { activeDomain } = useDomain();
      return (
        <div>
          <DomainSwitcher />
          <div data-testid="active-domain">{activeDomain}</div>
        </div>
      );
    };

    render(
      <DomainProvider>
        <BrowserRouter>
          <TestConsumer />
        </BrowserRouter>
      </DomainProvider>
    );

    const musicButton = screen.getByText('Music');
    fireEvent.click(musicButton);
    expect(screen.getByTestId('active-domain').textContent).toBe('music');
  });

  it('renders SongCard with song title, artist, rank badge, and score', () => {
    render(
      <DomainProvider>
        <SongCard song={mockSong} />
      </DomainProvider>
    );

    expect(screen.getByText('Sehr kosmisch')).toBeInTheDocument();
    expect(screen.getByText('Harmonia')).toBeInTheDocument();
    expect(screen.getByText('#1')).toBeInTheDocument();
    expect(screen.getByText('0.94')).toBeInTheDocument();
    expect(screen.getByText('SOFRQTD12A81C233C0')).toBeInTheDocument();
  });

  it('renders MusicModelSelector with all 6 models and triggers selection', () => {
    const handleSelect = vi.fn();
    render(
      <MusicModelSelector
        models={mockModels}
        selectedModelId="item_cf"
        onSelectModel={handleSelect}
      />
    );

    expect(screen.getByText('Item-Based CF')).toBeInTheDocument();
    expect(screen.getByText('User-Based CF')).toBeInTheDocument();
    expect(screen.getByText('Implicit ALS')).toBeInTheDocument();
    expect(screen.getByText('BPR-MF')).toBeInTheDocument();
    expect(screen.getByText('Neural CF (NeuMF)')).toBeInTheDocument();
    expect(screen.getByText('Popularity Baseline')).toBeInTheDocument();

    const userCfCard = screen.getByText('User-Based CF');
    fireEvent.click(userCfCard);
    expect(handleSelect).toHaveBeenCalledWith('user_cf');
  });

  it('renders MusicModelsPage with scientific benchmark leaderboard', () => {
    render(
      <BrowserRouter>
        <MusicModelsPage />
      </BrowserRouter>
    );

    expect(screen.getByText('MusicAI Model Leaderboard')).toBeInTheDocument();
    expect(screen.getByText('Item-Based CF')).toBeInTheDocument();
    expect(screen.getByText('0.2448')).toBeInTheDocument(); // HR@10
    expect(screen.getByText('0.1085')).toBeInTheDocument(); // NDCG@10
    expect(screen.getByText('93.20%')).toBeInTheDocument(); // Catalog Coverage
  });
});
