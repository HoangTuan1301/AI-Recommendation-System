import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { UserProvider } from './context/UserContext';
import { DomainProvider } from './context/DomainContext';
import { Navbar } from './components/layout/Navbar';
import { Footer } from './components/layout/Footer';
import { MovieDetailModal } from './components/movies/MovieDetailModal';
import { SongDetailModal } from './components/music/SongDetailModal';
import { MusicUserHistoryModal } from './components/music/MusicUserHistoryModal';

// CineAI Pages (Lazy Loaded for route-based code splitting)
const HomePage = lazy(() => import('./pages/HomePage').then((m) => ({ default: m.HomePage })));
const DiscoverPage = lazy(() => import('./pages/DiscoverPage').then((m) => ({ default: m.DiscoverPage })));
const RecommendationsPage = lazy(() => import('./pages/RecommendationsPage').then((m) => ({ default: m.RecommendationsPage })));
const AILabPage = lazy(() => import('./pages/AILabPage').then((m) => ({ default: m.AILabPage })));
const ProfilePage = lazy(() => import('./pages/ProfilePage').then((m) => ({ default: m.ProfilePage })));

// MusicAI Pages (Lazy Loaded)
const MusicHomePage = lazy(() => import('./pages/MusicHomePage').then((m) => ({ default: m.MusicHomePage })));
const MusicRecommendationsPage = lazy(() => import('./pages/MusicRecommendationsPage').then((m) => ({ default: m.MusicRecommendationsPage })));
const MusicModelsPage = lazy(() => import('./pages/MusicModelsPage').then((m) => ({ default: m.MusicModelsPage })));

const PageLoadingFallback: React.FC = () => (
  <div className="flex items-center justify-center min-h-[60vh]">
    <div className="flex flex-col items-center gap-3">
      <div className="w-10 h-10 border-4 border-zinc-800 border-t-emerald-500 rounded-full animate-spin" />
      <span className="text-xs text-zinc-400 font-medium tracking-wide">Loading application...</span>
    </div>
  </div>
);

export const App: React.FC = () => {
  return (
    <UserProvider>
      <DomainProvider>
        <BrowserRouter>
          <div className="min-h-screen bg-cine-bg text-white flex flex-col justify-between selection:bg-cine-red selection:text-white">
            <Navbar />

            <main className="flex-grow">
              <Suspense fallback={<PageLoadingFallback />}>
                <Routes>
                  {/* CineAI Movie Domain Routes */}
                  <Route path="/" element={<HomePage />} />
                  <Route path="/discover" element={<DiscoverPage />} />
                  <Route path="/recommendations" element={<RecommendationsPage />} />
                  <Route path="/lab" element={<AILabPage />} />
                  <Route path="/profile" element={<ProfilePage />} />

                  {/* MusicAI Song Domain Routes */}
                  <Route path="/music" element={<MusicHomePage />} />
                  <Route path="/music/recommendations" element={<MusicRecommendationsPage />} />
                  <Route path="/music/models" element={<MusicModelsPage />} />

                  {/* Fallback */}
                  <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
              </Suspense>
            </main>

            <Footer />
            <MovieDetailModal />
            <SongDetailModal />
            <MusicUserHistoryModal />
          </div>
        </BrowserRouter>
      </DomainProvider>
    </UserProvider>
  );
};

export default App;
