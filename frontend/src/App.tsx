import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { UserProvider } from './context/UserContext';
import { DomainProvider } from './context/DomainContext';
import { Navbar } from './components/layout/Navbar';
import { Footer } from './components/layout/Footer';
import { MovieDetailModal } from './components/movies/MovieDetailModal';
import { SongDetailModal } from './components/music/SongDetailModal';
import { MusicUserHistoryModal } from './components/music/MusicUserHistoryModal';

// CineAI Pages
import { HomePage } from './pages/HomePage';
import { DiscoverPage } from './pages/DiscoverPage';
import { RecommendationsPage } from './pages/RecommendationsPage';
import { AILabPage } from './pages/AILabPage';
import { ProfilePage } from './pages/ProfilePage';

// MusicAI Pages
import { MusicHomePage } from './pages/MusicHomePage';
import { MusicRecommendationsPage } from './pages/MusicRecommendationsPage';
import { MusicModelsPage } from './pages/MusicModelsPage';

export const App: React.FC = () => {
  return (
    <UserProvider>
      <DomainProvider>
        <BrowserRouter>
          <div className="min-h-screen bg-cine-bg text-white flex flex-col justify-between selection:bg-cine-red selection:text-white">
            <Navbar />

            <main className="flex-grow">
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
