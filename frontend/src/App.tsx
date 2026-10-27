import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { UserProvider } from './context/UserContext';
import { Navbar } from './components/layout/Navbar';
import { Footer } from './components/layout/Footer';
import { MovieDetailModal } from './components/movies/MovieDetailModal';

import { HomePage } from './pages/HomePage';
import { DiscoverPage } from './pages/DiscoverPage';
import { RecommendationsPage } from './pages/RecommendationsPage';
import { AILabPage } from './pages/AILabPage';
import { ProfilePage } from './pages/ProfilePage';

export const App: React.FC = () => {
  return (
    <UserProvider>
      <BrowserRouter>
        <div className="min-h-screen bg-cine-bg text-white flex flex-col justify-between selection:bg-cine-red selection:text-white">
          <Navbar />

          <main className="flex-grow">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/discover" element={<DiscoverPage />} />
              <Route path="/recommendations" element={<RecommendationsPage />} />
              <Route path="/lab" element={<AILabPage />} />
              <Route path="/profile" element={<ProfilePage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>

          <Footer />
          <MovieDetailModal />
        </div>
      </BrowserRouter>
    </UserProvider>
  );
};

export default App;
