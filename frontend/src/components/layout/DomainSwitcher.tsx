import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Film, Music } from 'lucide-react';
import { useDomain, AppDomain } from '../../context/DomainContext';

export const DomainSwitcher: React.FC = () => {
  const { activeDomain, setActiveDomain } = useDomain();
  const navigate = useNavigate();
  const location = useLocation();

  // Synchronize active domain based on current URL path
  const isMusicRoute = location.pathname.startsWith('/music');
  const currentDomain: AppDomain = isMusicRoute ? 'music' : 'movies';

  const handleDomainChange = (domain: AppDomain) => {
    setActiveDomain(domain);
    if (domain === 'music' && !isMusicRoute) {
      navigate('/music');
    } else if (domain === 'movies' && isMusicRoute) {
      navigate('/');
    }
  };

  return (
    <div
      role="tablist"
      aria-label="Recommendation Domain Switcher"
      className="inline-flex items-center p-1 rounded-full bg-zinc-900/90 border border-zinc-700/80 shadow-lg backdrop-blur-md"
    >
      <button
        role="tab"
        aria-selected={currentDomain === 'movies'}
        onClick={() => handleDomainChange('movies')}
        className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold transition-all duration-300 ${
          currentDomain === 'movies'
            ? 'bg-cine-red text-white shadow-glow-red scale-105'
            : 'text-zinc-400 hover:text-white hover:bg-zinc-800/60'
        }`}
      >
        <Film className="w-3.5 h-3.5" />
        <span>Movies</span>
      </button>

      <button
        role="tab"
        aria-selected={currentDomain === 'music'}
        onClick={() => handleDomainChange('music')}
        className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold transition-all duration-300 ${
          currentDomain === 'music'
            ? 'bg-gradient-to-r from-emerald-500 to-teal-600 text-white shadow-lg shadow-emerald-500/30 scale-105'
            : 'text-zinc-400 hover:text-white hover:bg-zinc-800/60'
        }`}
      >
        <Music className="w-3.5 h-3.5" />
        <span>Music</span>
        <span className="px-1 py-0.2 text-[9px] uppercase tracking-wider bg-emerald-950/80 text-emerald-300 border border-emerald-400/30 rounded-full font-black">
          New
        </span>
      </button>
    </div>
  );
};
