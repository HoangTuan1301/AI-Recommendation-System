import React, { useState, useEffect } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Search, User, Menu, X, Sparkles, ChevronDown, Music, History } from 'lucide-react';
import { useUser } from '../../context/UserContext';
import { useDomain, SAMPLE_MUSIC_USERS } from '../../context/DomainContext';
import { DomainSwitcher } from './DomainSwitcher';

export const Navbar: React.FC = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const location = useLocation();
  const navigate = useNavigate();

  const { currentUserId, setCurrentUserId } = useUser();
  const { musicUserId, setMusicUserId, setHistoryModalOpen } = useDomain();

  const isMusicDomain = location.pathname.startsWith('/music');

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      if (isMusicDomain) {
        navigate(`/music/recommendations?q=${encodeURIComponent(searchQuery.trim())}`);
      } else {
        navigate(`/discover?q=${encodeURIComponent(searchQuery.trim())}`);
      }
      setSearchOpen(false);
    }
  };

  // CineAI Nav Links
  const movieNavLinks = [
    { label: 'Home', path: '/' },
    { label: 'Discover', path: '/discover' },
    { label: 'AI Recommendations', path: '/recommendations', highlight: true },
    { label: 'AI Lab & Benchmark', path: '/lab' },
    { label: 'User Profile', path: '/profile' },
  ];

  // MusicAI Nav Links
  const musicNavLinks = [
    { label: 'Music Home', path: '/music' },
    { label: 'AI Recommendations', path: '/music/recommendations', highlight: true },
    { label: 'AI Models Leaderboard', path: '/music/models' },
  ];

  const currentNavLinks = isMusicDomain ? musicNavLinks : movieNavLinks;
  const popularMovieUsers = [1, 196, 42, 100, 250, 450, 700, 943];

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 h-[72px] flex items-center ${
        isScrolled
          ? 'bg-cine-bg/95 backdrop-blur-md border-b border-cine-border/80 shadow-2xl'
          : 'bg-gradient-to-b from-black/90 via-black/50 to-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full flex items-center justify-between">
        {/* Left: Brand Logo & Domain Switcher */}
        <div className="flex items-center gap-4 lg:gap-6">
          {/* Logo */}
          <Link
            to={isMusicDomain ? '/music' : '/'}
            className="flex items-center gap-2 group flex-shrink-0"
          >
            {isMusicDomain ? (
              <>
                <span className="text-2xl sm:text-3xl font-black tracking-wider text-emerald-400 group-hover:brightness-110 transition-all">
                  MUSIC<span className="text-white">AI</span>
                </span>
                <span className="hidden sm:inline-block px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-widest bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 rounded">
                  v1.0
                </span>
              </>
            ) : (
              <>
                <span className="text-2xl sm:text-3xl font-black tracking-wider text-cine-red group-hover:brightness-110 transition-all">
                  CINE<span className="text-white">AI</span>
                </span>
                <span className="hidden sm:inline-block px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-widest bg-cine-red/20 text-cine-red border border-cine-red/40 rounded">
                  v2.0
                </span>
              </>
            )}
          </Link>

          {/* Domain Switcher Pill */}
          <div className="hidden sm:block">
            <DomainSwitcher />
          </div>

          {/* Navigation Links */}
          <nav className="hidden lg:flex items-center gap-1">
            {currentNavLinks.map((link) => {
              const isActive = location.pathname === link.path;
              return (
                <Link
                  key={link.path}
                  to={link.path}
                  className={`px-3 py-1.5 rounded-lg text-sm font-semibold transition-all ${
                    isActive
                      ? isMusicDomain
                        ? 'text-white bg-zinc-800 border border-emerald-500/50 shadow-sm'
                        : 'text-white bg-cine-surface border border-cine-border shadow-sm'
                      : 'text-zinc-400 hover:text-white hover:bg-white/5'
                  } ${
                    link.highlight && !isActive
                      ? isMusicDomain
                        ? 'text-emerald-400 hover:text-emerald-300 flex items-center gap-1.5'
                        : 'text-cine-red hover:text-cine-red-hover flex items-center gap-1.5'
                      : ''
                  }`}
                >
                  {link.highlight && (
                    <Sparkles
                      className={`w-3.5 h-3.5 inline ${
                        isMusicDomain ? 'fill-emerald-400 text-emerald-400' : 'fill-cine-red text-cine-red'
                      }`}
                    />
                  )}
                  {link.label}
                </Link>
              );
            })}

            {isMusicDomain && (
              <button
                onClick={() => setHistoryModalOpen(true)}
                className="px-3 py-1.5 rounded-lg text-sm font-semibold text-zinc-400 hover:text-white hover:bg-white/5 transition-all flex items-center gap-1.5"
              >
                <History className="w-3.5 h-3.5 text-emerald-400" />
                History
              </button>
            )}
          </nav>
        </div>

        {/* Right: Search, Quick User Switcher, Profile */}
        <div className="flex items-center gap-3">
          {/* Mobile Domain Switcher */}
          <div className="sm:hidden">
            <DomainSwitcher />
          </div>

          {/* Search Toggle */}
          {searchOpen ? (
            <form onSubmit={handleSearchSubmit} className="relative flex items-center">
              <input
                type="text"
                placeholder={isMusicDomain ? 'Search songs...' : 'Search movies...'}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                autoFocus
                className={`w-36 sm:w-56 px-3.5 py-1.5 pr-8 bg-cine-surface border rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none transition-all shadow-inner ${
                  isMusicDomain ? 'border-emerald-500/50 focus:border-emerald-400' : 'border-cine-border focus:border-cine-red'
                }`}
              />
              <button
                type="button"
                onClick={() => setSearchOpen(false)}
                className="absolute right-2.5 text-zinc-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </form>
          ) : (
            <button
              onClick={() => setSearchOpen(true)}
              className="p-2 rounded-lg text-zinc-400 hover:text-white hover:bg-white/5 transition-colors"
              aria-label="Open search"
            >
              <Search className="w-5 h-5" />
            </button>
          )}

          {/* User Selector Dropdown (Domain-Aware) */}
          {isMusicDomain ? (
            <div className="relative group">
              <button className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-900 border border-emerald-500/40 hover:border-emerald-400 transition-all text-xs font-bold text-white shadow-sm">
                <span className="w-5 h-5 rounded-full bg-emerald-500 text-black flex items-center justify-center text-[10px] font-black">
                  ♫
                </span>
                <span className="hidden sm:inline font-mono">
                  {musicUserId.slice(0, 6)}...
                </span>
                <ChevronDown className="w-3.5 h-3.5 text-zinc-400" />
              </button>

              <div className="absolute right-0 top-full mt-2 w-56 py-2 bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50">
                <div className="px-3 py-1.5 text-[10px] uppercase tracking-wider text-emerald-400 font-extrabold border-b border-zinc-800 mb-1">
                  Music Demo Listeners
                </div>
                <div className="space-y-1 px-2 py-1">
                  {SAMPLE_MUSIC_USERS.map((u) => (
                    <button
                      key={u.id}
                      onClick={() => setMusicUserId(u.id)}
                      className={`w-full text-left px-2.5 py-1.5 rounded text-xs font-bold transition-colors flex items-center justify-between ${
                        musicUserId === u.id
                          ? 'bg-emerald-600 text-white'
                          : 'text-zinc-300 hover:bg-zinc-800 hover:text-white'
                      }`}
                    >
                      <span className="truncate">{u.shortId}</span>
                      <span className="text-[10px] opacity-80">{u.trainCount} songs</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="relative group">
              <button className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-cine-surface border border-cine-border hover:border-zinc-500 transition-all text-xs font-bold text-white shadow-sm">
                <span className="w-5 h-5 rounded-full bg-cine-red text-white flex items-center justify-center text-[10px] font-black">
                  {currentUserId}
                </span>
                <span className="hidden sm:inline">User #{currentUserId}</span>
                <ChevronDown className="w-3.5 h-3.5 text-zinc-400" />
              </button>

              <div className="absolute right-0 top-full mt-2 w-48 py-2 bg-cine-surface border border-cine-border rounded-xl shadow-2xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50">
                <div className="px-3 py-1.5 text-[10px] uppercase tracking-wider text-zinc-400 font-extrabold border-b border-cine-border mb-1">
                  Quick Switch User
                </div>
                <div className="grid grid-cols-4 gap-1 px-2 py-1">
                  {popularMovieUsers.map((uid) => (
                    <button
                      key={uid}
                      onClick={() => setCurrentUserId(uid)}
                      className={`py-1 rounded text-xs font-bold transition-colors ${
                        currentUserId === uid
                          ? 'bg-cine-red text-white'
                          : 'text-zinc-300 hover:bg-cine-hover hover:text-white'
                      }`}
                    >
                      #{uid}
                    </button>
                  ))}
                </div>
                <div className="mt-1 pt-1 border-t border-cine-border px-3 py-1">
                  <Link
                    to="/profile"
                    className="text-xs text-cine-red hover:underline flex items-center gap-1 font-semibold"
                  >
                    <User className="w-3 h-3" />
                    View User #{currentUserId} Profile
                  </Link>
                </div>
              </div>
            </div>
          )}

          {/* Mobile Hamburger Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="lg:hidden p-2 rounded-lg text-zinc-300 hover:text-white hover:bg-white/10"
            aria-label="Toggle navigation"
          >
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer Menu */}
      {mobileMenuOpen && (
        <div className="lg:hidden fixed top-[72px] left-0 right-0 bg-cine-bg border-b border-cine-border/90 px-4 py-4 space-y-2 shadow-2xl z-50">
          <div className="pb-3 border-b border-zinc-800 flex justify-center">
            <DomainSwitcher />
          </div>

          {currentNavLinks.map((link) => {
            const isActive = location.pathname === link.path;
            return (
              <Link
                key={link.path}
                to={link.path}
                onClick={() => setMobileMenuOpen(false)}
                className={`block px-3 py-2 rounded-lg text-sm font-semibold ${
                  isActive
                    ? isMusicDomain
                      ? 'text-white bg-zinc-800 font-bold border border-emerald-500/40'
                      : 'text-white bg-cine-surface font-bold border border-cine-border'
                    : 'text-zinc-300 hover:text-white hover:bg-white/5'
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </div>
      )}
    </header>
  );
};
