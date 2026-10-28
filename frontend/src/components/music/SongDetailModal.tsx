import React, { useEffect, useState } from 'react';
import { X, Music, Disc3, Headphones, Play, Layers } from 'lucide-react';
import { useDomain } from '../../context/DomainContext';
import { musicApi } from '../../services/musicApi';
import { SongDetailResponse } from '../../types/music';

export const SongDetailModal: React.FC = () => {
  const { selectedSongId, setSelectedSongId } = useDomain();
  const [song, setSong] = useState<SongDetailResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedSongId) {
      setSong(null);
      return;
    }

    let isMounted = true;
    setLoading(true);
    setError(null);

    musicApi
      .getSongDetail(selectedSongId)
      .then((data) => {
        if (isMounted) {
          setSong(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message || 'Could not load song details.');
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [selectedSongId]);

  if (!selectedSongId) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="song-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fadeIn"
      onClick={() => setSelectedSongId(null)}
    >
      <div
        className="relative w-full max-w-lg bg-zinc-900 border border-zinc-700/80 rounded-2xl shadow-2xl overflow-hidden p-6 text-white"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close Button */}
        <button
          onClick={() => setSelectedSongId(null)}
          className="absolute top-4 right-4 p-2 rounded-full bg-zinc-800 text-zinc-400 hover:text-white hover:bg-zinc-700 transition-colors"
          aria-label="Close song detail modal"
        >
          <X className="w-5 h-5" />
        </button>

        {loading ? (
          <div className="py-16 flex flex-col items-center justify-center space-y-3">
            <Disc3 className="w-10 h-10 text-emerald-400 animate-spin" />
            <p className="text-sm text-zinc-400 font-medium">Fetching song metadata from catalog...</p>
          </div>
        ) : error ? (
          <div className="py-8 text-center space-y-3">
            <p className="text-rose-400 text-sm font-semibold">{error}</p>
            <button
              onClick={() => setSelectedSongId(null)}
              className="px-4 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-xs font-bold"
            >
              Close
            </button>
          </div>
        ) : song ? (
          <div className="space-y-6">
            {/* Header Art & Titles */}
            <div className="flex items-start gap-4">
              <div className="w-20 h-20 rounded-xl bg-gradient-to-br from-emerald-600 via-teal-800 to-zinc-900 flex items-center justify-center shadow-lg border border-emerald-500/30 flex-shrink-0">
                <Music className="w-10 h-10 text-emerald-200" />
              </div>

              <div className="flex-grow pr-6">
                <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                  Track Details
                </span>
                <h2 id="song-modal-title" className="text-xl font-black text-white mt-1 leading-snug">
                  {song.title}
                </h2>
                <p className="text-sm font-medium text-emerald-400/90">{song.artist_name}</p>
              </div>
            </div>

            {/* Identifiers Grid */}
            <div className="grid grid-cols-2 gap-3 p-3 rounded-xl bg-zinc-950/80 border border-zinc-800">
              <div>
                <span className="text-[10px] uppercase font-bold text-zinc-500 tracking-wider">
                  Song ID (MSD)
                </span>
                <p className="font-mono text-xs text-zinc-300 font-semibold break-all">
                  {song.song_id}
                </p>
              </div>

              <div>
                <span className="text-[10px] uppercase font-bold text-zinc-500 tracking-wider">
                  Track ID (Echo Nest)
                </span>
                <p className="font-mono text-xs text-zinc-300 font-semibold break-all">
                  {song.track_id || 'N/A'}
                </p>
              </div>
            </div>

            {/* Listening Statistics */}
            <div>
              <h4 className="text-xs uppercase font-extrabold text-zinc-400 tracking-wider mb-2 flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-emerald-400" />
                Training Dataset Statistics
              </h4>

              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-xl bg-zinc-800/60 border border-zinc-700/60 flex items-center gap-3">
                  <Play className="w-5 h-5 text-emerald-400 fill-emerald-400/30" />
                  <div>
                    <div className="text-lg font-black text-white">
                      {(song.total_play_count ?? 0).toLocaleString()}
                    </div>
                    <div className="text-[10px] uppercase tracking-wider text-zinc-400 font-bold">
                      Total Listens (Train)
                    </div>
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-zinc-800/60 border border-zinc-700/60 flex items-center gap-3">
                  <Headphones className="w-5 h-5 text-teal-400" />
                  <div>
                    <div className="text-lg font-black text-white">
                      {(song.unique_listeners ?? 0).toLocaleString()}
                    </div>
                    <div className="text-[10px] uppercase tracking-wider text-zinc-400 font-bold">
                      Unique Listeners
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Academic Notice */}
            <p className="text-[11px] text-zinc-500 italic bg-zinc-950/50 p-2.5 rounded-lg border border-zinc-800/80">
              Note: Metadata and listening frequency are extracted strictly from the Million Song Dataset
              Taste Profile benchmark dataset (10-core filtered subset).
            </p>
          </div>
        ) : null}
      </div>
    </div>
  );
};
