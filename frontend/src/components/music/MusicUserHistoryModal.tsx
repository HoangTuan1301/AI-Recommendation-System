import React, { useEffect, useState } from 'react';
import { X, History, Play, Music, Disc3, AlertCircle } from 'lucide-react';
import { useDomain } from '../../context/DomainContext';
import { musicApi } from '../../services/musicApi';
import { UserHistoryResponse } from '../../types/music';

export const MusicUserHistoryModal: React.FC = () => {
  const { historyModalOpen, setHistoryModalOpen, musicUserId, setSelectedSongId } = useDomain();
  const [history, setHistory] = useState<UserHistoryResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!historyModalOpen || !musicUserId) {
      setHistory(null);
      return;
    }

    let isMounted = true;
    setLoading(true);
    setError(null);

    musicApi
      .getUserHistory(musicUserId)
      .then((data) => {
        if (isMounted) {
          setHistory(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message || 'Could not load listening history for this user.');
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [historyModalOpen, musicUserId]);

  if (!historyModalOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="history-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-md animate-fadeIn"
      onClick={() => setHistoryModalOpen(false)}
    >
      <div
        className="relative w-full max-w-2xl max-h-[85vh] bg-zinc-900 border border-zinc-700/80 rounded-2xl shadow-2xl flex flex-col text-white overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-5 border-b border-zinc-800 flex items-center justify-between flex-shrink-0">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
              <History className="w-6 h-6" />
            </div>
            <div>
              <h2 id="history-modal-title" className="text-lg font-black text-white">
                Training Listening History
              </h2>
              <p className="text-xs text-zinc-400 font-mono">User: {musicUserId}</p>
            </div>
          </div>

          <button
            onClick={() => setHistoryModalOpen(false)}
            className="p-2 rounded-full bg-zinc-800 text-zinc-400 hover:text-white hover:bg-zinc-700 transition-colors"
            aria-label="Close history modal"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-5 overflow-y-auto flex-grow space-y-4">
          {loading ? (
            <div className="py-16 flex flex-col items-center justify-center space-y-3">
              <Disc3 className="w-10 h-10 text-emerald-400 animate-spin" />
              <p className="text-sm text-zinc-400 font-medium">Fetching listening history from Train split...</p>
            </div>
          ) : error ? (
            <div className="py-12 text-center space-y-3">
              <AlertCircle className="w-10 h-10 text-amber-400 mx-auto" />
              <p className="text-sm font-semibold text-zinc-300">{error}</p>
              <p className="text-xs text-zinc-500">
                This user might be a new cold-start guest with no prior listening interactions recorded in the dataset.
              </p>
            </div>
          ) : history ? (
            <>
              {/* Summary Stats */}
              <div className="grid grid-cols-2 gap-3 p-3.5 rounded-xl bg-zinc-950/80 border border-zinc-800">
                <div className="flex items-center gap-3">
                  <Music className="w-5 h-5 text-emerald-400" />
                  <div>
                    <span className="text-lg font-black text-white">{history.total_songs}</span>
                    <p className="text-[10px] uppercase font-bold text-zinc-400 tracking-wider">
                      Distinct Songs in Train
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <Play className="w-5 h-5 text-teal-400 fill-teal-400/30" />
                  <div>
                    <span className="text-lg font-black text-white">{history.total_plays}</span>
                    <p className="text-[10px] uppercase font-bold text-zinc-400 tracking-wider">
                      Total Listen Plays
                    </p>
                  </div>
                </div>
              </div>

              {/* Tracks List */}
              <div className="space-y-2">
                <h4 className="text-xs font-bold uppercase tracking-wider text-zinc-400">
                  Recorded Tracks (Ranked by Play Count)
                </h4>

                <div className="divide-y divide-zinc-800/80 rounded-xl bg-zinc-950/40 border border-zinc-800 overflow-hidden">
                  {history.history.map((song, idx) => (
                    <div
                      key={song.song_id}
                      className="p-3 flex items-center justify-between hover:bg-zinc-800/40 transition-colors"
                    >
                      <div className="flex items-center gap-3 min-w-0 pr-4">
                        <span className="text-xs font-mono font-bold text-zinc-500 w-5 text-right">
                          #{idx + 1}
                        </span>
                        <div className="min-w-0">
                          <button
                            onClick={() => {
                              setHistoryModalOpen(false);
                              setSelectedSongId(song.song_id);
                            }}
                            className="text-sm font-bold text-white hover:text-emerald-400 text-left line-clamp-1 transition-colors"
                          >
                            {song.title}
                          </button>
                          <p className="text-xs text-zinc-400 line-clamp-1">{song.artist_name}</p>
                        </div>
                      </div>

                      <div className="flex items-center gap-2 flex-shrink-0">
                        <span className="px-2.5 py-0.5 rounded-full text-xs font-black bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                          {song.play_count} plays
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : null}
        </div>

        {/* Footer Note */}
        <div className="p-3.5 bg-zinc-950 border-t border-zinc-800 text-[11px] text-zinc-500 flex items-center justify-between">
          <span>Data source: Million Song Dataset Taste Profile (Train Split Only).</span>
          <button
            onClick={() => setHistoryModalOpen(false)}
            className="px-3 py-1 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-xs font-bold text-zinc-300 hover:text-white"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
