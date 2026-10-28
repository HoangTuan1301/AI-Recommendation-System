import React from 'react';
import { Music, Disc3, Info, Sparkles } from 'lucide-react';
import { RecommendedSong } from '../../types/music';
import { useDomain } from '../../context/DomainContext';

interface SongCardProps {
  song: RecommendedSong;
}

export const SongCard: React.FC<SongCardProps> = ({ song }) => {
  const { setSelectedSongId } = useDomain();

  // Color rank badge for top 3
  const getRankBadge = (rank: number) => {
    if (rank === 1) {
      return 'bg-amber-400 text-black border-amber-300 font-black shadow-lg shadow-amber-400/30';
    }
    if (rank === 2) {
      return 'bg-zinc-300 text-black border-zinc-200 font-bold';
    }
    if (rank === 3) {
      return 'bg-amber-700 text-white border-amber-600 font-bold';
    }
    return 'bg-zinc-800 text-zinc-300 border-zinc-700 font-semibold';
  };

  // Generate deterministic gradient background based on song_id hash
  const getCoverGradient = (id: string) => {
    const charCode = id.charCodeAt(id.length - 1) || 0;
    const gradients = [
      'from-emerald-900/80 via-teal-950/60 to-black',
      'from-indigo-900/80 via-purple-950/60 to-black',
      'from-rose-900/80 via-pink-950/60 to-black',
      'from-cyan-900/80 via-blue-950/60 to-black',
      'from-amber-900/80 via-orange-950/60 to-black',
    ];
    return gradients[charCode % gradients.length];
  };

  return (
    <div className="group relative bg-cine-surface/90 hover:bg-zinc-800/90 border border-cine-border hover:border-emerald-500/50 rounded-2xl overflow-hidden transition-all duration-300 hover:shadow-2xl hover:shadow-emerald-950/40 hover:-translate-y-1 flex flex-col justify-between">
      {/* Top Banner & Artwork Placeholder */}
      <div className={`h-32 w-full bg-gradient-to-br ${getCoverGradient(song.song_id)} p-4 flex flex-col justify-between relative`}>
        {/* Top Badges */}
        <div className="flex items-center justify-between w-full">
          <span
            className={`w-7 h-7 rounded-full flex items-center justify-center text-xs border ${getRankBadge(
              song.rank,
            )}`}
          >
            #{song.rank}
          </span>

          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 backdrop-blur-md">
            <Sparkles className="w-3 h-3 fill-emerald-300" />
            {song.score.toFixed(2)}
          </span>
        </div>

        {/* Center Disc Art Icon */}
        <div className="flex items-center justify-center">
          <Disc3 className="w-12 h-12 text-white/30 group-hover:text-emerald-400 group-hover:rotate-45 transition-all duration-500" />
        </div>

        {/* Song ID pill */}
        <div className="flex justify-start">
          <span className="font-mono text-[10px] text-zinc-400 bg-black/60 px-2 py-0.5 rounded-md border border-white/10">
            {song.song_id}
          </span>
        </div>
      </div>

      {/* Content Info */}
      <div className="p-4 flex-grow flex flex-col justify-between">
        <div>
          <h3
            className="text-base font-bold text-white group-hover:text-emerald-300 transition-colors line-clamp-1"
            title={song.title}
          >
            {song.title}
          </h3>
          <p
            className="text-xs font-medium text-zinc-400 line-clamp-1 mt-0.5"
            title={song.artist_name}
          >
            {song.artist_name}
          </p>
        </div>

        {/* Action Button */}
        <div className="mt-4 pt-3 border-t border-zinc-800/80 flex items-center justify-between">
          <span className="inline-flex items-center gap-1 text-[11px] text-zinc-500 font-medium">
            <Music className="w-3 h-3 text-emerald-400" />
            MSD Verified
          </span>

          <button
            onClick={() => setSelectedSongId(song.song_id)}
            className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-semibold bg-zinc-800 hover:bg-emerald-600 text-zinc-300 hover:text-white border border-zinc-700 hover:border-emerald-500 transition-all"
            aria-label={`View details for ${song.title}`}
          >
            <Info className="w-3 h-3" />
            Details
          </button>
        </div>
      </div>
    </div>
  );
};
