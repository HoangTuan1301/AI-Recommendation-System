import React from 'react';
import { User, Sparkles, Loader2, Sliders } from 'lucide-react';
import { useUser } from '../../context/UserContext';

interface RecommendationControlsProps {
  topN: number;
  setTopN: (n: number) => void;
  onGenerate: () => void;
  loading: boolean;
  modelName: string;
}

export const RecommendationControls: React.FC<RecommendationControlsProps> = ({
  topN,
  setTopN,
  onGenerate,
  loading,
  modelName,
}) => {
  const { currentUserId, setCurrentUserId } = useUser();

  return (
    <div className="bg-cine-surface border border-cine-border rounded-2xl p-6 shadow-xl space-y-6">
      <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-6">
        {/* User Selection */}
        <div className="flex-1 space-y-2">
          <label className="text-xs font-bold uppercase tracking-wider text-cine-muted flex items-center gap-1.5">
            <User className="w-3.5 h-3.5 text-cine-red" />
            <span>Target User ID (MovieLens 100K)</span>
          </label>
          <div className="flex items-center gap-3">
            <input
              type="number"
              min={1}
              max={943}
              value={currentUserId}
              onChange={(e) => {
                const val = parseInt(e.target.value, 10);
                if (!isNaN(val) && val >= 1 && val <= 943) {
                  setCurrentUserId(val);
                }
              }}
              className="w-32 px-4 py-2.5 rounded-xl bg-cine-bg border border-cine-border text-white font-bold text-center text-lg focus:outline-none focus:border-cine-red transition-all"
            />
            <div className="flex flex-wrap gap-1.5">
              {[1, 196, 42, 100, 250].map((uid) => (
                <button
                  key={uid}
                  type="button"
                  onClick={() => setCurrentUserId(uid)}
                  className={`px-2.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                    currentUserId === uid
                      ? 'bg-cine-red text-white'
                      : 'bg-cine-hover text-cine-muted hover:text-white'
                  }`}
                >
                  #{uid}
                </button>
              ))}
            </div>
          </div>
          <span className="text-[11px] text-zinc-500 block">
            Choose from 943 active users with training rating histories.
          </span>
        </div>

        {/* Top-N Selector */}
        <div className="space-y-2">
          <label className="text-xs font-bold uppercase tracking-wider text-cine-muted flex items-center gap-1.5">
            <Sliders className="w-3.5 h-3.5 text-cine-red" />
            <span>List Size (Cutoff K)</span>
          </label>
          <div className="flex items-center gap-2 bg-cine-bg p-1 rounded-xl border border-cine-border">
            {[5, 10, 20].map((k) => (
              <button
                key={k}
                type="button"
                onClick={() => setTopN(k)}
                className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${
                  topN === k
                    ? 'bg-cine-red text-white shadow'
                    : 'text-cine-muted hover:text-white'
                }`}
              >
                Top {k}
              </button>
            ))}
          </div>
        </div>

        {/* Generate Button */}
        <div className="flex items-end">
          <button
            onClick={onGenerate}
            disabled={loading}
            className="w-full md:w-auto px-8 py-3.5 rounded-xl bg-gradient-to-r from-cine-red to-purple-600 hover:brightness-110 text-white font-bold text-sm flex items-center justify-center gap-2 shadow-glow-red transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:scale-105 active:scale-95"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Generating AI Predictions...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                <span>Generate Recommendations</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
