import React from 'react';
import { Cpu, Sparkles, Zap, Layers, Activity } from 'lucide-react';
import { MusicModelInfo } from '../../types/music';

interface MusicModelSelectorProps {
  models: MusicModelInfo[];
  selectedModelId: string;
  onSelectModel: (modelId: string) => void;
  disabled?: boolean;
}

export const MusicModelSelector: React.FC<MusicModelSelectorProps> = ({
  models,
  selectedModelId,
  onSelectModel,
  disabled = false,
}) => {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="text-sm font-bold text-white flex items-center gap-2">
          <Cpu className="w-4 h-4 text-emerald-400" />
          Select AI Recommendation Algorithm:
        </label>
        <span className="text-xs text-zinc-400">6 models available</span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {models.map((model) => {
          const isSelected = model.id === selectedModelId;
          const isAvailable = model.is_available !== false;

          return (
            <button
              key={model.id}
              type="button"
              disabled={disabled || !isAvailable}
              onClick={() => onSelectModel(model.id)}
              className={`relative text-left p-3.5 rounded-xl border transition-all duration-200 flex flex-col justify-between ${
                isSelected
                  ? 'bg-emerald-950/40 border-emerald-500 shadow-lg shadow-emerald-950/40 scale-[1.02]'
                  : 'bg-cine-surface/90 hover:bg-zinc-800/80 border-cine-border hover:border-zinc-600'
              } ${!isAvailable ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
            >
              {/* Header Badges */}
              <div className="flex items-center justify-between gap-2 mb-2">
                <span className="text-[10px] font-extrabold uppercase tracking-wider px-2 py-0.5 rounded bg-zinc-800 text-zinc-300 border border-zinc-700/60">
                  {model.family}
                </span>

                <div className="flex items-center gap-1.5">
                  {model.is_default && (
                    <span className="text-[10px] font-black uppercase tracking-wider px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 flex items-center gap-1">
                      <Sparkles className="w-2.5 h-2.5 fill-emerald-300" />
                      Default (Top 1)
                    </span>
                  )}
                  {model.latency_profile && (
                    <span className="text-[10px] font-bold text-zinc-400 flex items-center gap-0.5">
                      <Zap className="w-2.5 h-2.5 text-amber-400" />
                      {model.latency_profile}
                    </span>
                  )}
                </div>
              </div>

              {/* Title & Description */}
              <div>
                <h4
                  className={`text-sm font-black transition-colors ${
                    isSelected ? 'text-emerald-300' : 'text-white'
                  }`}
                >
                  {model.name}
                </h4>
                <p className="text-xs text-zinc-400 mt-1 line-clamp-2 leading-relaxed">
                  {model.description}
                </p>
              </div>

              {/* Footer Status */}
              <div className="mt-3 pt-2 border-t border-zinc-800/80 flex items-center justify-between text-[11px]">
                <span className="text-zinc-500 font-mono text-[10px]">{model.id}</span>
                <span
                  className={`flex items-center gap-1 font-bold ${
                    isSelected ? 'text-emerald-400' : 'text-zinc-500'
                  }`}
                >
                  <Activity className="w-3 h-3" />
                  {isSelected ? 'Active Model' : 'Select'}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};
