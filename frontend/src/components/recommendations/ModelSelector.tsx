import React from 'react';
import { ModelInfo } from '../../types';
import { Brain, Layers, Cpu, Sparkles, Network } from 'lucide-react';

interface ModelSelectorProps {
  models: ModelInfo[];
  selectedModelId: string;
  onSelectModel: (modelId: string) => void;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  models,
  selectedModelId,
  onSelectModel,
}) => {
  const families = [
    { name: 'Deep Learning & Hybrid', badge: 'State-of-the-Art', icon: Sparkles },
    { name: 'Neighborhood CF', badge: 'Memory-Based', icon: Network },
    { name: 'Matrix Factorization', badge: 'Latent Factors', icon: Layers },
    { name: 'Baseline', badge: 'Heuristic & Content', icon: Brain },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-sm font-bold uppercase tracking-wider text-cine-muted mb-3">
          Select AI Algorithm (10 Models Available)
        </h3>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
        {models.map((m) => {
          const isSelected = m.id === selectedModelId;
          const isNeural = m.id === 'neural_cf';
          const isHybrid = m.id === 'hybrid';

          return (
            <div
              key={m.id}
              onClick={() => onSelectModel(m.id)}
              className={`p-3.5 rounded-xl border cursor-pointer transition-all flex flex-col justify-between relative ${
                isSelected
                  ? 'bg-gradient-to-b from-cine-red/20 to-cine-surface border-cine-red shadow-glow-red scale-[1.02]'
                  : 'bg-cine-surface border-cine-border hover:border-zinc-500 hover:bg-cine-hover'
              }`}
            >
              {/* Highlight Top 1 & Top 2 Models */}
              {isNeural && (
                <span className="absolute -top-2.5 right-2 px-2 py-0.5 rounded-full bg-cine-red text-white text-[9px] font-black uppercase tracking-wider shadow">
                  Rank #1 (82.03% HR)
                </span>
              )}
              {isHybrid && (
                <span className="absolute -top-2.5 right-2 px-2 py-0.5 rounded-full bg-cyan-600 text-white text-[9px] font-black uppercase tracking-wider shadow">
                  Rank #2 (65.78% HR)
                </span>
              )}

              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[10px] font-bold text-cine-muted uppercase tracking-wider">
                    {m.family}
                  </span>
                  <span className="text-[10px] text-zinc-500 font-mono">
                    {m.complexity}
                  </span>
                </div>
                <h4 className="text-sm font-bold text-white mb-1 leading-snug">
                  {m.name}
                </h4>
                <p className="text-[11px] text-cine-muted line-clamp-2 leading-relaxed">
                  {m.description}
                </p>
              </div>

              <div className="mt-3 pt-2.5 border-t border-cine-border/60 flex items-center justify-between text-[10px]">
                <span className="text-cine-muted">Scale:</span>
                <span className="font-semibold text-emerald-400 font-mono">
                  {m.score_range}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
