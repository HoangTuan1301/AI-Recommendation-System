import React, { useState } from 'react';
import { ModelMetric } from '../../types';
import { Trophy, ArrowUpDown, Clock, CheckCircle } from 'lucide-react';

interface LeaderboardTableProps {
  metrics: ModelMetric[];
}

export const LeaderboardTable: React.FC<LeaderboardTableProps> = ({ metrics }) => {
  const [sortKey, setSortKey] = useState<keyof ModelMetric>('hit_rate_at_10');
  const [sortAsc, setSortAsc] = useState(false);

  const handleSort = (key: keyof ModelMetric) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(false);
    }
  };

  const sorted = [...metrics].sort((a, b) => {
    const valA = a[sortKey];
    const valB = b[sortKey];
    if (typeof valA === 'number' && typeof valB === 'number') {
      return sortAsc ? valA - valB : valB - valA;
    }
    return 0;
  });

  return (
    <div className="bg-cine-surface border border-cine-border rounded-2xl overflow-hidden shadow-xl">
      <div className="p-4 sm:p-6 border-b border-cine-border flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <div>
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <Trophy className="w-5 h-5 text-amber-400" />
            <span>Official 10-Model Benchmark Leaderboard</span>
          </h3>
          <p className="text-xs text-cine-muted mt-0.5">
            Evaluated on held-out Test set (935 test users with positive ratings r &ge; 4.0) under identical Top-10 protocol.
          </p>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs sm:text-sm">
          <thead className="bg-cine-bg text-cine-muted uppercase text-[11px] font-bold tracking-wider border-b border-cine-border">
            <tr>
              <th className="py-3.5 px-4 text-center">Rank</th>
              <th className="py-3.5 px-4 cursor-pointer hover:text-white" onClick={() => handleSort('model')}>
                <div className="flex items-center gap-1">
                  <span>Model Name</span>
                  <ArrowUpDown className="w-3 h-3" />
                </div>
              </th>
              <th className="py-3.5 px-4 cursor-pointer hover:text-white text-right" onClick={() => handleSort('hit_rate_at_10')}>
                <div className="flex items-center justify-end gap-1">
                  <span>Hit Rate@10</span>
                  <ArrowUpDown className="w-3 h-3" />
                </div>
              </th>
              <th className="py-3.5 px-4 cursor-pointer hover:text-white text-right" onClick={() => handleSort('ndcg_at_10')}>
                <div className="flex items-center justify-end gap-1">
                  <span>NDCG@10</span>
                  <ArrowUpDown className="w-3 h-3" />
                </div>
              </th>
              <th className="py-3.5 px-4 cursor-pointer hover:text-white text-right" onClick={() => handleSort('precision_at_10')}>
                <div className="flex items-center justify-end gap-1">
                  <span>Precision@10</span>
                  <ArrowUpDown className="w-3 h-3" />
                </div>
              </th>
              <th className="py-3.5 px-4 cursor-pointer hover:text-white text-right" onClick={() => handleSort('recall_at_10')}>
                <div className="flex items-center justify-end gap-1">
                  <span>Recall@10</span>
                  <ArrowUpDown className="w-3 h-3" />
                </div>
              </th>
              <th className="py-3.5 px-4 cursor-pointer hover:text-white text-right" onClick={() => handleSort('f1_at_10')}>
                <div className="flex items-center justify-end gap-1">
                  <span>F1@10</span>
                  <ArrowUpDown className="w-3 h-3" />
                </div>
              </th>
              <th className="py-3.5 px-4 text-right">Eval Time</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-cine-border/60">
            {sorted.map((row, idx) => {
              const isBest = row.model_id === 'neural_cf';
              const isSecond = row.model_id === 'hybrid';

              return (
                <tr
                  key={row.model_id}
                  className={`hover:bg-cine-hover/60 transition-colors ${
                    isBest ? 'bg-cine-red/5' : ''
                  }`}
                >
                  <td className="py-3 px-4 text-center font-bold">
                    {idx === 0 ? (
                      <span className="w-6 h-6 rounded-full bg-amber-400 text-black font-black text-xs inline-flex items-center justify-center">
                        1
                      </span>
                    ) : idx === 1 ? (
                      <span className="w-6 h-6 rounded-full bg-zinc-300 text-black font-black text-xs inline-flex items-center justify-center">
                        2
                      </span>
                    ) : idx === 2 ? (
                      <span className="w-6 h-6 rounded-full bg-amber-700 text-white font-black text-xs inline-flex items-center justify-center">
                        3
                      </span>
                    ) : (
                      <span className="text-cine-muted">#{idx + 1}</span>
                    )}
                  </td>
                  <td className="py-3 px-4">
                    <div className="font-bold text-white flex items-center gap-2">
                      <span>{row.model}</span>
                      {isBest && (
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-cine-red text-white uppercase">
                          Winner
                        </span>
                      )}
                      {isSecond && (
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-cyan-700 text-white uppercase">
                          Top Hybrid
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="py-3 px-4 text-right font-bold text-emerald-400 font-mono">
                    {(row.hit_rate_at_10 * 100).toFixed(2)}%
                  </td>
                  <td className="py-3 px-4 text-right font-mono font-medium text-white">
                    {row.ndcg_at_10.toFixed(4)}
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-zinc-300">
                    {row.precision_at_10.toFixed(4)}
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-zinc-300">
                    {row.recall_at_10.toFixed(4)}
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-zinc-300">
                    {row.f1_at_10.toFixed(4)}
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-cine-muted text-xs">
                    {row.evaluate_time_seconds.toFixed(2)}s
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
