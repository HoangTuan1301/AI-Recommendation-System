import React from 'react';
import { Award, BarChart3, Cpu, Sparkles, CheckCircle2, ShieldCheck, Zap } from 'lucide-react';

export const MusicModelsPage: React.FC = () => {
  const leaderboard = [
    {
      rank: 1,
      name: 'Item-Based CF',
      family: 'Neighborhood CF',
      hr10: '0.2448',
      ndcg10: '0.1085',
      p10: '0.0307',
      r10: '0.1544',
      mrr10: '0.1238',
      coverage: '93.20%',
      latency: '9.34 ms',
      highlight: true,
    },
    {
      rank: 2,
      name: 'User-Based CF',
      family: 'Neighborhood CF',
      hr10: '0.2389',
      ndcg10: '0.1075',
      p10: '0.0297',
      r10: '0.1535',
      mrr10: '0.1201',
      coverage: '65.45%',
      latency: '1.33 ms',
      highlight: false,
    },
    {
      rank: 3,
      name: 'Implicit ALS (iALS)',
      family: 'Matrix Factorization',
      hr10: '0.2111',
      ndcg10: '0.0856',
      p10: '0.0249',
      r10: '0.1377',
      mrr10: '0.0866',
      coverage: '62.90%',
      latency: '4.91 ms',
      highlight: false,
    },
    {
      rank: 4,
      name: 'BPR-MF',
      family: 'Pairwise Ranking',
      hr10: '0.1643',
      ndcg10: '0.0674',
      p10: '0.0201',
      r10: '0.0966',
      mrr10: '0.0786',
      coverage: '45.82%',
      latency: '5.34 ms',
      highlight: false,
    },
    {
      rank: 5,
      name: 'Implicit Neural CF',
      family: 'Deep Learning Hybrid',
      hr10: '0.1345',
      ndcg10: '0.0514',
      p10: '0.0163',
      r10: '0.0774',
      mrr10: '0.0596',
      coverage: '21.03%',
      latency: '3.77 ms',
      highlight: false,
    },
    {
      rank: 6,
      name: 'Popularity Baseline',
      family: 'Heuristic Baseline',
      hr10: '0.1000',
      ndcg10: '0.0345',
      p10: '0.0116',
      r10: '0.0589',
      mrr10: '0.0366',
      coverage: '1.00%',
      latency: '0.97 ms',
      highlight: false,
    },
  ];

  return (
    <div className="min-h-screen pt-24 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-10 animate-fadeIn">
      {/* Header */}
      <div className="border-b border-zinc-800 pb-6 space-y-2">
        <div className="flex items-center gap-2">
          <span className="px-2.5 py-0.5 rounded-full text-xs font-black uppercase tracking-wider bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 flex items-center gap-1.5">
            <Award className="w-3.5 h-3.5" />
            Phase M3 Benchmark
          </span>
          <span className="text-xs text-zinc-400">Full-Catalog Ranking (K=10)</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-black text-white">MusicAI Model Leaderboard</h1>
        <p className="text-sm text-zinc-400 max-w-3xl">
          Scientific offline evaluation results across 2,520 test users on the sealed Million Song Dataset test split.
          All candidate rankings evaluate over the entire 2,706-song catalog (excluding user training history).
        </p>
      </div>

      {/* Leaderboard Table Card */}
      <div className="bg-cine-surface border border-cine-border rounded-2xl overflow-hidden shadow-2xl">
        <div className="p-4 bg-zinc-900 border-b border-zinc-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-emerald-400" />
            <span className="text-xs font-black uppercase tracking-wider text-white">
              Official Test Set Results (Top-10 Ranking)
            </span>
          </div>
          <span className="text-[11px] text-zinc-400 font-mono">N = 2,520 users</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-zinc-950/80 text-zinc-400 font-bold uppercase tracking-wider border-b border-zinc-800">
              <tr>
                <th className="py-3 px-4">Rank</th>
                <th className="py-3 px-4">Model Name</th>
                <th className="py-3 px-4">Algorithm Family</th>
                <th className="py-3 px-4 text-right">Hit Rate@10</th>
                <th className="py-3 px-4 text-right">NDCG@10</th>
                <th className="py-3 px-4 text-right">P@10</th>
                <th className="py-3 px-4 text-right">Recall@10</th>
                <th className="py-3 px-4 text-right">MRR@10</th>
                <th className="py-3 px-4 text-right">Catalog Coverage</th>
                <th className="py-3 px-4 text-right">Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800">
              {leaderboard.map((row) => (
                <tr
                  key={row.name}
                  className={`transition-colors ${
                    row.highlight
                      ? 'bg-emerald-950/30 hover:bg-emerald-900/30'
                      : 'hover:bg-zinc-800/40'
                  }`}
                >
                  <td className="py-3.5 px-4 font-black">
                    <span
                      className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs ${
                        row.rank === 1
                          ? 'bg-emerald-500 text-black font-black'
                          : row.rank === 2
                          ? 'bg-teal-700 text-white'
                          : 'bg-zinc-800 text-zinc-400'
                      }`}
                    >
                      {row.rank}
                    </span>
                  </td>
                  <td className="py-3.5 px-4 font-bold text-white flex items-center gap-2">
                    {row.name}
                    {row.highlight && (
                      <span className="px-1.5 py-0.5 rounded text-[9px] bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-black">
                        CHAMPION
                      </span>
                    )}
                  </td>
                  <td className="py-3.5 px-4 text-zinc-400">{row.family}</td>
                  <td className="py-3.5 px-4 text-right font-black text-emerald-400">{row.hr10}</td>
                  <td className="py-3.5 px-4 text-right font-black text-white">{row.ndcg10}</td>
                  <td className="py-3.5 px-4 text-right text-zinc-300">{row.p10}</td>
                  <td className="py-3.5 px-4 text-right text-zinc-300">{row.r10}</td>
                  <td className="py-3.5 px-4 text-right text-zinc-300">{row.mrr10}</td>
                  <td className="py-3.5 px-4 text-right font-semibold text-teal-300">{row.coverage}</td>
                  <td className="py-3.5 px-4 text-right font-mono text-zinc-400">{row.latency}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Scientific Insights Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="p-5 rounded-2xl bg-cine-surface border border-cine-border space-y-3">
          <div className="flex items-center gap-2 text-emerald-400 font-bold text-sm">
            <Sparkles className="w-4 h-4" />
            <h3>Why Item-Based CF Won</h3>
          </div>
          <p className="text-xs text-zinc-300 leading-relaxed">
            In music listening, song-to-song transitions have strong local co-occurrence patterns.
            With high graph sparsity (99.46%), Item-CF directly preserves pair listening frequencies
            without dimension compression loss, achieving <strong className="text-white">HR@10 = 24.5%</strong>.
          </p>
        </div>

        <div className="p-5 rounded-2xl bg-cine-surface border border-cine-border space-y-3">
          <div className="flex items-center gap-2 text-teal-400 font-bold text-sm">
            <CheckCircle2 className="w-4 h-4" />
            <h3>93.20% Catalog Coverage</h3>
          </div>
          <p className="text-xs text-zinc-300 leading-relaxed">
            Item-CF recommended <strong className="text-white">2,522 unique songs</strong> across the 2,520 test users
            with the lowest Gini coefficient (0.5661), completely avoiding popularity bias and promoting long-tail track discovery.
          </p>
        </div>

        <div className="p-5 rounded-2xl bg-cine-surface border border-cine-border space-y-3">
          <div className="flex items-center gap-2 text-indigo-400 font-bold text-sm">
            <Zap className="w-4 h-4" />
            <h3>Low-Latency Production Fit</h3>
          </div>
          <p className="text-xs text-zinc-300 leading-relaxed">
            Every model achieves single-digit millisecond inference on CPU. When integrated into FastAPI,
            end-to-end HTTP response latency operates comfortably under <strong className="text-white">25 ms (P99)</strong>.
          </p>
        </div>
      </div>
    </div>
  );
};
