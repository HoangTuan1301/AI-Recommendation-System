import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Music,
  Sparkles,
  Zap,
  TrendingUp,
  Database,
  Users,
  Layers,
  ArrowRight,
  Disc3,
  Award,
} from 'lucide-react';
import { musicApi } from '../services/musicApi';
import { MusicHealthResponse } from '../types/music';

export const MusicHomePage: React.FC = () => {
  const [health, setHealth] = useState<MusicHealthResponse | null>(null);

  useEffect(() => {
    musicApi.getHealth().then(setHealth).catch(() => {});
  }, []);

  const algorithms = [
    {
      name: 'Item-Based CF',
      badge: 'Benchmark Top 1',
      badgeColor: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
      description: 'Learns item-item listening co-occurrence via adjusted Cosine similarity across Top-50 neighbors.',
      metrics: 'HR@10: 24.5% | NDCG@10: 0.1085',
    },
    {
      name: 'User-Based CF',
      badge: 'Runner Up',
      badgeColor: 'bg-teal-500/20 text-teal-400 border-teal-500/30',
      description: 'Discovers listeners with similar music taste vectors and aggregates their high-affinity songs.',
      metrics: 'HR@10: 23.9% | NDCG@10: 0.1075',
    },
    {
      name: 'Implicit ALS',
      badge: 'Vector Embeddings',
      badgeColor: 'bg-indigo-500/20 text-indigo-400 border-indigo-500/30',
      description: 'Hu-Koren-Volinsky matrix factorization with logarithmic confidence weighting into 32-dim latent space.',
      metrics: 'HR@10: 21.1% | Latency: 4.9 ms',
    },
    {
      name: 'BPR-MF',
      badge: 'Pairwise Ranking',
      badgeColor: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
      description: 'Directly optimizes the relative preference margin between played and unplayed songs via sigmoid loss.',
      metrics: 'HR@10: 16.4% | Latency: 5.3 ms',
    },
    {
      name: 'Implicit Neural CF',
      badge: 'Deep Learning',
      badgeColor: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
      description: 'Non-linear neural architecture combining Generalized Matrix Factorization (GMF) and Multi-Layer Perceptrons.',
      metrics: 'HR@10: 13.5% | PyTorch Core',
    },
    {
      name: 'Popularity Baseline',
      badge: 'Cold-Start Fallback',
      badgeColor: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
      description: 'Aggregates global total play count frequencies across all listeners to safely serve new users.',
      metrics: 'HR@10: 10.0% | Latency: 0.9 ms',
    },
  ];

  return (
    <div className="min-h-screen pt-20 pb-20 space-y-16 animate-fadeIn">
      {/* Hero Section */}
      <section className="relative overflow-hidden px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto pt-10 pb-16">
        <div className="absolute top-0 right-1/4 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-1/3 w-80 h-80 bg-teal-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative text-center max-w-3xl mx-auto space-y-6">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-black uppercase tracking-wider">
            <Sparkles className="w-3.5 h-3.5 fill-emerald-400" />
            Multi-Domain Music Intelligence
          </div>

          <h1 className="text-4xl sm:text-6xl font-black text-white tracking-tight leading-tight">
            Personalized Music Discovery Powered by <span className="text-emerald-400">Real AI</span>
          </h1>

          <p className="text-base sm:text-lg text-zinc-300 font-medium leading-relaxed">
            Move beyond generic top hits. MusicAI trains 6 collaborative filtering and deep learning
            models to recommend individual songs tailored to your unique taste.
          </p>

          <div className="flex flex-wrap items-center justify-center gap-4 pt-4">
            <Link
              to="/music/recommendations"
              className="px-6 py-3.5 rounded-2xl font-black text-sm bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-white shadow-xl shadow-emerald-950/60 transition-all flex items-center gap-2 hover:scale-105"
            >
              <Music className="w-4 h-4" />
              Explore Recommendations
              <ArrowRight className="w-4 h-4" />
            </Link>

            <Link
              to="/music/models"
              className="px-6 py-3.5 rounded-2xl font-bold text-sm bg-zinc-900/90 hover:bg-zinc-800 text-zinc-200 border border-zinc-700/80 transition-all flex items-center gap-2"
            >
              <Award className="w-4 h-4 text-emerald-400" />
              View Benchmark Leaderboard
            </Link>
          </div>
        </div>
      </section>

      {/* Real-time Dataset Stats Strip */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-6 rounded-3xl bg-zinc-900/80 border border-zinc-800 shadow-2xl backdrop-blur-md">
          <div className="flex items-center gap-3.5">
            <div className="p-3 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
              <Database className="w-6 h-6" />
            </div>
            <div>
              <div className="text-2xl font-black text-white">
                {health ? health.catalog_songs.toLocaleString() : '2,706'}
              </div>
              <div className="text-xs uppercase tracking-wider text-zinc-400 font-bold">Catalog Songs</div>
            </div>
          </div>

          <div className="flex items-center gap-3.5">
            <div className="p-3 rounded-2xl bg-teal-500/10 border border-teal-500/20 text-teal-400">
              <Users className="w-6 h-6" />
            </div>
            <div>
              <div className="text-2xl font-black text-white">
                {health ? health.total_users.toLocaleString() : '2,520'}
              </div>
              <div className="text-xs uppercase tracking-wider text-zinc-400 font-bold">Listeners Evaluated</div>
            </div>
          </div>

          <div className="flex items-center gap-3.5">
            <div className="p-3 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
              <TrendingUp className="w-6 h-6" />
            </div>
            <div>
              <div className="text-2xl font-black text-white">
                {health ? health.total_train_interactions.toLocaleString() : '36,994'}
              </div>
              <div className="text-xs uppercase tracking-wider text-zinc-400 font-bold">Train Interactions</div>
            </div>
          </div>

          <div className="flex items-center gap-3.5">
            <div className="p-3 rounded-2xl bg-purple-500/10 border border-purple-500/20 text-purple-400">
              <Layers className="w-6 h-6" />
            </div>
            <div>
              <div className="text-2xl font-black text-white">
                {health ? health.available_models : '6'} Models
              </div>
              <div className="text-xs uppercase tracking-wider text-zinc-400 font-bold">AI Architectures</div>
            </div>
          </div>
        </div>
      </section>

      {/* Algorithm Showcase */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6">
        <div className="text-center max-w-2xl mx-auto space-y-2">
          <h2 className="text-2xl sm:text-3xl font-black text-white">
            6 Specialized AI Music Models
          </h2>
          <p className="text-sm text-zinc-400">
            From classic neighborhood methods to neural embeddings, compare how different algorithms
            handle high data sparsity (99.46%).
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {algorithms.map((algo) => (
            <div
              key={algo.name}
              className="p-5 rounded-2xl bg-cine-surface border border-cine-border hover:border-emerald-500/40 transition-all space-y-3 flex flex-col justify-between"
            >
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <h3 className="text-base font-black text-white">{algo.name}</h3>
                  <span
                    className={`text-[10px] font-extrabold uppercase tracking-wider px-2 py-0.5 rounded border ${algo.badgeColor}`}
                  >
                    {algo.badge}
                  </span>
                </div>
                <p className="text-xs text-zinc-400 leading-relaxed">{algo.description}</p>
              </div>

              <div className="pt-3 border-t border-zinc-800 text-[11px] font-bold text-emerald-400 font-mono">
                {algo.metrics}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};
