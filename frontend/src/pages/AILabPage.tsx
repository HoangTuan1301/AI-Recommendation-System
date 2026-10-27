import React, { useEffect, useState } from 'react';
import { api } from '../services/api';
import { MetricsSummaryResponse, ModelInfo } from '../types';
import { LeaderboardTable } from '../components/lab/LeaderboardTable';
import { MetricCharts } from '../components/lab/MetricCharts';
import { ModelBattle } from '../components/lab/ModelBattle';
import { MetricsExplainer } from '../components/lab/MetricsExplainer';
import { TableSkeleton } from '../components/ui/LoadingSkeleton';
import { Trophy, BarChart2, Swords, BookOpen, Layers } from 'lucide-react';

export const AILabPage: React.FC = () => {
  const [metricsSummary, setMetricsSummary] = useState<MetricsSummaryResponse | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'leaderboard' | 'charts' | 'battle' | 'explainer'>('leaderboard');

  useEffect(() => {
    Promise.all([api.getMetrics(), api.getModels()])
      .then(([metricsData, modelsData]) => {
        setMetricsSummary(metricsData);
        setModels(modelsData);
      })
      .catch((err) => console.error('Failed to load lab data:', err))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 pb-16 space-y-8">
      {/* Page Header */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <span className="px-2.5 py-1 rounded-md bg-purple-600 text-white text-[11px] font-black uppercase tracking-wider">
            AI RESEARCH LAB
          </span>
          <span className="text-xs text-cine-muted font-bold">
            Empirical Benchmark & Evaluation Harness
          </span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-black text-white tracking-tight">
          Model Leaderboard & Academic Evaluation
        </h1>
        <p className="text-sm text-cine-muted mt-1 max-w-3xl">
          Comparative analysis of 10 recommendation systems on MovieLens 100K across 5 top-N ranking metrics evaluated on a held-out test set of 935 users.
        </p>
      </div>

      {/* KPI Cards */}
      {metricsSummary && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl bg-cine-surface border border-cine-border text-center">
            <span className="text-[11px] text-cine-muted uppercase font-bold block mb-1">
              Top Hit Rate@10
            </span>
            <span className="text-2xl font-black text-emerald-400 font-mono">82.03%</span>
            <span className="text-[11px] text-zinc-400 block mt-1 font-semibold">{metricsSummary.best_hit_rate_model}</span>
          </div>

          <div className="p-4 rounded-xl bg-cine-surface border border-cine-border text-center">
            <span className="text-[11px] text-cine-muted uppercase font-bold block mb-1">
              Top NDCG@10
            </span>
            <span className="text-2xl font-black text-cyan-400 font-mono">0.3083</span>
            <span className="text-[11px] text-zinc-400 block mt-1 font-semibold">{metricsSummary.best_ndcg_model}</span>
          </div>

          <div className="p-4 rounded-xl bg-cine-surface border border-cine-border text-center">
            <span className="text-[11px] text-cine-muted uppercase font-bold block mb-1">
              Evaluated Test Users
            </span>
            <span className="text-2xl font-black text-white font-mono">935</span>
            <span className="text-[11px] text-zinc-400 block mt-1 font-semibold">Positive Test Interactions</span>
          </div>

          <div className="p-4 rounded-xl bg-cine-surface border border-cine-border text-center">
            <span className="text-[11px] text-cine-muted uppercase font-bold block mb-1">
              Models Tested
            </span>
            <span className="text-2xl font-black text-amber-400 font-mono">10</span>
            <span className="text-[11px] text-zinc-400 block mt-1 font-semibold">Across 4 AI Families</span>
          </div>
        </div>
      )}

      {/* Sub-Navigation Tabs */}
      <div className="flex border-b border-cine-border overflow-x-auto no-scrollbar gap-2">
        <button
          onClick={() => setActiveTab('leaderboard')}
          className={`flex items-center gap-2 px-5 py-3 font-semibold text-xs uppercase tracking-wider border-b-2 transition-all ${
            activeTab === 'leaderboard'
              ? 'border-cine-red text-white'
              : 'border-transparent text-cine-muted hover:text-white'
          }`}
        >
          <Trophy className="w-4 h-4 text-amber-400" />
          <span>Leaderboard Table</span>
        </button>

        <button
          onClick={() => setActiveTab('charts')}
          className={`flex items-center gap-2 px-5 py-3 font-semibold text-xs uppercase tracking-wider border-b-2 transition-all ${
            activeTab === 'charts'
              ? 'border-cine-red text-white'
              : 'border-transparent text-cine-muted hover:text-white'
          }`}
        >
          <BarChart2 className="w-4 h-4 text-cyan-400" />
          <span>Metric Visualizations</span>
        </button>

        <button
          onClick={() => setActiveTab('battle')}
          className={`flex items-center gap-2 px-5 py-3 font-semibold text-xs uppercase tracking-wider border-b-2 transition-all ${
            activeTab === 'battle'
              ? 'border-cine-red text-white'
              : 'border-transparent text-cine-muted hover:text-white'
          }`}
        >
          <Swords className="w-4 h-4 text-cine-red" />
          <span>1v1 Model Battle</span>
        </button>

        <button
          onClick={() => setActiveTab('explainer')}
          className={`flex items-center gap-2 px-5 py-3 font-semibold text-xs uppercase tracking-wider border-b-2 transition-all ${
            activeTab === 'explainer'
              ? 'border-cine-red text-white'
              : 'border-transparent text-cine-muted hover:text-white'
          }`}
        >
          <BookOpen className="w-4 h-4 text-purple-400" />
          <span>Metrics & Methodology</span>
        </button>
      </div>

      {/* Active Tab Views */}
      {loading ? (
        <TableSkeleton />
      ) : (
        <div className="space-y-8">
          {activeTab === 'leaderboard' && metricsSummary && (
            <LeaderboardTable metrics={metricsSummary.metrics} />
          )}

          {activeTab === 'charts' && metricsSummary && (
            <MetricCharts metrics={metricsSummary.metrics} />
          )}

          {activeTab === 'battle' && (
            <ModelBattle models={models} />
          )}

          {activeTab === 'explainer' && (
            <MetricsExplainer />
          )}
        </div>
      )}
    </div>
  );
};
