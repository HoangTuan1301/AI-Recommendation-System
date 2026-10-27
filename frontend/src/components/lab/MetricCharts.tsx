import React, { useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Legend,
} from 'recharts';
import { ModelMetric } from '../../types';

interface MetricChartsProps {
  metrics: ModelMetric[];
}

export const MetricCharts: React.FC<MetricChartsProps> = ({ metrics }) => {
  const [selectedMetric, setSelectedMetric] = useState<string>('hit_rate_at_10');

  // Chart 1 data: Hit Rate@10
  const hitRateData = [...metrics]
    .sort((a, b) => b.hit_rate_at_10 - a.hit_rate_at_10)
    .map((m) => ({
      name: m.model,
      hitRate: parseFloat((m.hit_rate_at_10 * 100).toFixed(2)),
      isBest: m.model_id === 'neural_cf',
      isHybrid: m.model_id === 'hybrid',
    }));

  // Chart 2 data: Execution Time
  const timeData = [...metrics].map((m) => ({
    name: m.model,
    fitTime: m.fit_time_seconds,
    evalTime: m.evaluate_time_seconds,
  }));

  const metricOptions = [
    { key: 'hit_rate_at_10', label: 'Hit Rate@10', format: (v: number) => `${(v * 100).toFixed(2)}%` },
    { key: 'ndcg_at_10', label: 'NDCG@10', format: (v: number) => v.toFixed(4) },
    { key: 'precision_at_10', label: 'Precision@10', format: (v: number) => v.toFixed(4) },
    { key: 'recall_at_10', label: 'Recall@10', format: (v: number) => v.toFixed(4) },
    { key: 'f1_at_10', label: 'F1-Score@10', format: (v: number) => v.toFixed(4) },
  ];

  const currentOption = metricOptions.find((o) => o.key === selectedMetric) || metricOptions[0];

  const dynamicMetricData = [...metrics]
    .sort((a, b) => {
      const vA = (a as any)[selectedMetric] || 0;
      const vB = (b as any)[selectedMetric] || 0;
      return vB - vA;
    })
    .map((m) => ({
      name: m.model,
      value: (m as any)[selectedMetric] || 0,
      isBest: m.model_id === 'neural_cf',
    }));

  return (
    <div className="space-y-8">
      {/* Chart 1: Hit Rate@10 Main Comparison */}
      <div className="bg-cine-surface border border-cine-border rounded-2xl p-6 shadow-xl">
        <div className="mb-4">
          <h3 className="text-base font-bold text-white">
            Hit Rate@10 Benchmark Comparison (%)
          </h3>
          <p className="text-xs text-cine-muted">
            Fraction of evaluated test users receiving at least 1 relevant recommendation in Top 10.
          </p>
        </div>

        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={hitRateData}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
            >
              <XAxis type="number" domain={[0, 100]} tick={{ fill: '#A1A1AA', fontSize: 11 }} />
              <YAxis dataKey="name" type="category" tick={{ fill: '#FFFFFF', fontSize: 11 }} width={120} />
              <Tooltip
                contentStyle={{ backgroundColor: '#181A21', borderColor: '#292B33', borderRadius: '8px', color: '#fff' }}
                formatter={(val: any) => [`${val}%`, 'Hit Rate@10']}
              />
              <Bar dataKey="hitRate" radius={[0, 6, 6, 0]}>
                {hitRateData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.isBest ? '#E50914' : entry.isHybrid ? '#06B6D4' : '#334155'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 2: Dynamic Metric Selector Chart */}
      <div className="bg-cine-surface border border-cine-border rounded-2xl p-6 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
          <div>
            <h3 className="text-base font-bold text-white">
              Ranking Metric Distribution
            </h3>
            <p className="text-xs text-cine-muted">
              Select any evaluation metric to compare algorithm performance.
            </p>
          </div>

          <div className="flex flex-wrap gap-1.5">
            {metricOptions.map((opt) => (
              <button
                key={opt.key}
                onClick={() => setSelectedMetric(opt.key)}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  selectedMetric === opt.key
                    ? 'bg-cine-red text-white'
                    : 'bg-cine-bg text-cine-muted hover:text-white border border-cine-border'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={dynamicMetricData} margin={{ top: 10, right: 20, left: 10, bottom: 40 }}>
              <XAxis
                dataKey="name"
                tick={{ fill: '#A1A1AA', fontSize: 10 }}
                angle={-25}
                textAnchor="end"
                interval={0}
              />
              <YAxis tick={{ fill: '#A1A1AA', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#181A21', borderColor: '#292B33', borderRadius: '8px', color: '#fff' }}
                formatter={(val: any) => [currentOption.format(val), currentOption.label]}
              />
              <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                {dynamicMetricData.map((entry, index) => (
                  <Cell
                    key={`d-cell-${index}`}
                    fill={entry.isBest ? '#E50914' : '#6366F1'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 3: Execution Latency (Fit Time vs Eval Time) */}
      <div className="bg-cine-surface border border-cine-border rounded-2xl p-6 shadow-xl">
        <div className="mb-4">
          <h3 className="text-base font-bold text-white">
            Computational Efficiency (Seconds)
          </h3>
          <p className="text-xs text-cine-muted">
            Evaluation time across all 935 test users vs Offline training time.
          </p>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={timeData} margin={{ top: 10, right: 20, left: 10, bottom: 40 }}>
              <XAxis
                dataKey="name"
                tick={{ fill: '#A1A1AA', fontSize: 10 }}
                angle={-25}
                textAnchor="end"
                interval={0}
              />
              <YAxis tick={{ fill: '#A1A1AA', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#181A21', borderColor: '#292B33', borderRadius: '8px', color: '#fff' }}
                formatter={(val: any) => [`${val}s`, '']}
              />
              <Legend wrapperStyle={{ color: '#fff', fontSize: '12px' }} />
              <Bar dataKey="evalTime" name="Evaluation Time (935 users)" fill="#10B981" radius={[4, 4, 0, 0]} />
              <Bar dataKey="fitTime" name="Training / Fit Time" fill="#F59E0B" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
