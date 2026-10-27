import React from 'react';
import { AlertCircle, CheckCircle2, BookOpen } from 'lucide-react';

export const MetricsExplainer: React.FC = () => {
  return (
    <div className="bg-cine-surface border border-cine-border rounded-2xl p-6 shadow-xl space-y-6">
      <div className="flex items-center gap-2 border-b border-cine-border pb-4">
        <BookOpen className="w-5 h-5 text-cine-red" />
        <h3 className="text-base font-bold text-white">
          Scientific Evaluation Protocol & Metric Definitions
        </h3>
      </div>

      {/* Critical Clarification Callout */}
      <div className="p-4 rounded-xl bg-cine-red/10 border border-cine-red/40 flex items-start gap-3">
        <AlertCircle className="w-5 h-5 text-cine-red flex-shrink-0 mt-0.5" />
        <div className="text-xs sm:text-sm text-zinc-200 leading-relaxed">
          <strong className="text-white font-bold block mb-1">
            Crucial Scientific Clarification on Hit Rate@10 = 82.03%:
          </strong>
          Hit Rate@10 does <strong>NOT</strong> mean that 82.03% of all recommended movies are correct.
          In academic recommender systems, <strong>Hit Rate@10 = 82.03%</strong> strictly means:
          Out of the 935 evaluated test users, exactly <strong>82.03% of users had at least ONE relevant movie ($r \ge 4.0$)</strong> in their personalized Top-10 recommendation list.
        </div>
      </div>

      {/* 5 Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="p-4 rounded-xl bg-cine-bg border border-cine-border space-y-1.5">
          <div className="text-xs font-bold text-cine-red uppercase tracking-wider">
            1. Precision@10
          </div>
          <p className="text-xs text-cine-muted leading-relaxed">
            Measures the proportion of recommended items that are actually relevant ($r \ge 4.0$ in Test):
            <br />
            <code className="text-zinc-300 font-mono text-[11px]">|Rec_10 ∩ Rel_u| / 10</code>
          </p>
        </div>

        <div className="p-4 rounded-xl bg-cine-bg border border-cine-border space-y-1.5">
          <div className="text-xs font-bold text-cyan-400 uppercase tracking-wider">
            2. Recall@10
          </div>
          <p className="text-xs text-cine-muted leading-relaxed">
            Measures coverage of the user's test preferences recovered by the top-10 list:
            <br />
            <code className="text-zinc-300 font-mono text-[11px]">|Rec_10 ∩ Rel_u| / |Rel_u|</code>
          </p>
        </div>

        <div className="p-4 rounded-xl bg-cine-bg border border-cine-border space-y-1.5">
          <div className="text-xs font-bold text-amber-400 uppercase tracking-wider">
            3. F1-Score@10
          </div>
          <p className="text-xs text-cine-muted leading-relaxed">
            Harmonic mean balancing precision and recall:
            <br />
            <code className="text-zinc-300 font-mono text-[11px]">2 · (P · R) / (P + R)</code>
          </p>
        </div>

        <div className="p-4 rounded-xl bg-cine-bg border border-cine-border space-y-1.5">
          <div className="text-xs font-bold text-purple-400 uppercase tracking-wider">
            4. NDCG@10
          </div>
          <p className="text-xs text-cine-muted leading-relaxed">
            Normalized Discounted Cumulative Gain penalizing relevant items placed lower in the ranking:
            <br />
            <code className="text-zinc-300 font-mono text-[11px]">DCG_10 / IDCG_10</code>
          </p>
        </div>

        <div className="p-4 rounded-xl bg-cine-bg border border-cine-border space-y-1.5 sm:col-span-2">
          <div className="text-xs font-bold text-emerald-400 uppercase tracking-wider">
            5. Hit Rate@10 (HR@10)
          </div>
          <p className="text-xs text-cine-muted leading-relaxed">
            Binary indicator of user satisfaction:
            <br />
            <code className="text-zinc-300 font-mono text-[11px]">I(|Rec_10 ∩ Rel_u| &gt; 0)</code>
            <br />
            Averaged over all 935 evaluation users. NeuMF achieved 82.03%, outperforming all baseline and matrix factorization models.
          </p>
        </div>
      </div>
    </div>
  );
};
