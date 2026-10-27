import React from 'react';
import { Heart, ExternalLink, Code } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="bg-cine-secondary border-t border-cine-border/80 pt-12 pb-8 mt-20 text-cine-muted text-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-12">
          {/* Brand Info */}
          <div className="space-y-3 md:col-span-1">
            <span className="text-2xl font-black tracking-wider text-cine-red">
              CINE<span className="text-white">AI</span>
            </span>
            <p className="text-xs text-cine-muted leading-relaxed">
              Discover movies powered by artificial intelligence. A modern recommendation platform comparing 10 state-of-the-art algorithms on MovieLens 100K.
            </p>
            <div className="text-[11px] text-zinc-500">
              Đồ án Môn học Trí tuệ Nhân tạo - K5
            </div>
          </div>

          {/* Model Categories */}
          <div>
            <h4 className="text-xs font-bold uppercase tracking-wider text-white mb-3">AI Model Families</h4>
            <ul className="space-y-1.5 text-xs">
              <li>Baselines (Popularity, TF-IDF Content-Based)</li>
              <li>Collaborative Filtering (User-CF, Item-CF, KNN)</li>
              <li>Matrix Factorization (Funk SVD, SVD++, NMF)</li>
              <li>Deep Learning (Neural CF - NeuMF)</li>
              <li>Weighted Hybrid Recommender</li>
            </ul>
          </div>

          {/* Evaluation Metrics */}
          <div>
            <h4 className="text-xs font-bold uppercase tracking-wider text-white mb-3">Top-N Evaluation</h4>
            <ul className="space-y-1.5 text-xs">
              <li>Precision@10 & Recall@10</li>
              <li>F1-Score@10</li>
              <li>NDCG@10 (Ranking Quality)</li>
              <li>Hit Rate@10 (Coverage Quality)</li>
              <li>Stratified Per-User Test Protocol</li>
            </ul>
          </div>

          {/* Dataset & References */}
          <div>
            <h4 className="text-xs font-bold uppercase tracking-wider text-white mb-3">Datasets & Academic</h4>
            <ul className="space-y-1.5 text-xs">
              <li>
                <a
                  href="https://grouplens.org/datasets/movielens/100k/"
                  target="_blank"
                  rel="noreferrer"
                  className="hover:text-white flex items-center gap-1"
                >
                  MovieLens 100K Dataset <ExternalLink className="w-3 h-3" />
                </a>
              </li>
              <li>100,000 ratings • 943 users • 1,682 movies</li>
              <li>
                <a
                  href="https://github.com/HoangTuan1301/AI-Recommendation-System"
                  target="_blank"
                  rel="noreferrer"
                  className="hover:text-white flex items-center gap-1"
                >
                  GitHub Repository <ExternalLink className="w-3 h-3" />
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom copyright */}
        <div className="pt-6 border-t border-cine-border flex flex-col sm:flex-row items-center justify-between text-xs gap-3">
          <div>
            &copy; 2026 CineAI Recommendation System. All research models and datasets belong to their respective authors.
          </div>
          <div className="flex items-center gap-1 text-zinc-500">
            Built with <Heart className="w-3.5 h-3.5 text-cine-red fill-current" /> using React, TypeScript & FastAPI
          </div>
        </div>
      </div>
    </footer>
  );
};
