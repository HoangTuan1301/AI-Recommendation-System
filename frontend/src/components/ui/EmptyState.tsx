import React from 'react';
import { Film } from 'lucide-react';

interface EmptyStateProps {
  title?: string;
  description?: string;
  actionText?: string;
  onAction?: () => void;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title = "No movies found",
  description = "Try adjusting your search query, filter options, or selecting another user.",
  actionText,
  onAction,
}) => {
  return (
    <div className="flex flex-col items-center justify-center p-12 text-center bg-cine-surface/40 rounded-2xl border border-cine-border my-8">
      <div className="w-16 h-16 rounded-full bg-cine-hover flex items-center justify-center mb-4 text-cine-muted">
        <Film className="w-8 h-8" />
      </div>
      <h3 className="text-xl font-bold text-white mb-2">{title}</h3>
      <p className="text-cine-muted text-sm max-w-md mb-6">{description}</p>
      {actionText && onAction && (
        <button
          onClick={onAction}
          className="px-5 py-2.5 rounded-lg bg-cine-red hover:bg-cine-red-hover text-white font-medium text-sm transition-all"
        >
          {actionText}
        </button>
      )}
    </div>
  );
};
