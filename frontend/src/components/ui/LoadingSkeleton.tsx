import React from 'react';

export const MovieCardSkeleton: React.FC = () => {
  return (
    <div className="flex-shrink-0 w-36 sm:w-44 md:w-52 rounded-lg bg-cine-surface border border-cine-border animate-pulse overflow-hidden">
      <div className="w-full aspect-[2/3] bg-cine-hover" />
      <div className="p-3 space-y-2">
        <div className="h-4 bg-cine-hover rounded w-3/4" />
        <div className="h-3 bg-cine-hover rounded w-1/2" />
        <div className="h-3 bg-cine-hover rounded w-2/3" />
      </div>
    </div>
  );
};

export const CarouselSkeleton: React.FC<{ count?: number }> = ({ count = 6 }) => {
  return (
    <div className="my-6 px-4 md:px-12">
      <div className="h-6 w-48 bg-cine-surface rounded mb-4 animate-pulse" />
      <div className="flex gap-4 overflow-hidden">
        {Array.from({ length: count }).map((_, i) => (
          <MovieCardSkeleton key={i} />
        ))}
      </div>
    </div>
  );
};

export const TableSkeleton: React.FC = () => {
  return (
    <div className="w-full space-y-3 p-4 bg-cine-surface rounded-xl border border-cine-border animate-pulse">
      <div className="h-8 bg-cine-hover rounded w-1/3 mb-4" />
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="h-10 bg-cine-hover rounded w-full" />
      ))}
    </div>
  );
};
