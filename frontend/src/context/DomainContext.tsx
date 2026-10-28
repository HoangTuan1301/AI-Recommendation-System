import React, { createContext, useContext, useState, ReactNode } from 'react';

export type AppDomain = 'movies' | 'music';

export interface SampleMusicUser {
  id: string;
  label: string;
  shortId: string;
  trainCount: number;
}

export const SAMPLE_MUSIC_USERS: SampleMusicUser[] = [
  {
    id: '0023d50b3c99c4352585408b1b21f3762655f434',
    label: 'User A — 4 songs (Pop & Hip-Hop)',
    shortId: '0023d',
    trainCount: 4,
  },
  {
    id: '002bbc4812895c84c68d584292bfdd7c349b2739',
    label: 'User B — 12 songs (Rock & Alternative)',
    shortId: '002bb',
    trainCount: 12,
  },
  {
    id: '004821937d53dbc375cda80ef5269cbd47efff4a',
    label: 'User C — 28 songs (High Activity)',
    shortId: '00482',
    trainCount: 28,
  },
  {
    id: 'guest_cold_start_demo_user',
    label: 'Guest Demo (Cold-Start / Unseen)',
    shortId: 'Guest',
    trainCount: 0,
  },
];

interface DomainContextType {
  activeDomain: AppDomain;
  setActiveDomain: (domain: AppDomain) => void;
  musicUserId: string;
  setMusicUserId: (id: string) => void;
  selectedSongId: string | null;
  setSelectedSongId: (id: string | null) => void;
  historyModalOpen: boolean;
  setHistoryModalOpen: (open: boolean) => void;
}

const DomainContext = createContext<DomainContextType | undefined>(undefined);

export const DomainProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [activeDomain, setActiveDomain] = useState<AppDomain>('movies');
  const [musicUserId, setMusicUserId] = useState<string>(SAMPLE_MUSIC_USERS[0].id);
  const [selectedSongId, setSelectedSongId] = useState<string | null>(null);
  const [historyModalOpen, setHistoryModalOpen] = useState<boolean>(false);

  return (
    <DomainContext.Provider
      value={{
        activeDomain,
        setActiveDomain,
        musicUserId,
        setMusicUserId,
        selectedSongId,
        setSelectedSongId,
        historyModalOpen,
        setHistoryModalOpen,
      }}
    >
      {children}
    </DomainContext.Provider>
  );
};

export const useDomain = (): DomainContextType => {
  const context = useContext(DomainContext);
  if (!context) {
    throw new Error('useDomain must be used within a DomainProvider');
  }
  return context;
};
