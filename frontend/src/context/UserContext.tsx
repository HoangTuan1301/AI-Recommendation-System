import React, { createContext, useContext, useState, useEffect } from 'react';

interface UserContextType {
  currentUserId: number;
  setCurrentUserId: (id: number) => void;
  selectedMovieId: number | null;
  setSelectedMovieId: (id: number | null) => void;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

export const UserProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [currentUserId, setCurrentUserIdState] = useState<number>(() => {
    const saved = localStorage.getItem('cineai_user_id');
    return saved ? parseInt(saved, 10) : 1;
  });

  const [selectedMovieId, setSelectedMovieId] = useState<number | null>(null);

  const setCurrentUserId = (id: number) => {
    setCurrentUserIdState(id);
    localStorage.setItem('cineai_user_id', id.toString());
  };

  return (
    <UserContext.Provider
      value={{
        currentUserId,
        setCurrentUserId,
        selectedMovieId,
        setSelectedMovieId,
      }}
    >
      {children}
    </UserContext.Provider>
  );
};

export const useUser = (): UserContextType => {
  const context = useContext(UserContext);
  if (!context) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
};
