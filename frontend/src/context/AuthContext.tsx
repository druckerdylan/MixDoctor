import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import type { User } from '../types';

import { API_BASE } from '../config';

interface AuthContextType {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, username: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Load token from localStorage on mount
  useEffect(() => {
    const storedToken = localStorage.getItem('mixdoctor_token');
    const storedUser = localStorage.getItem('mixdoctor_user');

    if (storedToken && storedUser) {
      setToken(storedToken);
      setUser(JSON.parse(storedUser));
    }
    setIsLoading(false);
  }, []);

  // Persist token and user to localStorage
  const persistAuth = useCallback((newToken: string, newUser: User) => {
    localStorage.setItem('mixdoctor_token', newToken);
    localStorage.setItem('mixdoctor_user', JSON.stringify(newUser));
    setToken(newToken);
    setUser(newUser);
  }, []);

  const clearAuth = useCallback(() => {
    localStorage.removeItem('mixdoctor_token');
    localStorage.removeItem('mixdoctor_user');
    setToken(null);
    setUser(null);
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const response = await fetch(`${API_BASE}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Login failed');
    }

    const data = await response.json();
    persistAuth(data.access_token, data.user);
  }, [persistAuth]);

  const signup = useCallback(async (email: string, username: string, password: string) => {
    const response = await fetch(`${API_BASE}/auth/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, username, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Signup failed');
    }

    const data = await response.json();
    persistAuth(data.access_token, data.user);
  }, [persistAuth]);

  const logout = useCallback(() => {
    clearAuth();
  }, [clearAuth]);

  return (
    <AuthContext.Provider value={{ user, token, isLoading, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
