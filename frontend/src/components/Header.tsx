import { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { AuthModal } from './AuthModal';
import { SettingsModal } from './SettingsModal';

export function Header() {
  const { user, isLoading, logout } = useAuth();
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);

  return (
    <>
      <header className="border-b border-gray-800/50">
        <div className="max-w-4xl mx-auto px-4 py-5 flex items-center gap-3">
          {/* Logo */}
          <svg className="w-7 h-7 text-mix-primary" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M9 19V6l12-3v13" />
            <circle cx="6" cy="18" r="3" />
            <circle cx="18" cy="16" r="3" />
          </svg>
          <h1 className="text-lg font-semibold text-white">MixDoctor</h1>
          <span className="text-xs text-gray-500">AI Mix Analysis</span>

          {/* Spacer */}
          <div className="flex-1" />

          {/* Auth UI */}
          {isLoading ? (
            <div className="w-24 h-8 bg-gray-800 rounded animate-pulse" />
          ) : user ? (
            <div className="flex items-center gap-3">
              {/* Plugins Button */}
              <button
                onClick={() => setShowSettingsModal(true)}
                className="flex items-center gap-2 px-3 py-1.5 text-xs text-gray-400 hover:text-white bg-mix-surface hover:bg-mix-muted rounded-lg transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
                Plugins
              </button>

              {/* Username */}
              <span className="text-xs text-gray-500 hidden sm:block">{user.username}</span>

              {/* Logout Button */}
              <button
                onClick={logout}
                className="px-3 py-1.5 text-xs text-gray-400 hover:text-white rounded-lg transition-colors"
              >
                Log Out
              </button>
            </div>
          ) : (
            <button
              onClick={() => setShowAuthModal(true)}
              className="px-4 py-1.5 text-sm font-medium text-white bg-mix-primary hover:bg-mix-primary/90 rounded-lg transition-colors"
            >
              Log In
            </button>
          )}
        </div>
      </header>

      {/* Modals */}
      <AuthModal isOpen={showAuthModal} onClose={() => setShowAuthModal(false)} />
      <SettingsModal isOpen={showSettingsModal} onClose={() => setShowSettingsModal(false)} />
    </>
  );
}
