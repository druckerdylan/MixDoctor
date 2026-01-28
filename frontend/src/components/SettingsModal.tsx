import { useState, useEffect, useCallback } from 'react';
import { Modal } from './Modal';
import { useAuth } from '../context/AuthContext';
import { Plugin } from '../types';
import { PLUGIN_DATABASE, MANUFACTURERS, SUBSCRIPTION_MANUFACTURERS } from '../data/plugins';

import { API_BASE } from '../config';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const { token } = useAuth();
  const [plugins, setPlugins] = useState<Plugin[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Selection state
  const [selectedManufacturer, setSelectedManufacturer] = useState<string>('');
  const [selectedPlugins, setSelectedPlugins] = useState<Set<string>>(new Set());
  const [isAdding, setIsAdding] = useState(false);

  // Get plugins for selected manufacturer
  const manufacturerPlugins = selectedManufacturer ? PLUGIN_DATABASE[selectedManufacturer] || [] : [];
  const hasSubscription = SUBSCRIPTION_MANUFACTURERS.includes(selectedManufacturer);

  // Get existing plugin names for this manufacturer
  const existingPluginNames = new Set(
    plugins
      .filter(p => p.manufacturer === selectedManufacturer)
      .map(p => p.name)
  );

  // Available plugins (not already added)
  const availablePlugins = manufacturerPlugins.filter(p => !existingPluginNames.has(p.name));

  // Fetch plugins
  const fetchPlugins = useCallback(async () => {
    if (!token) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/plugins`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (!response.ok) {
        throw new Error('Failed to fetch plugins');
      }

      const data = await response.json();
      setPlugins(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  }, [token]);

  useEffect(() => {
    if (isOpen && token) {
      fetchPlugins();
    }
  }, [isOpen, token, fetchPlugins]);

  // Reset selections when manufacturer changes
  useEffect(() => {
    setSelectedPlugins(new Set());
  }, [selectedManufacturer]);

  // Toggle plugin selection
  const togglePlugin = (pluginName: string) => {
    setSelectedPlugins(prev => {
      const next = new Set(prev);
      if (next.has(pluginName)) {
        next.delete(pluginName);
      } else {
        next.add(pluginName);
      }
      return next;
    });
  };

  // Select all available plugins
  const selectAll = () => {
    setSelectedPlugins(new Set(availablePlugins.map(p => p.name)));
  };

  // Deselect all
  const deselectAll = () => {
    setSelectedPlugins(new Set());
  };

  // Add selected plugins
  const handleAddPlugins = async () => {
    if (!token || !selectedManufacturer || selectedPlugins.size === 0) return;

    setIsAdding(true);
    setError(null);

    try {
      const pluginsToAdd = manufacturerPlugins.filter(p => selectedPlugins.has(p.name));

      for (const pluginData of pluginsToAdd) {
        const response = await fetch(`${API_BASE}/plugins`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            name: pluginData.name,
            category: pluginData.category,
            manufacturer: selectedManufacturer,
          }),
        });

        if (!response.ok) {
          console.warn(`Failed to add ${pluginData.name}`);
        }
      }

      setSelectedPlugins(new Set());
      await fetchPlugins();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsAdding(false);
    }
  };

  // Delete plugin
  const handleDeletePlugin = async (pluginId: number) => {
    if (!token) return;

    try {
      const response = await fetch(`${API_BASE}/plugins/${pluginId}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` },
      });

      if (!response.ok) {
        throw new Error('Failed to delete plugin');
      }

      await fetchPlugins();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    }
  };

  // Group plugins by manufacturer
  const pluginsByManufacturer = plugins.reduce((acc, plugin) => {
    const manufacturer = plugin.manufacturer || 'Other';
    if (!acc[manufacturer]) {
      acc[manufacturer] = [];
    }
    acc[manufacturer].push(plugin);
    return acc;
  }, {} as Record<string, Plugin[]>);

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="My Plugins">
      <div className="space-y-6">
        {/* Add Plugin Section */}
        <div className="space-y-3">
          <div className="text-sm font-medium text-gray-300">Add Plugins</div>

          {/* Manufacturer Select */}
          <div>
            <label className="block text-xs text-gray-500 mb-1">Manufacturer</label>
            <select
              value={selectedManufacturer}
              onChange={(e) => setSelectedManufacturer(e.target.value)}
              className="input-field w-full"
            >
              <option value="">Select a manufacturer...</option>
              {MANUFACTURERS.map((manufacturer) => (
                <option key={manufacturer} value={manufacturer}>
                  {manufacturer}
                </option>
              ))}
            </select>
          </div>

          {/* Plugin Checklist */}
          {selectedManufacturer && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs text-gray-500">
                  Select Plugins ({selectedPlugins.size} selected)
                </label>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={selectAll}
                    className="text-xs text-mix-primary hover:underline"
                    disabled={availablePlugins.length === 0}
                  >
                    Select All
                  </button>
                  <span className="text-gray-600">|</span>
                  <button
                    type="button"
                    onClick={deselectAll}
                    className="text-xs text-gray-400 hover:underline"
                  >
                    Clear
                  </button>
                </div>
              </div>

              {availablePlugins.length === 0 ? (
                <div className="text-center py-4 text-gray-500 text-sm bg-mix-darker rounded-lg">
                  All plugins from {selectedManufacturer} have been added
                </div>
              ) : (
                <div className="max-h-48 overflow-y-auto bg-mix-darker rounded-lg p-2 space-y-1">
                  {availablePlugins.map((plugin) => (
                    <label
                      key={plugin.name}
                      className="flex items-center gap-3 px-2 py-1.5 rounded hover:bg-gray-800 cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={selectedPlugins.has(plugin.name)}
                        onChange={() => togglePlugin(plugin.name)}
                        className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-mix-primary focus:ring-mix-primary focus:ring-offset-0"
                      />
                      <span className="text-white text-sm flex-1">{plugin.name}</span>
                      <span className="text-gray-500 text-xs">{plugin.category}</span>
                    </label>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Add Button */}
          {selectedManufacturer && availablePlugins.length > 0 && (
            <button
              onClick={handleAddPlugins}
              disabled={isAdding || selectedPlugins.size === 0}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {isAdding ? (
                <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
              ) : null}
              Add {selectedPlugins.size} Plugin{selectedPlugins.size !== 1 ? 's' : ''}
            </button>
          )}

          {hasSubscription && selectedManufacturer && (
            <p className="text-xs text-gray-500">
              Tip: {selectedManufacturer} offers a subscription - use "Select All" to add everything.
            </p>
          )}
        </div>

        {error && (
          <div className="p-3 bg-red-900/30 border border-red-500 rounded-lg">
            <p className="text-sm text-red-200">{error}</p>
          </div>
        )}

        {/* Plugin List */}
        <div className="border-t border-gray-700 pt-4">
          <div className="text-sm font-medium text-gray-300 mb-3">
            Your Plugins ({plugins.length})
          </div>

          {isLoading ? (
            <div className="text-center py-4 text-gray-400">Loading...</div>
          ) : plugins.length === 0 ? (
            <div className="text-center py-4 text-gray-500">
              No plugins added yet. Add your plugins so MixDoctor can recommend specific settings.
            </div>
          ) : (
            <div className="space-y-4 max-h-64 overflow-y-auto">
              {Object.entries(pluginsByManufacturer)
                .sort(([a], [b]) => a.localeCompare(b))
                .map(([manufacturer, manufacturerPluginsList]) => (
                  <div key={manufacturer}>
                    <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">
                      {manufacturer} ({manufacturerPluginsList.length})
                    </div>
                    <div className="space-y-1">
                      {manufacturerPluginsList
                        .sort((a, b) => a.name.localeCompare(b.name))
                        .map((plugin) => (
                          <div
                            key={plugin.id}
                            className="flex items-center justify-between px-3 py-2 bg-mix-darker rounded-lg group"
                          >
                            <div>
                              <span className="text-white">{plugin.name}</span>
                              <span className="text-gray-500 text-sm ml-2">
                                {plugin.category}
                              </span>
                            </div>
                            <button
                              onClick={() => handleDeletePlugin(plugin.id)}
                              className="p-1 text-gray-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all"
                              title="Remove plugin"
                            >
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                              </svg>
                            </button>
                          </div>
                        ))}
                    </div>
                  </div>
                ))}
            </div>
          )}
        </div>

        <div className="text-xs text-gray-500 border-t border-gray-700 pt-4">
          MixDoctor will recommend your specific plugins with parameter settings when analyzing your mixes.
        </div>
      </div>
    </Modal>
  );
}
