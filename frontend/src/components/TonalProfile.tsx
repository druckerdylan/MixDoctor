import type { ReferenceComparison } from '../types';

interface TonalProfileProps {
  spectrum: Record<string, number>;
  referenceComparison?: ReferenceComparison;
}

interface BandInfo {
  key: string;
  label: string;
  range: string;
  color: string;
}

const BAND_INFO: BandInfo[] = [
  { key: 'sub', label: 'Sub', range: '20-60 Hz', color: 'bg-purple-500' },
  { key: 'low', label: 'Low', range: '60-250 Hz', color: 'bg-blue-500' },
  { key: 'low_mid', label: 'Low Mid', range: '250-500 Hz', color: 'bg-cyan-500' },
  { key: 'mid', label: 'Mid', range: '500-2k Hz', color: 'bg-emerald-500' },
  { key: 'presence', label: 'Presence', range: '2-6k Hz', color: 'bg-amber-500' },
  { key: 'air', label: 'Air', range: '6-20k Hz', color: 'bg-rose-500' },
];

export function TonalProfile({ spectrum, referenceComparison }: TonalProfileProps) {
  // Find max value for scaling (use -60 as floor)
  const values = Object.values(spectrum);
  const maxValue = Math.max(...values, -10);
  const minValue = -40; // Floor for visualization

  const normalizeValue = (value: number): number => {
    // Clamp value between min and max
    const clamped = Math.max(minValue, Math.min(maxValue, value));
    // Normalize to 0-100%
    return ((clamped - minValue) / (maxValue - minValue)) * 100;
  };

  const getReferenceValue = (band: string): number | null => {
    if (!referenceComparison) return null;
    const userValue = spectrum[band] ?? -60;
    const delta = referenceComparison.spectrum_comparison[band] ?? 0;
    return userValue - delta;
  };

  return (
    <div className="bg-mix-surface rounded-2xl p-6">
      <div className="flex items-center justify-between mb-5">
        <h3 className="text-base font-medium text-white">Tonal Profile</h3>
        {referenceComparison && (
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-3 h-1.5 rounded-full bg-white/80" />
              <span className="text-gray-500">Your Mix</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-1.5 rounded-full bg-purple-400/60" />
              <span className="text-gray-500">Reference</span>
            </div>
          </div>
        )}
      </div>

      <div className="space-y-3">
        {BAND_INFO.map((band) => {
          const value = spectrum[band.key] ?? -60;
          const refValue = getReferenceValue(band.key);
          const normalizedValue = normalizeValue(value);
          const normalizedRef = refValue !== null ? normalizeValue(refValue) : null;

          return (
            <div key={band.key} className="group">
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-white w-16">{band.label}</span>
                  <span className="text-xs text-gray-600">{band.range}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-mono text-gray-300">{value.toFixed(1)} dB</span>
                  {refValue !== null && (
                    <span className="text-xs text-purple-400/70">
                      (ref: {refValue.toFixed(1)})
                    </span>
                  )}
                </div>
              </div>

              <div className="relative h-3 bg-mix-darker rounded-full overflow-hidden">
                {/* Reference bar (behind main bar) */}
                {normalizedRef !== null && (
                  <div
                    className="absolute inset-y-0 left-0 bg-purple-400/30 rounded-full transition-all duration-300"
                    style={{ width: `${normalizedRef}%` }}
                  />
                )}

                {/* Main bar */}
                <div
                  className={`absolute inset-y-0 left-0 ${band.color} rounded-full transition-all duration-300 opacity-80 group-hover:opacity-100`}
                  style={{ width: `${normalizedValue}%` }}
                />

                {/* Reference marker line */}
                {normalizedRef !== null && (
                  <div
                    className="absolute inset-y-0 w-0.5 bg-purple-300 transition-all duration-300"
                    style={{ left: `${normalizedRef}%` }}
                  />
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Legend for dB scale */}
      <div className="flex justify-between mt-4 text-xs text-gray-600">
        <span>{minValue} dB</span>
        <span>0 dB</span>
      </div>
    </div>
  );
}
