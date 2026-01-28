import { useState, useCallback } from 'react';
import { FileUpload } from './FileUpload';
import type { StemType } from '../types';
import { STEM_LABELS } from '../types';

// Short labels for compact button display
const STEM_SHORT_LABELS: Record<StemType, string> = {
  kick: 'Kick',
  snare_hats: 'Snare/Hats',
  percussion: 'Perc',
  bass: 'Bass',
  vocals: 'Vocals',
  lead: 'Lead',
  pads: 'Pads',
  guitars: 'Guitars',
  fx: 'FX',
  other: 'Other',
};

interface StemUploadProps {
  onStemsChange: (stems: Map<StemType, File>) => void;
  disabled?: boolean;
}

const STEM_TYPES: StemType[] = [
  'kick',
  'snare_hats',
  'percussion',
  'bass',
  'vocals',
  'lead',
  'pads',
  'guitars',
  'fx',
  'other',
];

export function StemUpload({ onStemsChange, disabled }: StemUploadProps) {
  const [stems, setStems] = useState<Map<StemType, File>>(new Map());
  const [enabledStems, setEnabledStems] = useState<Set<StemType>>(new Set());

  const toggleStem = useCallback((stemType: StemType) => {
    setEnabledStems((prev) => {
      const next = new Set(prev);
      if (next.has(stemType)) {
        next.delete(stemType);
        // Also remove the file
        setStems((prevStems) => {
          const nextStems = new Map(prevStems);
          nextStems.delete(stemType);
          onStemsChange(nextStems);
          return nextStems;
        });
      } else {
        next.add(stemType);
      }
      return next;
    });
  }, [onStemsChange]);

  const handleFileSelect = useCallback((stemType: StemType, file: File) => {
    setStems((prev) => {
      const next = new Map(prev);
      next.set(stemType, file);
      onStemsChange(next);
      return next;
    });
  }, [onStemsChange]);

  const uploadedCount = stems.size;

  return (
    <div className="space-y-4">
      {/* Instructions */}
      <div className="text-sm text-gray-400">
        Select which stems you want to upload (minimum 2):
      </div>

      {/* Upload count indicator */}
      <div className={`text-sm ${uploadedCount >= 2 ? 'text-green-400' : 'text-amber-400'}`}>
        {uploadedCount} of {enabledStems.size} stems uploaded
        {uploadedCount < 2 && ' (need at least 2)'}
      </div>

      {/* Stem selection grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2">
        {STEM_TYPES.map((stemType) => (
          <button
            key={stemType}
            type="button"
            onClick={() => toggleStem(stemType)}
            disabled={disabled}
            className={`
              px-2 py-2 rounded-lg text-xs font-medium transition-all text-center truncate
              ${enabledStems.has(stemType)
                ? stems.has(stemType)
                  ? 'bg-green-600/30 border-green-500 text-green-300 border'
                  : 'bg-mix-primary/30 border-mix-primary text-mix-primary border'
                : 'bg-mix-dark/50 border-gray-600 text-gray-400 border hover:border-gray-500'
              }
              ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
            `}
            title={STEM_LABELS[stemType]}
          >
            {STEM_SHORT_LABELS[stemType]}
            {stems.has(stemType) && (
              <svg className="inline w-3 h-3 ml-1 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            )}
          </button>
        ))}
      </div>

      {/* File upload slots for enabled stems */}
      {enabledStems.size > 0 && (
        <div className="space-y-3 mt-4">
          <div className="text-sm font-medium text-gray-300">
            Upload your stems:
          </div>
          <div className="grid gap-3">
            {Array.from(enabledStems).map((stemType) => (
              <div key={stemType} className="flex items-center gap-3">
                <div className="w-32 text-sm text-gray-400 flex-shrink-0">
                  {STEM_LABELS[stemType]}
                </div>
                <div className="flex-1">
                  <FileUpload
                    onFileSelect={(file) => handleFileSelect(stemType, file)}
                    selectedFile={stems.get(stemType) || null}
                    disabled={disabled}
                    variant="compact"
                    label={`Drop ${STEM_LABELS[stemType]} here`}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
