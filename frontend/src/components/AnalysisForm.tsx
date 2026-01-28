import { useState, useCallback } from 'react';
import { FileUpload } from './FileUpload';
import { StemUpload } from './StemUpload';
import type { AnalysisStatus, AnalysisMode, StemType } from '../types';

interface AnalysisFormProps {
  onSubmit: (file: File, genre: string, reference?: string, referenceFile?: File) => void;
  onSubmitStems: (stems: Map<StemType, File>, genre: string) => void;
  status: AnalysisStatus;
}

const GENRES = [
  'Pop',
  'Rock',
  'Metal',
  'Hip Hop / Trap',
  'R&B / Soul',
  'EDM / Electronic',
  'House',
  'Techno',
  'Drum & Bass',
  'Jazz',
  'Classical',
  'Orchestral / Cinematic',
  'Indie / Alternative',
  'Acoustic',
  'Ambient',
  'Country',
  'Lo-Fi',
  'Other',
];

export function AnalysisForm({ onSubmit, onSubmitStems, status }: AnalysisFormProps) {
  const [mode, setMode] = useState<AnalysisMode>('full_mix');
  const [file, setFile] = useState<File | null>(null);
  const [genre, setGenre] = useState('');
  const [reference, setReference] = useState('');
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [stems, setStems] = useState<Map<StemType, File>>(new Map());

  const isLoading = status === 'uploading' || status === 'analyzing';

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    if (mode === 'full_mix') {
      if (file && genre) {
        onSubmit(file, genre, reference || undefined, referenceFile || undefined);
      }
    } else {
      if (stems.size >= 2 && genre) {
        onSubmitStems(stems, genre);
      }
    }
  }, [mode, file, genre, reference, referenceFile, stems, onSubmit, onSubmitStems]);

  const canSubmit = mode === 'full_mix'
    ? file && genre
    : stems.size >= 2 && genre;

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Analysis Mode Toggle */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Analysis Mode
        </label>
        <div className="flex gap-1 bg-mix-surface rounded-xl p-1">
          <button
            type="button"
            onClick={() => setMode('full_mix')}
            disabled={isLoading}
            className={`flex-1 py-2.5 px-4 rounded-lg font-medium text-sm transition-all ${
              mode === 'full_mix'
                ? 'bg-mix-muted text-white'
                : 'text-gray-500 hover:text-gray-300'
            } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            Full Mix
          </button>
          <button
            type="button"
            onClick={() => setMode('stems')}
            disabled={isLoading}
            className={`flex-1 py-2.5 px-4 rounded-lg font-medium text-sm transition-all ${
              mode === 'stems'
                ? 'bg-mix-muted text-white'
                : 'text-gray-500 hover:text-gray-300'
            } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            Stems
          </button>
        </div>
        <p className="text-xs text-gray-500 mt-2">
          {mode === 'full_mix'
            ? 'Analyze your complete mix file'
            : 'Analyze individual stems for detailed feedback on each element'}
        </p>
      </div>

      {/* Full Mix Upload */}
      {mode === 'full_mix' && (
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Audio File
          </label>
          <FileUpload
            onFileSelect={setFile}
            selectedFile={file}
            disabled={isLoading}
          />
        </div>
      )}

      {/* Stems Upload */}
      {mode === 'stems' && (
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Stems
          </label>
          <StemUpload
            onStemsChange={setStems}
            disabled={isLoading}
          />
        </div>
      )}

      <div>
        <label htmlFor="genre" className="block text-sm font-medium text-gray-300 mb-2">
          Genre
        </label>
        <select
          id="genre"
          value={genre}
          onChange={(e) => setGenre(e.target.value)}
          className="input-field"
          disabled={isLoading}
          required
        >
          <option value="">Select a genre...</option>
          {GENRES.map((g) => (
            <option key={g} value={g}>{g}</option>
          ))}
        </select>
      </div>

      {/* Advanced Options Toggle (only for full mix mode) */}
      {mode === 'full_mix' && (
        <div className="border-t border-gray-800/50 pt-4">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm text-gray-500 hover:text-gray-300 transition-colors"
            disabled={isLoading}
          >
            <svg
              className={`w-4 h-4 transition-transform ${showAdvanced ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
            Advanced Options
          </button>

          {showAdvanced && (
            <div className="mt-4 space-y-4">
              {/* Reference Track Upload */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Reference Track <span className="text-gray-500">(optional)</span>
                </label>
                <p className="text-xs text-gray-500 mb-2">
                  Upload a professional track to compare against your mix
                </p>
                <FileUpload
                  onFileSelect={setReferenceFile}
                  selectedFile={referenceFile}
                  disabled={isLoading}
                  variant="compact"
                  label="Drop reference track here"
                />
              </div>

              {/* Reference Notes */}
              <div>
                <label htmlFor="reference" className="block text-sm font-medium text-gray-300 mb-2">
                  Reference Notes <span className="text-gray-500">(optional)</span>
                </label>
                <input
                  type="text"
                  id="reference"
                  value={reference}
                  onChange={(e) => setReference(e.target.value)}
                  placeholder="e.g., 'Daft Punk - Get Lucky' or 'aiming for warm, punchy sound'"
                  className="input-field"
                  disabled={isLoading}
                />
              </div>
            </div>
          )}
        </div>
      )}

      <button
        type="submit"
        disabled={!canSubmit || isLoading}
        className="btn-primary w-full py-3 text-lg"
      >
        {status === 'uploading' && (
          <>
            <Spinner /> Uploading...
          </>
        )}
        {status === 'analyzing' && (
          <>
            <Spinner /> {mode === 'stems' ? 'Analyzing Stems...' : 'Analyzing Mix...'}
          </>
        )}
        {(status === 'idle' || status === 'complete' || status === 'error') && (
          mode === 'stems' ? 'Analyze My Stems' : 'Analyze My Mix'
        )}
      </button>
    </form>
  );
}

function Spinner() {
  return (
    <svg className="inline w-5 h-5 mr-2 animate-spin" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
    </svg>
  );
}
