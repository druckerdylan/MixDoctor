import { useCallback, useState } from 'react';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  disabled?: boolean;
  variant?: 'default' | 'compact';
  label?: string;
}

const SUPPORTED_FORMATS = ['.mp3', '.wav', '.m4a', '.flac', '.aiff', '.aif'];
const MAX_SIZE_MB = 250;

export function FileUpload({ onFileSelect, selectedFile, disabled, variant = 'default', label }: FileUploadProps) {
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validateFile = useCallback((file: File): string | null => {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!SUPPORTED_FORMATS.includes(ext)) {
      return `Unsupported format. Please use: ${SUPPORTED_FORMATS.join(', ')}`;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      return `File too large. Maximum size is ${MAX_SIZE_MB}MB`;
    }
    return null;
  }, []);

  const handleFile = useCallback((file: File) => {
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }
    setError(null);
    onFileSelect(file);
  }, [validateFile, onFileSelect]);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, [disabled, handleFile]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  }, [handleFile]);

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const isCompact = variant === 'compact';

  return (
    <div className="w-full">
      <label
        className={`
          flex flex-col items-center justify-center w-full
          ${isCompact ? 'h-24' : 'h-44'}
          rounded-2xl cursor-pointer
          transition-all duration-200
          ${dragActive
            ? 'bg-mix-primary/10 ring-2 ring-mix-primary/50'
            : 'bg-mix-surface hover:bg-mix-muted'
          }
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        `}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <div className={`flex flex-col items-center justify-center ${isCompact ? 'py-2' : 'pt-5 pb-6'}`}>
          {selectedFile ? (
            <>
              <svg className={`${isCompact ? 'w-6 h-6 mb-1' : 'w-10 h-10 mb-3'} text-mix-success`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className={`${isCompact ? 'text-xs' : 'mb-1 text-sm'} text-gray-300`}>
                <span className="font-semibold">{selectedFile.name}</span>
              </p>
              {!isCompact && <p className="text-xs text-gray-500">{formatFileSize(selectedFile.size)}</p>}
              <p className={`${isCompact ? 'text-xs' : 'mt-2 text-xs'} text-mix-primary`}>Click or drop to change</p>
            </>
          ) : (
            <>
              <svg className={`${isCompact ? 'w-6 h-6 mb-1' : 'w-10 h-10 mb-3'} text-gray-400`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              {isCompact ? (
                <p className="text-xs text-gray-400">{label || 'Click or drop to upload'}</p>
              ) : (
                <>
                  <p className="mb-2 text-sm text-gray-400">
                    <span className="font-semibold">Click to upload</span> or drag and drop
                  </p>
                  <p className="text-xs text-gray-500">
                    MP3, WAV, M4A, FLAC, AIFF (max {MAX_SIZE_MB}MB)
                  </p>
                </>
              )}
            </>
          )}
        </div>
        <input
          type="file"
          className="hidden"
          accept={SUPPORTED_FORMATS.join(',')}
          onChange={handleChange}
          disabled={disabled}
        />
      </label>
      {error && (
        <p className="mt-2 text-sm text-red-400">{error}</p>
      )}
    </div>
  );
}
