import { useCallback, useEffect, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';

import Shell from './components/Shell';
import Landing from './components/Landing';
import Intake from './components/Intake';
import type { IntakeSubmission } from './components/Intake';
import AnalyzingSequence from './components/AnalyzingSequence';
import ResultsView from './components/results/ResultsView';
import { useAnalysis } from './hooks/useAnalysis';
import { AuthProvider } from './context/AuthContext';

const EASE = [0.16, 1, 0.3, 1] as const;

type Phase = 'intake' | 'analyzing' | 'results';

function scrollToIntake() {
  const target = document.getElementById('intake');
  if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function AppContent() {
  const reduce = useReducedMotion();
  const {
    status,
    analysis,
    error,
    audioUrl,
    analyze,
    applyAnalysis,
    reset,
    engineerStatus,
    engineerError,
    retryEngineer,
  } = useAnalysis();

  /**
   * The last thing the user submitted. Kept in App so the analyzing sequence
   * can name the file and draw its waveform, and so a failed run can put the
   * form back exactly as it was rather than making them re-pick the file.
   */
  const [submission, setSubmission] = useState<IntakeSubmission | null>(null);

  const working = status === 'uploading' || status === 'measuring' || status === 'consulting';
  const phase: Phase = analysis && status === 'complete' ? 'results' : working ? 'analyzing' : 'intake';

  // Each phase is a different page; land at the top of it.
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'auto' });
  }, [phase]);

  const handleSubmit = useCallback(
    (next: IntakeSubmission) => {
      setSubmission(next);
      void analyze({
        file: next.file,
        intent: next.intent,
        genre: next.genre,
        referenceFile: next.referenceFile,
        notes: next.notes,
        separateStems: next.separateStems,
      });
    },
    [analyze],
  );

  const handleReset = useCallback(() => {
    reset();
    setSubmission(null);
  }, [reset]);

  /** Abandon an in-flight run but keep the form contents. */
  const handleCancel = useCallback(() => {
    reset();
  }, [reset]);

  const actions =
    phase === 'results' ? (
      <button type="button" onClick={handleReset} className="btn-ghost whitespace-nowrap px-4 py-2 text-xs">
        New analysis
      </button>
    ) : phase === 'intake' ? (
      <button type="button" onClick={scrollToIntake} className="btn-ghost whitespace-nowrap px-4 py-2 text-xs">
        Analyze a mix
      </button>
    ) : null;

  const fade = {
    initial: reduce ? { opacity: 0 } : { opacity: 0, y: 16 },
    animate: reduce ? { opacity: 1 } : { opacity: 1, y: 0 },
    exit: reduce ? { opacity: 0 } : { opacity: 0, y: -12 },
    transition: { duration: 0.55, ease: EASE },
  };

  return (
    <Shell actions={actions} onHome={phase === 'results' ? handleReset : null}>
      <AnimatePresence mode="wait" initial={false}>
        {phase === 'intake' && (
          <motion.div key="intake" {...fade}>
            <Landing onStart={scrollToIntake} />
            <Intake
              onSubmit={handleSubmit}
              error={status === 'error' ? error : null}
              restore={submission}
            />
          </motion.div>
        )}

        {phase === 'analyzing' && (
          <motion.div key="analyzing" {...fade}>
            <AnalyzingSequence
              status={status}
              filename={submission?.file.name ?? 'Your mix'}
              genre={submission?.genre ?? '—'}
              peaks={submission?.peaks ?? null}
              onCancel={handleCancel}
            />
          </motion.div>
        )}

        {phase === 'results' && analysis && (
          <motion.div key="results" {...fade}>
            <ResultsView
              analysis={analysis}
              audioUrl={audioUrl}
              onReset={handleReset}
              onAnalysisChange={applyAnalysis}
              engineerStatus={engineerStatus}
              engineerError={engineerError}
              onRetryEngineer={retryEngineer}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </Shell>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}
