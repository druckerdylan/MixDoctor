import { AnalysisForm } from './components/AnalysisForm';
import { ResultsView } from './components/ResultsView';
import { StemsResultsView } from './components/StemsResultsView';
import { Header } from './components/Header';
import { useAnalysis } from './hooks/useAnalysis';
import { AuthProvider, useAuth } from './context/AuthContext';
import type { StemType } from './types';

function AppContent() {
  const { status, result, stemsResult, error, analyze, analyzeStems, reset } = useAnalysis();
  const { token } = useAuth();

  // Wrapper to include token in analyze call
  const handleAnalyze = async (file: File, genre: string, reference?: string, referenceFile?: File) => {
    await analyze(file, genre, reference, token, referenceFile);
  };

  // Wrapper for stems analysis
  const handleAnalyzeStems = async (stems: Map<StemType, File>, genre: string) => {
    await analyzeStems(stems, genre, token);
  };

  return (
    <div className="min-h-screen bg-mix-darker">
      {/* Header */}
      <Header />

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 py-8">
        {status === 'complete' && result ? (
          <ResultsView result={result} onReset={reset} />
        ) : status === 'complete' && stemsResult ? (
          <StemsResultsView result={stemsResult} onReset={reset} />
        ) : (
          <div className="max-w-xl mx-auto">
            <div className="text-center mb-10">
              <h2 className="text-2xl font-semibold text-white mb-2">
                Get Expert Feedback on Your Mix
              </h2>
              <p className="text-gray-500 text-sm">
                Upload your track for AI-powered analysis with actionable suggestions
              </p>
            </div>

            <div className="bg-mix-surface rounded-2xl p-6">
              <AnalysisForm onSubmit={handleAnalyze} onSubmitStems={handleAnalyzeStems} status={status} />

              {error && (
                <div className="mt-4 p-4 bg-red-950/30 rounded-xl">
                  <p className="text-red-400 text-sm">{error}</p>
                </div>
              )}
            </div>

            {/* Features */}
            <div className="mt-16 grid md:grid-cols-3 gap-8">
              <FeatureCard
                icon={
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                  </svg>
                }
                title="Genre-Aware"
                description="Tailored to your genre's standards"
              />
              <FeatureCard
                icon={
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                }
                title="Quick Wins"
                description="Fixes that make the biggest impact"
              />
              <FeatureCard
                icon={
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                }
                title="Mastering Ready"
                description="Know when it's ready to send"
              />
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800/30 mt-16">
        <div className="max-w-4xl mx-auto px-4 py-8 text-center text-xs text-gray-600">
          MixDoctor uses advanced audio analysis and AI to provide professional-grade feedback.
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <div className="text-center">
      <div className="inline-flex items-center justify-center w-10 h-10 rounded-xl bg-mix-primary/10 text-mix-primary mb-3">
        {icon}
      </div>
      <h3 className="font-medium text-white text-sm mb-1">{title}</h3>
      <p className="text-xs text-gray-500">{description}</p>
    </div>
  );
}

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App;
