import { useState } from 'react';
import type { AnalysisResult, Diagnosis, ReferenceComparison } from '../types';
import { QuickWins } from './QuickWins';
import { DetailsSection } from './DetailsSection';
import { TonalProfile } from './TonalProfile';

interface ResultsViewProps {
  result: AnalysisResult;
  onReset: () => void;
}

export function ResultsView({ result, onReset }: ResultsViewProps) {
  const [activeTab, setActiveTab] = useState<'start' | 'details' | 'metrics'>('start');
  const [completedFixes, setCompletedFixes] = useState<Set<number>>(new Set());
  const [engineerExpanded, setEngineerExpanded] = useState(false);

  // Group diagnoses by priority
  const criticalIssues = result.diagnoses.filter(d => d.priority === 1);
  const importantIssues = result.diagnoses.filter(d => d.priority === 2);
  const optionalIssues = result.diagnoses.filter(d => d.priority === 3);

  // Get top 3 actionable fixes (prioritize critical, then important)
  const topFixes = [...criticalIssues, ...importantIssues, ...optionalIssues].slice(0, 3);

  // Calculate estimated improvement
  const estimatedImprovement = topFixes.reduce((acc, fix) => {
    if (fix.priority === 1) return acc + 8;
    if (fix.priority === 2) return acc + 5;
    return acc + 2;
  }, 0);

  const handleFixComplete = (index: number) => {
    setCompletedFixes(prev => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  const estimatedNewScore = Math.min(100, result.health_score +
    Array.from(completedFixes).reduce((acc, idx) => {
      const fix = topFixes[idx];
      if (!fix) return acc;
      if (fix.priority === 1) return acc + 8;
      if (fix.priority === 2) return acc + 5;
      return acc + 2;
    }, 0)
  );

  // Get truncated engineer summary for preview
  const engineerPreview = result.engineer_summary.slice(0, 150) + (result.engineer_summary.length > 150 ? '...' : '');

  return (
    <div className="space-y-8">
      {/* Header - Clean and minimal */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Analysis Complete</h2>
          <p className="text-gray-500 text-sm mt-1">Here's what we found</p>
        </div>
        <button onClick={onReset} className="text-gray-400 hover:text-white transition-colors text-sm">
          Analyze Another
        </button>
      </div>

      {/* START HERE - The Hero Section */}
      <div className="rounded-2xl bg-gradient-to-br from-emerald-950/50 to-mix-surface p-8">
        <div className="flex items-start justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center">
              <svg className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">Start Here</h3>
              <p className="text-emerald-400/70 text-sm">Fix these {topFixes.length} things first</p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-emerald-400">{completedFixes.size}/{topFixes.length}</div>
            <div className="text-xs text-gray-500">+{estimatedImprovement} points potential</div>
          </div>
        </div>

        <div className="space-y-3">
          {topFixes.map((fix, index) => (
            <StartHereFixCard
              key={index}
              fix={fix}
              index={index + 1}
              isCompleted={completedFixes.has(index)}
              onToggleComplete={() => handleFixComplete(index)}
            />
          ))}
        </div>
      </div>

      {/* Health Score + Engineer's Take Row */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Health Score */}
        <div className="bg-mix-surface rounded-2xl p-6">
          <div className="flex items-center gap-5">
            <HealthScoreRing
              score={result.health_score}
              estimatedScore={completedFixes.size > 0 ? estimatedNewScore : undefined}
            />
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <h3 className="text-lg font-semibold text-white">Mix Health</h3>
                <ScoreLabel score={result.health_score} />
              </div>
              <p className="text-sm text-gray-500">
                {getScoreContext(result.health_score)}
              </p>
              {completedFixes.size > 0 && (
                <p className="text-sm text-emerald-400 mt-2">
                  After fixes: <span className="font-semibold">{estimatedNewScore}</span>
                  <span className="text-gray-500 ml-1">(+{estimatedNewScore - result.health_score})</span>
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Engineer's Take - Collapsed by default */}
        <div className="bg-mix-surface rounded-2xl p-6">
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-lg bg-mix-primary/20 flex items-center justify-center flex-shrink-0">
              <svg className="w-4 h-4 text-mix-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
              </svg>
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="text-sm font-medium text-mix-primary mb-2">Engineer's Take</h3>
              <p className="text-gray-300 text-sm leading-relaxed">
                {engineerExpanded ? result.engineer_summary : engineerPreview}
              </p>
              {result.engineer_summary.length > 150 && (
                <button
                  onClick={() => setEngineerExpanded(!engineerExpanded)}
                  className="text-mix-primary/70 hover:text-mix-primary text-sm mt-2 transition-colors"
                >
                  {engineerExpanded ? 'Show less' : 'Read full analysis'}
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Tab Navigation - Cleaner */}
      <div className="flex gap-1 bg-mix-surface rounded-xl p-1">
        <TabButton active={activeTab === 'start'} onClick={() => setActiveTab('start')}>
          Quick Wins
        </TabButton>
        <TabButton active={activeTab === 'details'} onClick={() => setActiveTab('details')}>
          All Issues ({result.diagnoses.length})
        </TabButton>
        <TabButton active={activeTab === 'metrics'} onClick={() => setActiveTab('metrics')}>
          Technical
        </TabButton>
      </div>

      {/* Tab Content */}
      {activeTab === 'start' && (
        <div className="space-y-6">
          {/* Mastering Status */}
          <MasteringStatus ready={result.mastering_ready} notes={result.mastering_notes} />

          {/* Quick Wins */}
          <QuickWins quickWins={result.quick_wins} />

          {/* Reference Comparison */}
          {result.reference_comparison && (
            <ReferenceComparisonCard comparison={result.reference_comparison} />
          )}

          {/* Next Session Focus */}
          {result.next_session_focus && result.next_session_focus.length > 0 && (
            <NextSessionFocus items={result.next_session_focus} />
          )}
        </div>
      )}

      {activeTab === 'details' && (
        <div className="space-y-4">
          {criticalIssues.length > 0 && (
            <IssueSection
              title="Fix First"
              count={criticalIssues.length}
              issues={criticalIssues}
              color="red"
              defaultExpanded={true}
            />
          )}

          {importantIssues.length > 0 && (
            <IssueSection
              title="Should Fix"
              count={importantIssues.length}
              issues={importantIssues}
              color="amber"
              defaultExpanded={false}
            />
          )}

          {optionalIssues.length > 0 && (
            <IssueSection
              title="Nice to Have"
              count={optionalIssues.length}
              issues={optionalIssues}
              color="blue"
              defaultExpanded={false}
            />
          )}
        </div>
      )}

      {activeTab === 'metrics' && (
        <div className="space-y-6">
          <DetailsSection metrics={result.metrics} />
          <TonalProfile
            spectrum={result.metrics.spectrum}
            referenceComparison={result.reference_comparison}
          />
          <MetricsSummary metrics={result.metrics} />
          {result.reference_comparison && (
            <StereoWidthVisualization
              userWidth={result.metrics.stereo_width}
              referenceWidth={result.metrics.stereo_width - result.reference_comparison.stereo_width_delta}
            />
          )}
        </div>
      )}
    </div>
  );
}

function TabButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`flex-1 px-4 py-2.5 text-sm font-medium rounded-lg transition-all ${
        active
          ? 'bg-mix-muted text-white'
          : 'text-gray-500 hover:text-gray-300'
      }`}
    >
      {children}
    </button>
  );
}

function ScoreLabel({ score }: { score: number }) {
  if (score >= 85) return <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/15 text-emerald-400">Release Ready</span>;
  if (score >= 70) return <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-amber-500/15 text-amber-400">Almost There</span>;
  if (score >= 50) return <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-orange-500/15 text-orange-400">Needs Work</span>;
  return <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-red-500/15 text-red-400">Major Issues</span>;
}

function getScoreContext(score: number): string {
  if (score >= 85) return "Pro-level mix. Ready for mastering.";
  if (score >= 70) return "Good foundation. A few fixes away from release.";
  if (score >= 50) return "Typical pro mix: 80-90. Fix issues below to level up.";
  return "Significant issues detected. Start with the fixes above.";
}

interface StartHereFixCardProps {
  fix: Diagnosis;
  index: number;
  isCompleted: boolean;
  onToggleComplete: () => void;
}

function StartHereFixCard({ fix, index: _index, isCompleted, onToggleComplete }: StartHereFixCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const getQuickAction = (suggestion: string): string => {
    const optionMatch = suggestion.match(/OPTION\s*A[^:]*:\s*([^.]+)/i);
    if (optionMatch) return optionMatch[1].trim();
    return suggestion.slice(0, 80) + (suggestion.length > 80 ? '...' : '');
  };

  const confidenceColors = {
    high: 'bg-emerald-500/20 text-emerald-400',
    medium: 'bg-amber-500/20 text-amber-400',
    low: 'bg-gray-500/20 text-gray-400',
  };

  return (
    <div className={`rounded-xl transition-all ${
      isCompleted ? 'bg-emerald-900/20' : 'bg-mix-darker/50'
    }`}>
      <div className="p-4">
        <div className="flex items-start gap-3">
          {/* Checkbox */}
          <button
            onClick={onToggleComplete}
            className={`mt-0.5 w-5 h-5 rounded-full border-2 flex items-center justify-center flex-shrink-0 transition-all ${
              isCompleted
                ? 'bg-emerald-500 border-emerald-500'
                : 'border-gray-600 hover:border-emerald-400'
            }`}
          >
            {isCompleted && (
              <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
              </svg>
            )}
          </button>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${
                fix.priority === 1 ? 'bg-red-500/20 text-red-400' :
                fix.priority === 2 ? 'bg-amber-500/20 text-amber-400' :
                'bg-blue-500/20 text-blue-400'
              }`}>
                {fix.category}
              </span>
              {fix.affected_frequencies && (
                <span className="text-xs text-gray-600">{fix.affected_frequencies}</span>
              )}
              {fix.confidence && (
                <span className={`text-xs px-1.5 py-0.5 rounded ${confidenceColors[fix.confidence]}`}>
                  {fix.confidence}
                </span>
              )}
            </div>
            <p className={`font-medium text-sm ${isCompleted ? 'text-gray-500 line-through' : 'text-white'}`}>
              {fix.issue}
            </p>
            <p className="text-xs text-gray-500 mt-1">{getQuickAction(fix.suggestion)}</p>
          </div>

          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-gray-600 hover:text-gray-400 p-1 transition-colors"
          >
            <svg
              className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>
      </div>

      {isExpanded && (
        <div className="px-4 pb-4">
          <ExpandedFixDetails suggestion={fix.suggestion} diagnosis={fix} />
        </div>
      )}
    </div>
  );
}

interface ExpandedFixDetailsProps {
  suggestion: string;
  diagnosis?: Diagnosis;
}

function ExpandedFixDetails({ suggestion, diagnosis }: ExpandedFixDetailsProps) {
  const parts = parseSuggestion(suggestion);

  return (
    <div className="pt-3 space-y-3 border-t border-gray-800/50 mt-2">
      {/* What it affects & Likely causes */}
      {diagnosis && (diagnosis.what_it_affects || diagnosis.likely_causes) && (
        <div className="grid md:grid-cols-2 gap-2">
          {diagnosis.what_it_affects && (
            <div className="bg-blue-500/10 rounded-lg p-3">
              <p className="text-xs font-medium text-blue-400 mb-1">What it affects</p>
              <p className="text-sm text-gray-300">{diagnosis.what_it_affects}</p>
            </div>
          )}
          {diagnosis.likely_causes && diagnosis.likely_causes.length > 0 && (
            <div className="bg-amber-500/10 rounded-lg p-3">
              <p className="text-xs font-medium text-amber-400 mb-1">Likely causes</p>
              <ul className="text-sm text-gray-300 space-y-1">
                {diagnosis.likely_causes.map((cause, idx) => (
                  <li key={idx} className="flex items-start gap-1.5">
                    <span className="text-amber-500 mt-1">•</span>
                    <span>{cause}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {(parts.applyTo || parts.doNotApplyTo) && (
        <div className="grid md:grid-cols-2 gap-2">
          {parts.applyTo && (
            <div className="bg-emerald-500/10 rounded-lg p-3">
              <p className="text-xs font-medium text-emerald-400 mb-1">Apply to</p>
              <p className="text-sm text-gray-300">{parts.applyTo}</p>
            </div>
          )}
          {parts.doNotApplyTo && (
            <div className="bg-red-500/10 rounded-lg p-3">
              <p className="text-xs font-medium text-red-400 mb-1">Don't apply to</p>
              <p className="text-sm text-gray-300">{parts.doNotApplyTo}</p>
            </div>
          )}
        </div>
      )}

      {parts.options.length > 0 && (
        <div className="space-y-2">
          {parts.options.map((option, idx) => (
            <div key={idx} className="bg-mix-darker rounded-lg p-3">
              <p className="text-xs font-medium text-mix-primary mb-1">{option.name}</p>
              <p className="text-sm text-gray-400 font-mono">{option.details}</p>
            </div>
          ))}
        </div>
      )}

      {parts.target && (
        <div className="bg-purple-500/10 rounded-lg p-3">
          <p className="text-xs font-medium text-purple-400 mb-1">Target</p>
          <p className="text-sm text-gray-300">{parts.target}</p>
        </div>
      )}

      {/* Don't do & Done when */}
      {diagnosis && (diagnosis.dont_do || diagnosis.done_when) && (
        <div className="grid md:grid-cols-2 gap-2">
          {diagnosis.dont_do && (
            <div className="bg-red-500/10 rounded-lg p-3 border border-red-500/20">
              <p className="text-xs font-medium text-red-400 mb-1">Don't do this</p>
              <p className="text-sm text-gray-300">{diagnosis.dont_do}</p>
            </div>
          )}
          {diagnosis.done_when && (
            <div className="bg-emerald-500/10 rounded-lg p-3 border border-emerald-500/20">
              <p className="text-xs font-medium text-emerald-400 mb-1">You're done when</p>
              <p className="text-sm text-gray-300">{diagnosis.done_when}</p>
            </div>
          )}
        </div>
      )}

      {!parts.applyTo && parts.options.length === 0 && !parts.target && !diagnosis?.what_it_affects && (
        <p className="text-sm text-gray-400 leading-relaxed">{suggestion}</p>
      )}
    </div>
  );
}

function parseSuggestion(suggestion: string) {
  const parts: {
    applyTo?: string;
    doNotApplyTo?: string;
    options: Array<{ name: string; details: string }>;
    target?: string;
  } = { options: [] };

  const applyToMatch = suggestion.match(/APPLY TO:\s*([^.]+?)(?:\.|DO NOT|OPTION|TARGET|$)/i);
  if (applyToMatch) parts.applyTo = applyToMatch[1].trim();

  const doNotMatch = suggestion.match(/DO NOT APPLY TO:\s*([^.]+?)(?:\.|OPTION|TARGET|$)/i);
  if (doNotMatch) parts.doNotApplyTo = doNotMatch[1].trim();

  const optionMatches = suggestion.matchAll(/OPTION\s*([A-C])\s*[-–—:]\s*([^:]+?):\s*([^.]+?)(?=OPTION|TARGET|$)/gi);
  for (const match of optionMatches) {
    parts.options.push({
      name: `Option ${match[1]}: ${match[2].trim()}`,
      details: match[3].trim(),
    });
  }

  const targetMatch = suggestion.match(/TARGET:\s*(.+?)(?:$)/i);
  if (targetMatch) parts.target = targetMatch[1].trim();

  return parts;
}

function MasteringStatus({ ready, notes }: { ready: boolean; notes: string }) {
  return (
    <div className={`rounded-2xl p-5 ${ready ? 'bg-emerald-950/30' : 'bg-amber-950/30'}`}>
      <div className="flex items-center gap-4">
        <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${ready ? 'bg-emerald-500/20' : 'bg-amber-500/20'}`}>
          {ready ? (
            <svg className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          ) : (
            <svg className="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          )}
        </div>
        <div>
          <h3 className={`font-medium ${ready ? 'text-emerald-400' : 'text-amber-400'}`}>
            {ready ? 'Ready for Mastering' : 'Not Mastering Ready'}
          </h3>
          <p className="text-sm text-gray-500 mt-0.5">{notes}</p>
        </div>
      </div>
    </div>
  );
}

function HealthScoreRing({ score, estimatedScore }: { score: number; estimatedScore?: number }) {
  const circumference = 2 * Math.PI * 36;
  const strokeDashoffset = circumference - (score / 100) * circumference;
  const estimatedOffset = estimatedScore ? circumference - (estimatedScore / 100) * circumference : undefined;

  const getColor = () => {
    if (score >= 80) return '#34d399';
    if (score >= 60) return '#fbbf24';
    return '#f87171';
  };

  return (
    <div className="relative w-20 h-20 flex-shrink-0">
      <svg className="w-full h-full transform -rotate-90">
        <circle cx="40" cy="40" r="36" stroke="#1a1a2e" strokeWidth="6" fill="none" />
        {estimatedOffset !== undefined && (
          <circle
            cx="40"
            cy="40"
            r="36"
            stroke="#34d399"
            strokeWidth="6"
            fill="none"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={estimatedOffset}
            opacity={0.25}
          />
        )}
        <circle
          cx="40"
          cy="40"
          r="36"
          stroke={getColor()}
          strokeWidth="6"
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          className="transition-all duration-500"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-xl font-bold text-white">{score}</span>
      </div>
    </div>
  );
}

interface IssueSectionProps {
  title: string;
  count: number;
  issues: Diagnosis[];
  color: 'red' | 'amber' | 'blue';
  defaultExpanded: boolean;
}

function IssueSection({ title, count, issues, color, defaultExpanded }: IssueSectionProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  const colorClasses = {
    red: { text: 'text-red-400', badge: 'bg-red-500/15 text-red-400' },
    amber: { text: 'text-amber-400', badge: 'bg-amber-500/15 text-amber-400' },
    blue: { text: 'text-blue-400', badge: 'bg-blue-500/15 text-blue-400' },
  };

  const colors = colorClasses[color];

  return (
    <div className="bg-mix-surface rounded-2xl overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-5 flex items-center justify-between hover:bg-mix-muted/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <h4 className={`font-medium ${colors.text}`}>{title}</h4>
          <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${colors.badge}`}>
            {count}
          </span>
        </div>
        <svg
          className={`w-5 h-5 text-gray-500 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isExpanded && (
        <div className="px-5 pb-5 space-y-2">
          {issues.map((diagnosis, index) => (
            <DiagnosisCard key={index} diagnosis={diagnosis} />
          ))}
        </div>
      )}
    </div>
  );
}

function DiagnosisCard({ diagnosis }: { diagnosis: Diagnosis }) {
  const [isExpanded, setIsExpanded] = useState(false);

  const confidenceColors = {
    high: 'bg-emerald-500/20 text-emerald-400',
    medium: 'bg-amber-500/20 text-amber-400',
    low: 'bg-gray-500/20 text-gray-400',
  };

  return (
    <div className="bg-mix-darker/50 rounded-xl overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-4 text-left hover:bg-mix-darker transition-colors"
      >
        <div className="flex items-start gap-3">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-medium px-2 py-0.5 rounded bg-gray-800 text-gray-400">
                {diagnosis.category}
              </span>
              {diagnosis.affected_frequencies && (
                <span className="text-xs text-gray-600">{diagnosis.affected_frequencies}</span>
              )}
              {diagnosis.confidence && (
                <span className={`text-xs px-1.5 py-0.5 rounded ${confidenceColors[diagnosis.confidence]}`}>
                  {diagnosis.confidence}
                </span>
              )}
            </div>
            <p className="text-white text-sm">{diagnosis.issue}</p>
          </div>
          <svg
            className={`w-4 h-4 text-gray-600 transition-transform flex-shrink-0 mt-1 ${isExpanded ? 'rotate-180' : ''}`}
            fill="none" stroke="currentColor" viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {isExpanded && (
        <div className="px-4 pb-4">
          <ExpandedFixDetails suggestion={diagnosis.suggestion} diagnosis={diagnosis} />
        </div>
      )}
    </div>
  );
}

function StereoWidthVisualization({ userWidth, referenceWidth }: { userWidth: number; referenceWidth: number }) {
  const userPercent = Math.min(100, (userWidth / 1) * 100);
  const refPercent = Math.min(100, (referenceWidth / 1) * 100);
  const gap = refPercent - userPercent;

  return (
    <div className="bg-mix-surface rounded-2xl p-6">
      <h3 className="text-base font-medium text-white mb-5">Stereo Width Comparison</h3>

      <div className="space-y-4">
        <div>
          <div className="flex justify-between text-sm mb-2">
            <span className="text-gray-500">Your Mix</span>
            <span className="text-white font-mono text-sm">{userWidth.toFixed(2)}</span>
          </div>
          <div className="h-2 bg-mix-darker rounded-full overflow-hidden">
            <div
              className="h-full bg-mix-primary rounded-full transition-all"
              style={{ width: `${userPercent}%` }}
            />
          </div>
        </div>

        <div>
          <div className="flex justify-between text-sm mb-2">
            <span className="text-gray-500">Reference</span>
            <span className="text-white font-mono text-sm">{referenceWidth.toFixed(2)}</span>
          </div>
          <div className="h-2 bg-mix-darker rounded-full overflow-hidden">
            <div
              className="h-full bg-purple-500 rounded-full transition-all"
              style={{ width: `${refPercent}%` }}
            />
          </div>
        </div>

        {Math.abs(gap) > 5 && (
          <p className={`text-sm ${gap > 0 ? 'text-purple-400' : 'text-emerald-400'}`}>
            Your mix is {Math.abs(gap).toFixed(0)}% {gap > 0 ? 'narrower' : 'wider'} than reference
          </p>
        )}
      </div>
    </div>
  );
}

function MetricsSummary({ metrics }: { metrics: AnalysisResult['metrics'] }) {
  return (
    <div className="bg-mix-surface rounded-2xl p-6">
      <h3 className="text-base font-medium text-white mb-5">Technical Metrics</h3>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-y-4 gap-x-6">
        <MetricItem label="Integrated LUFS" value={`${metrics.integrated_lufs}`} unit="LUFS" />
        <MetricItem label="True Peak" value={`${metrics.true_peak_dbfs}`} unit="dBFS" />
        <MetricItem label="Loudness Range" value={`${metrics.loudness_range}`} unit="LU" />
        <MetricItem label="Stereo Correlation" value={metrics.stereo_correlation.toFixed(2)} />
        <MetricItem label="Stereo Width" value={metrics.stereo_width.toFixed(2)} />
        <MetricItem label="Crest Factor" value={`${metrics.crest_factor_db}`} unit="dB" />
        <MetricItem label="Dynamic Range" value={`${metrics.dynamic_range_db}`} unit="dB" />
        <MetricItem label="Tempo" value={`${metrics.estimated_tempo}`} unit="BPM" />
        <MetricItem label="Duration" value={`${Math.floor(metrics.duration_seconds / 60)}:${String(Math.floor(metrics.duration_seconds % 60)).padStart(2, '0')}`} />
      </div>

      <div className="mt-6 pt-5 border-t border-gray-800/50">
        <h4 className="text-sm text-gray-500 mb-3">Spectrum Balance</h4>
        <div className="grid grid-cols-6 gap-2">
          {Object.entries(metrics.spectrum).map(([band, value]) => (
            <div key={band} className="text-center">
              <div className="text-xs text-gray-600 capitalize mb-1">{band.replace('_', ' ')}</div>
              <div className="text-sm font-mono text-gray-300">{value}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function MetricItem({ label, value, unit }: { label: string; value: string; unit?: string }) {
  return (
    <div>
      <div className="text-xs text-gray-500 mb-0.5">{label}</div>
      <div className="text-sm text-white">
        <span className="font-mono">{value}</span>
        {unit && <span className="text-gray-500 ml-1 text-xs">{unit}</span>}
      </div>
    </div>
  );
}

function NextSessionFocus({ items }: { items: string[] }) {
  return (
    <div className="bg-gradient-to-br from-mix-primary/10 to-mix-surface rounded-2xl p-6">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-8 h-8 rounded-lg bg-mix-primary/20 flex items-center justify-center">
          <svg className="w-4 h-4 text-mix-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </div>
        <div>
          <h3 className="font-medium text-white">Next Session Focus</h3>
          <p className="text-xs text-gray-500">Re-upload after these changes for a new pass</p>
        </div>
      </div>

      <ol className="space-y-2">
        {items.map((item, index) => (
          <li key={index} className="flex items-start gap-3">
            <span className="flex-shrink-0 w-6 h-6 rounded-full bg-mix-primary/20 text-mix-primary text-sm font-medium flex items-center justify-center">
              {index + 1}
            </span>
            <span className="text-gray-300 text-sm pt-0.5">{item}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function ReferenceComparisonCard({ comparison }: { comparison: ReferenceComparison }) {
  const getSimilarityColor = () => {
    if (comparison.overall_similarity >= 80) return 'text-emerald-400';
    if (comparison.overall_similarity >= 60) return 'text-amber-400';
    return 'text-red-400';
  };

  return (
    <div className="bg-mix-surface rounded-2xl p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center">
            <svg className="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
            </svg>
          </div>
          <h3 className="font-medium text-white">vs Reference Track</h3>
        </div>
        <div className="text-right">
          <div className={`text-2xl font-bold ${getSimilarityColor()}`}>
            {comparison.overall_similarity}%
          </div>
          <div className="text-xs text-gray-500">Match</div>
        </div>
      </div>

      {comparison.key_differences.length > 0 && (
        <ul className="space-y-1.5">
          {comparison.key_differences.slice(0, 3).map((diff, index) => (
            <li key={index} className="flex items-start gap-2 text-sm text-gray-400">
              <span className="text-purple-400 mt-0.5">•</span>
              {diff}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
