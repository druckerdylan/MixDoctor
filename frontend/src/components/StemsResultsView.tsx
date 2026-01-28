import { useState } from 'react';
import type { StemsAnalysisResult, StemAnalysis, StemInteraction, Diagnosis } from '../types';
import { STEM_LABELS, PRIORITY_LABELS } from '../types';
import { QuickWins } from './QuickWins';

interface StemsResultsViewProps {
  result: StemsAnalysisResult;
  onReset: () => void;
}

export function StemsResultsView({ result, onReset }: StemsResultsViewProps) {
  const [engineerExpanded, setEngineerExpanded] = useState(false);
  const engineerPreview = result.engineer_summary.slice(0, 150) + (result.engineer_summary.length > 150 ? '...' : '');

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Stems Analysis</h2>
          <p className="text-gray-500 text-sm mt-1">Here's what we found in your stems</p>
        </div>
        <button onClick={onReset} className="text-gray-400 hover:text-white transition-colors text-sm">
          Analyze Another
        </button>
      </div>

      {/* Health Score + Engineer's Take Row */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Health Score */}
        <div className="bg-mix-surface rounded-2xl p-6">
          <div className="flex items-center gap-5">
            <HealthScoreRing score={result.overall_health_score} />
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-white mb-1">Overall Health</h3>
              <p className="text-sm text-gray-500">{result.summary}</p>
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

      {/* Mastering Readiness */}
      <div className={`rounded-2xl p-5 ${result.mastering_ready ? 'bg-emerald-950/30' : 'bg-amber-950/30'}`}>
        <div className="flex items-center gap-4">
          <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${result.mastering_ready ? 'bg-emerald-500/20' : 'bg-amber-500/20'}`}>
            {result.mastering_ready ? (
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
            <h3 className={`font-medium ${result.mastering_ready ? 'text-emerald-400' : 'text-amber-400'}`}>
              {result.mastering_ready ? 'Ready for Mastering' : 'Not Mastering Ready'}
            </h3>
            <p className="text-sm text-gray-500 mt-0.5">{result.mastering_notes}</p>
          </div>
        </div>
      </div>

      {/* Stem Interactions */}
      {result.interactions.length > 0 && (
        <InteractionsCard interactions={result.interactions} />
      )}

      {/* Quick Wins */}
      <QuickWins quickWins={result.quick_wins} />

      {/* Per-Stem Analysis */}
      <div className="space-y-4">
        <h3 className="text-base font-medium text-white">Per-Stem Analysis</h3>
        {result.stem_analyses.map((stemAnalysis) => (
          <StemAnalysisCard key={stemAnalysis.stem_type} analysis={stemAnalysis} />
        ))}
      </div>
    </div>
  );
}

function HealthScoreRing({ score }: { score: number }) {
  const circumference = 2 * Math.PI * 36;
  const strokeDashoffset = circumference - (score / 100) * circumference;

  const getColor = () => {
    if (score >= 80) return '#34d399';
    if (score >= 60) return '#fbbf24';
    return '#f87171';
  };

  return (
    <div className="relative w-20 h-20 flex-shrink-0">
      <svg className="w-full h-full transform -rotate-90">
        <circle cx="40" cy="40" r="36" stroke="#1a1a2e" strokeWidth="6" fill="none" />
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
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-xl font-bold text-white">{score}</span>
      </div>
    </div>
  );
}

function InteractionsCard({ interactions }: { interactions: StemInteraction[] }) {
  const getInteractionIcon = (type: string) => {
    switch (type) {
      case 'frequency_masking':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
          </svg>
        );
      case 'phase_issue':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        );
      case 'balance':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" />
          </svg>
        );
      default:
        return null;
    }
  };

  const getSeverityStyles = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-950/40 text-red-400';
      case 'warning': return 'bg-amber-950/40 text-amber-400';
      default: return 'bg-blue-950/40 text-blue-400';
    }
  };

  return (
    <div className="bg-mix-surface rounded-2xl p-6">
      <div className="flex items-center gap-2 mb-4">
        <h3 className="text-base font-medium text-white">Stem Interactions</h3>
        <span className="text-xs text-gray-500">{interactions.length} issues</span>
      </div>
      <div className="space-y-3">
        {interactions.map((interaction, index) => (
          <div
            key={index}
            className={`rounded-xl p-4 ${getSeverityStyles(interaction.severity)}`}
          >
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 mt-0.5 opacity-70">
                {getInteractionIcon(interaction.interaction_type)}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 text-sm mb-1 flex-wrap">
                  <span className="font-medium text-white">
                    {STEM_LABELS[interaction.stem_a as keyof typeof STEM_LABELS] || interaction.stem_a}
                  </span>
                  <span className="text-gray-500">↔</span>
                  <span className="font-medium text-white">
                    {STEM_LABELS[interaction.stem_b as keyof typeof STEM_LABELS] || interaction.stem_b}
                  </span>
                  <span className="text-xs opacity-70">
                    {interaction.interaction_type.replace('_', ' ')}
                  </span>
                  {interaction.affected_frequencies && (
                    <span className="text-xs opacity-50">{interaction.affected_frequencies}</span>
                  )}
                </div>
                <p className="text-sm text-gray-300">{interaction.description}</p>
                <div className="mt-3 pt-3 border-t border-white/10">
                  <p className="text-xs text-gray-400">{interaction.suggestion}</p>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function StemAnalysisCard({ analysis }: { analysis: StemAnalysis }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const stemLabel = STEM_LABELS[analysis.stem_type as keyof typeof STEM_LABELS] || analysis.stem_type;

  const issueCount = analysis.issues.length;
  const criticalCount = analysis.issues.filter(i => i.severity === 'critical').length;
  const warningCount = analysis.issues.filter(i => i.severity === 'warning').length;

  const getStatusDot = () => {
    if (criticalCount > 0) return 'bg-red-400';
    if (warningCount > 0) return 'bg-amber-400';
    if (issueCount > 0) return 'bg-blue-400';
    return 'bg-emerald-400';
  };

  return (
    <div className="bg-mix-surface rounded-2xl overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-5 flex items-center justify-between hover:bg-mix-muted/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className={`w-2 h-2 rounded-full ${getStatusDot()}`} />
          <div className="text-left">
            <h4 className="font-medium text-white">{stemLabel}</h4>
            <div className="flex items-center gap-2 text-xs text-gray-500">
              {criticalCount > 0 && (
                <span className="text-red-400">{criticalCount} critical</span>
              )}
              {warningCount > 0 && (
                <span className="text-amber-400">{warningCount} warning</span>
              )}
              {issueCount === 0 && (
                <span className="text-emerald-400">No issues</span>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right text-sm">
            <div className="text-xs text-gray-500">LUFS</div>
            <div className="text-white font-mono text-sm">{analysis.metrics.integrated_lufs}</div>
          </div>
          <svg
            className={`w-4 h-4 text-gray-500 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {isExpanded && (
        <div className="px-5 pb-5 border-t border-gray-800/50">
          {/* Metrics Summary */}
          <div className="grid grid-cols-3 gap-4 py-4">
            <div>
              <div className="text-xs text-gray-500 mb-0.5">True Peak</div>
              <div className="text-sm font-mono text-white">{analysis.metrics.true_peak_dbfs} dBFS</div>
            </div>
            <div>
              <div className="text-xs text-gray-500 mb-0.5">Stereo Width</div>
              <div className="text-sm font-mono text-white">{analysis.metrics.stereo_width.toFixed(2)}</div>
            </div>
            <div>
              <div className="text-xs text-gray-500 mb-0.5">Dynamic Range</div>
              <div className="text-sm font-mono text-white">{analysis.metrics.dynamic_range_db} dB</div>
            </div>
          </div>

          {/* Spectrum */}
          <div className="py-4 border-t border-gray-800/50">
            <div className="text-xs text-gray-500 mb-3">Spectrum Balance</div>
            <div className="grid grid-cols-6 gap-2">
              {Object.entries(analysis.metrics.spectrum).map(([band, value]) => (
                <div key={band} className="text-center">
                  <div className="text-xs text-gray-600 capitalize mb-1">{band.replace('_', ' ')}</div>
                  <div className="text-sm font-mono text-gray-300">{value}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Issues */}
          {analysis.issues.length > 0 && (
            <div className="pt-4 border-t border-gray-800/50 space-y-2">
              <div className="text-xs text-gray-500 mb-2">Issues</div>
              {analysis.issues.map((issue, index) => (
                <DiagnosisCard key={index} diagnosis={issue} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function DiagnosisCard({ diagnosis }: { diagnosis: Diagnosis }) {
  const getSeverityStyles = () => {
    switch (diagnosis.severity) {
      case 'critical': return 'bg-red-950/30';
      case 'warning': return 'bg-amber-950/30';
      case 'info': return 'bg-blue-950/30';
      default: return 'bg-mix-darker';
    }
  };

  const getPriorityColor = () => {
    const priority = diagnosis.priority as 1 | 2 | 3;
    switch (priority) {
      case 1: return 'bg-red-500/20 text-red-400';
      case 2: return 'bg-amber-500/20 text-amber-400';
      case 3: return 'bg-blue-500/20 text-blue-400';
      default: return 'bg-gray-500/20 text-gray-400';
    }
  };

  const priorityInfo = PRIORITY_LABELS[diagnosis.priority as 1 | 2 | 3] || { label: 'UNKNOWN', description: '' };

  return (
    <div className={`rounded-xl p-4 ${getSeverityStyles()}`}>
      <div className="flex items-center gap-2 text-xs mb-2 flex-wrap">
        <span className={`font-medium px-1.5 py-0.5 rounded ${getPriorityColor()}`}>
          {priorityInfo.label}
        </span>
        <span className="text-gray-500">{diagnosis.category}</span>
        {diagnosis.affected_frequencies && (
          <span className="text-gray-600">{diagnosis.affected_frequencies}</span>
        )}
      </div>
      <p className="text-sm text-white mb-2">{diagnosis.issue}</p>
      <p className="text-xs text-gray-400">{diagnosis.suggestion}</p>
    </div>
  );
}
