import type { QuickWin } from '../types';

interface QuickWinsProps {
  quickWins: QuickWin[];
}

export function QuickWins({ quickWins }: QuickWinsProps) {
  const getImpactDot = (impact: string) => {
    switch (impact) {
      case 'high': return 'bg-emerald-400';
      case 'medium': return 'bg-amber-400';
      case 'low': return 'bg-blue-400';
      default: return 'bg-gray-400';
    }
  };

  return (
    <div className="bg-mix-surface rounded-2xl p-6">
      <div className="flex items-center gap-2 mb-4">
        <svg className="w-4 h-4 text-mix-success" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
        <h3 className="text-base font-medium text-white">Quick Wins</h3>
      </div>
      <ul className="space-y-3">
        {quickWins.map((win, index) => (
          <li key={index} className="flex items-start gap-3">
            <span className={`mt-1.5 w-1.5 h-1.5 rounded-full flex-shrink-0 ${getImpactDot(win.impact)}`} />
            <div className="flex-1 min-w-0">
              <p className="text-sm text-gray-300">{win.action}</p>
              {win.plugin_suggestions && win.plugin_suggestions.length > 0 && (
                <p className="text-xs text-gray-600 mt-1">
                  Try: {win.plugin_suggestions.join(', ')}
                </p>
              )}
            </div>
            <span className={`
              text-xs px-2 py-0.5 rounded-full font-medium flex-shrink-0
              ${win.impact === 'high' ? 'bg-emerald-500/15 text-emerald-400' : ''}
              ${win.impact === 'medium' ? 'bg-amber-500/15 text-amber-400' : ''}
              ${win.impact === 'low' ? 'bg-blue-500/15 text-blue-400' : ''}
            `}>
              {win.impact}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
