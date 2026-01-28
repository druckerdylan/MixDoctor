import type { AudioMetrics } from '../types';

interface DetailsSectionProps {
  metrics: AudioMetrics;
}

type StatusType = 'good' | 'warning' | 'critical';

interface DetailCard {
  label: string;
  value: string;
  status: StatusType;
  subtext: string;
}

export function DetailsSection({ metrics }: DetailsSectionProps) {
  const getSampleRateStatus = (): StatusType => {
    if (metrics.sample_rate >= 44100) return 'good';
    return 'warning';
  };

  const getClippingStatus = (): StatusType => {
    return metrics.clipping_severity as StatusType;
  };

  const getTruePeakStatus = (): StatusType => {
    if (metrics.true_peak_dbfs <= -1.0) return 'good';
    if (metrics.true_peak_dbfs <= 0) return 'warning';
    return 'critical';
  };

  const getPhaseStatus = (): StatusType => {
    if (metrics.phase_status === 'good' || metrics.phase_status === 'acceptable') return 'good';
    if (metrics.phase_status === 'warning') return 'warning';
    return 'critical';
  };

  const formatSampleRate = (sr: number): string => {
    return `${(sr / 1000).toFixed(1)} kHz`;
  };

  const cards: DetailCard[] = [
    {
      label: 'Sample Rate',
      value: formatSampleRate(metrics.sample_rate),
      status: getSampleRateStatus(),
      subtext: metrics.sample_rate >= 44100 ? 'CD quality or better' : 'Below CD quality',
    },
    {
      label: 'Clipping',
      value: metrics.has_clipping ? `${metrics.clipped_samples} samples` : 'None',
      status: getClippingStatus(),
      subtext: metrics.has_clipping
        ? `${metrics.clip_percentage.toFixed(3)}% of audio`
        : 'Clean signal',
    },
    {
      label: 'True Peak',
      value: `${metrics.true_peak_dbfs} dBFS`,
      status: getTruePeakStatus(),
      subtext: metrics.true_peak_dbfs <= -1.0 ? 'Safe headroom' : 'May clip on conversion',
    },
    {
      label: 'Mono Compatibility',
      value: metrics.mono_compatible ? 'Compatible' : 'Issues detected',
      status: getPhaseStatus(),
      subtext: metrics.mono_energy_loss_db !== 0
        ? `${metrics.mono_energy_loss_db > 0 ? '+' : ''}${metrics.mono_energy_loss_db} dB in mono`
        : 'No phase issues',
    },
  ];

  return (
    <div className="bg-mix-surface rounded-2xl p-6">
      <h3 className="text-base font-medium text-white mb-5">Details</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {cards.map((card) => (
          <DetailCardItem key={card.label} card={card} />
        ))}
      </div>
    </div>
  );
}

function DetailCardItem({ card }: { card: DetailCard }) {
  const statusColors = {
    good: {
      bg: 'bg-emerald-500/10',
      border: 'border-emerald-500/20',
      icon: 'text-emerald-400',
    },
    warning: {
      bg: 'bg-amber-500/10',
      border: 'border-amber-500/20',
      icon: 'text-amber-400',
    },
    critical: {
      bg: 'bg-red-500/10',
      border: 'border-red-500/20',
      icon: 'text-red-400',
    },
  };

  const colors = statusColors[card.status];

  return (
    <div className={`rounded-xl p-4 ${colors.bg} border ${colors.border}`}>
      <div className="flex items-start justify-between mb-2">
        <span className="text-xs text-gray-500 uppercase tracking-wide">{card.label}</span>
        <StatusIcon status={card.status} className={colors.icon} />
      </div>
      <div className="text-white font-medium text-sm mb-1">{card.value}</div>
      <div className="text-xs text-gray-500">{card.subtext}</div>
    </div>
  );
}

function StatusIcon({ status, className }: { status: StatusType; className: string }) {
  if (status === 'good') {
    return (
      <svg className={`w-4 h-4 ${className}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
      </svg>
    );
  }

  if (status === 'warning') {
    return (
      <svg className={`w-4 h-4 ${className}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    );
  }

  return (
    <svg className={`w-4 h-4 ${className}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
    </svg>
  );
}
