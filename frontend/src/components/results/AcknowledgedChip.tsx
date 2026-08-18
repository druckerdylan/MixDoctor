/**
 * "You told us this one was deliberate."
 *
 * An acknowledged finding is not deleted and not fixed. It is still measured,
 * still in the report, still on the timeline — it has simply stopped being a
 * work item, because the person who made the record said they meant it. Hiding
 * it would throw away a real measurement; leaving it looking like a fault would
 * be the same argument the question was asked to end.
 *
 * So it gets a chip. Signal green rather than a severity color: this is a
 * confirmed decision, and the severity scale has nothing to say about it.
 */

export interface AcknowledgedChipProps {
  /** Set where the chip stands for several findings at once, as on a grid card. */
  count?: number;
  className?: string;
}

export default function AcknowledgedChip({ count, className = '' }: AcknowledgedChipProps) {
  const many = typeof count === 'number' && count > 1;
  return (
    <span
      className={`sev-chip sev-clean ${className}`.trim()}
      title={
        many
          ? `${count} findings you confirmed were intentional. They stay in the report as observations and no longer count against the score.`
          : 'You confirmed this was intentional. It stays in the report as an observation and no longer counts against the score.'
      }
    >
      <span aria-hidden="true">✓</span>
      {many ? `${count} your call` : 'Your call'}
    </span>
  );
}
