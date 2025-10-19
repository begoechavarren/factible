import { useEffect, useRef } from 'react';

/**
 * ClaimTooltip - Floating popover that shows claim analysis
 * Positioned near the clicked highlight on desktop, bottom sheet on mobile
 */
export function ClaimTooltip({ claim, position, onClose }) {
  const tooltipRef = useRef(null);
  const isMobile = window.innerWidth < 768;

  // Close on click outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (tooltipRef.current && !tooltipRef.current.contains(event.target)) {
        onClose();
      }
    };

    // Close on Escape key
    const handleEscape = (event) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleEscape);

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [onClose]);

  if (!claim) return null;

  const stanceColors = {
    supports: 'border-green-500 bg-green-50',
    refutes: 'border-red-500 bg-red-50',
    mixed: 'border-yellow-500 bg-yellow-50',
    unclear: 'border-gray-400 bg-gray-50',
  };

  const stanceBadgeColors = {
    supports: 'bg-green-500 text-white',
    refutes: 'bg-red-500 text-white',
    mixed: 'bg-yellow-500 text-white',
    unclear: 'bg-gray-500 text-white',
  };

  const stanceColor = stanceColors[claim.overall_stance] || stanceColors.unclear;
  const badgeColor = stanceBadgeColors[claim.overall_stance] || stanceBadgeColors.unclear;

  // Mobile: bottom sheet
  if (isMobile) {
    return (
      <div className="fixed inset-0 z-50 bg-black/40 animate-fade-in">
        <div
          ref={tooltipRef}
          className="absolute bottom-0 left-0 right-0 max-h-[80vh] overflow-y-auto rounded-t-3xl border-2 border-primary/30 bg-white shadow-2xl animate-slide-up"
        >
          <div className="sticky top-0 flex items-center justify-between border-b border-primary/10 bg-white/95 px-5 py-4 backdrop-blur">
            <h3 className="pixel-text text-sm uppercase tracking-wider text-primary">
              Claim Analysis
            </h3>
            <button
              onClick={onClose}
              className="pixel-text rounded-lg px-3 py-1 text-gray-500 transition-colors hover:bg-gray-100"
            >
              ✕
            </button>
          </div>

          <div className="p-5 space-y-4">
            <ClaimContent claim={claim} stanceColor={stanceColor} badgeColor={badgeColor} />
          </div>
        </div>
      </div>
    );
  }

  // Desktop: floating tooltip
  const tooltipStyle = position
    ? {
        position: 'fixed',
        top: `${Math.min(position.top + 30, window.innerHeight - 400)}px`,
        left: `${Math.max(20, Math.min(position.left - 200, window.innerWidth - 420))}px`,
      }
    : {};

  return (
    <div
      ref={tooltipRef}
      style={tooltipStyle}
      className="z-50 w-[400px] max-h-[500px] overflow-y-auto rounded-2xl border-2 border-primary/30 bg-white shadow-2xl animate-fade-in"
    >
      <div className="sticky top-0 flex items-center justify-between border-b border-primary/10 bg-white/95 px-4 py-3 backdrop-blur">
        <h3 className="pixel-text text-xs uppercase tracking-wider text-primary">
          Claim Analysis
        </h3>
        <button
          onClick={onClose}
          className="pixel-text rounded-lg px-2 py-1 text-sm text-gray-500 transition-colors hover:bg-gray-100"
        >
          ✕
        </button>
      </div>

      <div className="p-4 space-y-3">
        <ClaimContent claim={claim} stanceColor={stanceColor} badgeColor={badgeColor} />
      </div>
    </div>
  );
}

function ClaimContent({ claim, stanceColor, badgeColor }) {
  return (
    <>
      {/* Claim Text */}
      <div>
        <p className="text-sm leading-relaxed text-gray-800">{claim.claim_text}</p>
        <p className="pixel-text mt-1 text-xs text-gray-400">
          {claim.claim_category} • {Math.round(claim.claim_confidence * 100)}% confidence
        </p>
      </div>

      {/* Stance Badge */}
      <div className="flex items-center gap-2">
        <span className={`pixel-text rounded-lg px-3 py-1 text-xs uppercase ${badgeColor}`}>
          {claim.overall_stance}
        </span>
        <span className="pixel-text text-xs text-gray-500">
          {claim.verdict_confidence} confidence
        </span>
      </div>

      {/* Verdict */}
      <div className={`rounded-xl border-2 p-3 ${stanceColor}`}>
        <p className="pixel-text mb-1 text-xs uppercase tracking-wide text-gray-700">
          Verdict
        </p>
        <p className="text-sm leading-relaxed text-gray-800">{claim.verdict_summary}</p>
      </div>

      {/* Evidence Groups */}
      {claim.evidence_by_stance && Object.keys(claim.evidence_by_stance).length > 0 && (
        <div className="space-y-2">
          <p className="pixel-text text-xs uppercase tracking-wide text-gray-500">
            Evidence ({claim.total_sources} sources)
          </p>

          {Object.entries(claim.evidence_by_stance).map(([stance, sources]) => (
            <details key={stance} className="group">
              <summary className="pixel-text cursor-pointer text-xs uppercase tracking-wider text-gray-600 hover:text-primary">
                {stance} ({sources.length}) ▸
              </summary>
              <div className="mt-2 space-y-2 pl-2">
                {sources.slice(0, 3).map((source, index) => (
                  <a
                    key={index}
                    href={source.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block rounded-lg border border-gray-200 bg-white/50 p-2 text-xs transition-all hover:border-primary/40 hover:shadow-sm"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <p className="font-medium text-gray-800 hover:text-primary">
                      {source.title}
                    </p>
                    <p className="pixel-text mt-1 text-xs text-gray-400">
                      {new URL(source.url).hostname}
                    </p>
                  </a>
                ))}
                {sources.length > 3 && (
                  <p className="pixel-text text-xs text-gray-400">
                    +{sources.length - 3} more sources
                  </p>
                )}
              </div>
            </details>
          ))}
        </div>
      )}
    </>
  );
}
