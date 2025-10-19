import { useState, useRef, useMemo, useCallback } from 'react';
import YouTubePlayer from '@components/video/YouTubePlayer';
import { formatTime } from '@/utils/segmentGrouping';
import { matchClaimsToSegments } from '@/utils/fuzzyMatching';

/**
 * Interactive Results View - Redesigned with centered video and full-width transcript
 */
export function InteractiveResultsView({ result, onReset }) {
  const playerRef = useRef(null);
  const [activeClaimIndex, setActiveClaimIndex] = useState(-1);

  // Extract data from result
  const transcriptData = result?.transcript_data;
  const claims = result?.claim_reports || [];
  const videoId = transcriptData?.video_id;
  const rawSegments = transcriptData?.segments || [];

  // Match claims to precise transcript timestamps
  const claimMatches = useMemo(() => {
    if (!claims || !rawSegments) return new Map();
    return matchClaimsToSegments(claims, rawSegments);
  }, [claims, rawSegments]);

  // Prepare timeline entries sorted by chronological order when matches exist
  const timelineEntries = useMemo(() => {
    return claims
      .map((claim, index) => ({
        claim,
        index,
        match: claimMatches.get(index) || null,
      }))
      .sort((a, b) => {
        if (a.match && b.match) {
          return a.match.start - b.match.start;
        }
        if (a.match) return -1;
        if (b.match) return 1;
        return a.index - b.index;
      });
  }, [claims, claimMatches]);

  const orderedMatches = useMemo(() => {
    return timelineEntries
      .filter((entry) => entry.match)
      .map((entry) => ({
        claimIndex: entry.index,
        start: entry.match.start,
      }));
  }, [timelineEntries]);

  const handleSeekToTime = useCallback((time) => {
    if (playerRef.current) {
      playerRef.current.seekTo(time);
      playerRef.current.playVideo();
    }
  }, []);

  const handleClaimSeek = useCallback(
    (claimIndex) => {
      const match = claimMatches.get(claimIndex);
      if (!match) {
        return;
      }
      handleSeekToTime(match.start);
      setActiveClaimIndex(claimIndex);
    },
    [claimMatches, handleSeekToTime],
  );

  const handleTimeUpdate = useCallback(
    (currentTime) => {
      if (!orderedMatches.length) {
        return;
      }

      let nextActive = -1;
      for (let i = orderedMatches.length - 1; i >= 0; i--) {
        const match = orderedMatches[i];
        if (currentTime >= match.start - 0.75) {
          nextActive = match.claimIndex;
          break;
        }
      }

      setActiveClaimIndex((prev) => (prev === nextActive ? prev : nextActive));
    },
    [orderedMatches],
  );

  if (!result || !transcriptData) {
    return null;
  }

  const processedClaims = claims.length;
  const totalSources = claims.reduce((count, report) => count + (report.total_sources || 0), 0);

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-8 px-4 pb-10 pt-2 sm:px-6 lg:px-8">
      <div className="relative w-full rounded-3xl border-2 border-primary/30 bg-primary/10 px-6 py-6 shadow-sm backdrop-blur-sm">
        <div className="flex flex-col items-center gap-2 text-center">
          <span className="pixel-text text-sm uppercase tracking-[0.45em] text-accent/70">
            Claims & Evidence
          </span>
          <h2 className="pixel-text text-xl text-primary sm:text-2xl">Fact Check Results</h2>
          <p className="pixel-text text-base text-primary/80">
            {processedClaims} claim{processedClaims === 1 ? '' : 's'} · {totalSources} source{totalSources === 1 ? '' : 's'}
          </p>
        </div>

        <div className="mt-4 flex w-full justify-center sm:hidden">
          <button
            type="button"
            onClick={onReset}
            className="pixel-text rounded-xl border-2 border-primary px-4 py-1.5 text-xs text-primary transition-colors hover:bg-primary hover:text-white"
          >
            New check
          </button>
        </div>

        <button
          type="button"
          onClick={onReset}
          className="pixel-text hidden sm:block sm:absolute sm:right-6 sm:top-6 rounded-xl border-2 border-primary px-4 py-1.5 text-xs text-primary transition-colors hover:bg-primary hover:text-white"
        >
          New check
        </button>
      </div>

      {/* Video Player */}
      {videoId && (
        <div className="w-full max-w-[36rem] self-center">
          <YouTubePlayer
            ref={playerRef}
            videoId={videoId}
            onTimeUpdate={handleTimeUpdate}
            className="rounded-3xl shadow-xl ring-1 ring-primary/10"
          />
        </div>
      )}

      {/* Claim timeline */}
      <section className="w-full space-y-5">
        {timelineEntries.map(({ claim, index, match }) => {
          const isActive = activeClaimIndex === index;
          const hasMatch = Boolean(match);
          const stance = claim.overall_stance || 'unclear';

          return (
            <article
              key={claim.claim_text ?? index}
              className={`rounded-3xl border-2 bg-white/95 p-6 shadow-lg transition-all duration-300 hover:-translate-y-1 hover:shadow-xl ${
                isActive ? 'border-primary/60 shadow-primary/20' : 'border-primary/20'
              }`}
            >
              <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                <div className="flex-1 min-w-0">
                  <p className="pixel-text mb-2 text-xs uppercase tracking-widest text-accent">
                    Claim {index + 1}
                  </p>
                  <p className="text-lg font-semibold leading-relaxed text-gray-900 break-words">“{claim.claim_text}”</p>
                </div>

                <button
                  type="button"
                  onClick={() => (hasMatch ? handleClaimSeek(index) : null)}
                  className={`pixel-text text-xs px-3 py-1.5 rounded-lg transition-all self-start md:self-auto flex-shrink-0 whitespace-nowrap ${
                    hasMatch
                      ? 'bg-primary text-white shadow-md hover:bg-primary/90'
                      : 'bg-gray-100 text-gray-500'
                  }`}
                  disabled={!hasMatch}
                >
                  {hasMatch ? `Jump • ${formatTime(match.start)}` : 'No timestamp'}
                </button>
              </div>

              <div className="mt-4 rounded-2xl border border-primary/10 bg-primary/5 p-4">
                <div className="mb-2 flex flex-wrap items-center gap-2">
                  <span
                    className={`pixel-text rounded-lg px-3 py-1 text-xs uppercase ${
                      stanceStyles[stance] || stanceStyles.unclear
                    }`}
                  >
                    {stance}
                  </span>
                  <span className="pixel-text text-xs text-gray-500">
                    Confidence: {(claim.verdict_confidence || '').toUpperCase()}
                  </span>
                </div>
                <p className="text-sm leading-relaxed text-gray-700">{claim.verdict_summary}</p>
              </div>

              <EvidenceSection evidenceByStance={claim.evidence_by_stance} />
            </article>
          );
        })}
      </section>
    </div>
  );
}

const stanceStyles = {
  supports: 'border-green-400/60 bg-green-50/70 text-green-700',
  refutes: 'border-red-400/60 bg-red-50/70 text-red-700',
  mixed: 'border-yellow-400/70 bg-yellow-50/70 text-yellow-700',
  unclear: 'border-slate-300 bg-slate-50 text-slate-600',
};

function EvidenceSection({ evidenceByStance }) {
  const entries = Object.entries(evidenceByStance || {});

  const allSources = entries.flatMap(([stance, sources]) =>
    sources.map((source) => ({ ...source, stance })),
  );

  if (!allSources.length) {
    return (
      <div className="mt-4 rounded-2xl border border-primary/10 bg-white/80 p-4">
        <p className="pixel-text text-xs uppercase tracking-widest text-gray-500">
          No supporting sources yet
        </p>
      </div>
    );
  }

  return (
    <div className="mt-4 rounded-2xl border border-primary/10 bg-white/80 p-4">
      <p className="pixel-text mb-2 text-xs uppercase tracking-widest text-gray-500">
        Sources: {allSources.length}
      </p>
      <div className="space-y-2">
        {allSources.map((source, idx) => (
          <div
            key={`${source.stance}-${idx}`}
            className="rounded-xl border border-primary/10 bg-white/90 p-3"
          >
            <div className="mb-1 flex flex-wrap items-center gap-2">
              <span className="pixel-text text-[11px] uppercase text-gray-400">
                {source.stance?.toUpperCase()}
              </span>
              <span className="pixel-text text-[11px] uppercase text-gray-400">
                {source.reliability?.rating ?? 'unknown'} reliability
              </span>
            </div>
            <a
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm font-medium text-primary hover:text-primary/80"
            >
              {source.title}
            </a>
            {source.evidence_summary && (
              <p className="mt-1 text-xs text-gray-600">{source.evidence_summary}</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
