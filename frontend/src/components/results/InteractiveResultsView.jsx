import { useState, useRef, useMemo, useCallback } from 'react';
import YouTubePlayer from '@components/video/YouTubePlayer';
import { TranscriptPanel } from '@components/transcript/TranscriptPanel';
import { ClaimTooltip } from '@components/transcript/ClaimTooltip';
import { groupTranscriptSegments, findSegmentAtTime } from '@/utils/segmentGrouping';
import { matchClaimsToSegments } from '@/utils/fuzzyMatching';

/**
 * Interactive Results View - Redesigned with centered video and full-width transcript
 */
export function InteractiveResultsView({ result, onReset }) {
  const playerRef = useRef(null);
  const [activeSegmentIndex, setActiveSegmentIndex] = useState(-1);
  const [selectedClaim, setSelectedClaim] = useState(null);
  const [tooltipPosition, setTooltipPosition] = useState(null);

  // Extract data from result
  const transcriptData = result?.transcript_data;
  const claims = result?.claim_reports || [];
  const videoId = transcriptData?.video_id;

  // Group transcript segments for display
  const groupedSegments = useMemo(() => {
    if (!transcriptData?.segments) return [];
    return groupTranscriptSegments(transcriptData.segments);
  }, [transcriptData]);

  // Match claims to transcript segments
  const claimMatches = useMemo(() => {
    if (!claims || !groupedSegments) return new Map();
    return matchClaimsToSegments(claims, groupedSegments);
  }, [claims, groupedSegments]);

  // Handle segment click - seek video to that timestamp
  const handleSegmentClick = useCallback((time) => {
    if (playerRef.current) {
      playerRef.current.seekTo(time);
      playerRef.current.playVideo();
    }
  }, []);

  // Handle claim highlight click - show tooltip
  const handleClaimClick = useCallback((claimIndex, event) => {
    const claim = claims[claimIndex];
    if (!claim) return;

    // Get click position for tooltip
    const rect = event.target.getBoundingClientRect();
    setTooltipPosition({
      top: rect.top,
      left: rect.left + rect.width / 2,
    });
    setSelectedClaim(claim);

    // Seek video to claim location
    const match = claimMatches.get(claimIndex);
    if (match && playerRef.current) {
      const segment = groupedSegments[match.segmentIndex];
      playerRef.current.seekTo(segment.start);
    }
  }, [claims, claimMatches, groupedSegments]);

  // Update active segment based on video playback
  const handleTimeUpdate = useCallback((currentTime) => {
    const segmentIndex = findSegmentAtTime(groupedSegments, currentTime);
    if (segmentIndex !== activeSegmentIndex) {
      setActiveSegmentIndex(segmentIndex);
    }
  }, [groupedSegments, activeSegmentIndex]);

  if (!result || !transcriptData) {
    return null;
  }

  const processedClaims = claims.length;
  const totalSources = claims.reduce((count, report) => count + (report.total_sources || 0), 0);

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-8 px-4 pb-10 pt-2 sm:px-6 lg:px-8">
      {/* Compact Header */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="pixel-text text-xs uppercase tracking-widest text-accent/70">
            Analysis Complete
          </p>
          <h2 className="pixel-text text-lg text-primary sm:text-xl">
            {processedClaims} Claim{processedClaims === 1 ? '' : 's'} • {totalSources} Source
            {totalSources === 1 ? '' : 's'}
          </h2>
        </div>

        <button
          onClick={onReset}
          className="pixel-text self-start rounded-xl border-2 border-primary px-4 py-1.5 text-sm text-primary transition-colors hover:bg-primary hover:text-white sm:self-auto"
        >
          New Check
        </button>
      </div>

      {/* Video Player */}
      {videoId && (
        <div className="w-full self-center sm:max-w-3xl lg:max-w-4xl xl:max-w-[36rem]">
          <YouTubePlayer
            ref={playerRef}
            videoId={videoId}
            onTimeUpdate={handleTimeUpdate}
            className="rounded-3xl shadow-xl ring-1 ring-primary/10"
          />
        </div>
      )}

      {/* Transcript */}
      <section className="w-full overflow-hidden rounded-3xl border-2 border-primary/20 bg-white/95 shadow-lg">
        <TranscriptPanel
          groupedSegments={groupedSegments}
          claimMatches={claimMatches}
          claims={claims}
          activeSegmentIndex={activeSegmentIndex}
          onSegmentClick={handleSegmentClick}
          onClaimClick={handleClaimClick}
        />
      </section>

      {/* Claim Tooltip */}
      {selectedClaim && (
        <ClaimTooltip
          claim={selectedClaim}
          position={tooltipPosition}
          onClose={() => setSelectedClaim(null)}
        />
      )}
    </div>
  );
}
