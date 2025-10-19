import { useRef, useEffect } from 'react';
import { formatTime } from '@/utils/segmentGrouping';

/**
 * TranscriptPanel - Full-width transcript with inline yellow highlight for claims
 * Retro highlighter style for facts
 */
export function TranscriptPanel({
  groupedSegments,
  claimMatches,
  claims,
  activeSegmentIndex,
  onSegmentClick,
  onClaimClick,
}) {
  const segmentRefs = useRef([]);

  // Auto-scroll to active segment
  useEffect(() => {
    if (activeSegmentIndex !== null && activeSegmentIndex !== -1 && segmentRefs.current[activeSegmentIndex]) {
      const element = segmentRefs.current[activeSegmentIndex];
      const rect = element.getBoundingClientRect();
      const isVisible = rect.top >= 0 && rect.bottom <= window.innerHeight;

      if (!isVisible) {
        element.scrollIntoView({
          behavior: 'smooth',
          block: 'nearest',
        });
      }
    }
  }, [activeSegmentIndex]);

  // Build map of segmentIndex -> claimIndex
  const segmentClaims = new Map();
  if (claimMatches) {
    claimMatches.forEach((match, claimIndex) => {
      if (!segmentClaims.has(match.segmentIndex)) {
        segmentClaims.set(match.segmentIndex, []);
      }
      segmentClaims.get(match.segmentIndex).push({
        claimIndex,
        start: match.highlightStart,
        end: match.highlightEnd,
      });
    });
  }

  if (!groupedSegments || groupedSegments.length === 0) {
    return (
      <div className="p-6 text-center text-gray-500 pixel-text">
        No transcript available
      </div>
    );
  }

  return (
    <div className="space-y-6 px-6 pt-4 pb-8 md:px-10">
      {groupedSegments.map((segment, segmentIndex) => {
        const isActive = segmentIndex === activeSegmentIndex;
        const claimsInSegment = segmentClaims.get(segmentIndex) || [];

        return (
          <div
            key={segmentIndex}
            ref={(el) => (segmentRefs.current[segmentIndex] = el)}
            className={`
              group transition-all duration-200 rounded-xl p-5
              ${isActive ? 'bg-primary/5' : ''}
            `}
          >
            <div className="flex items-start gap-4">
              {/* Timestamp Button */}
              <button
                onClick={() => onSegmentClick && onSegmentClick(segment.start)}
                className={`
                  pixel-text text-xs px-3 py-1.5 rounded-lg flex-shrink-0 transition-all
                  ${isActive ? 'bg-primary text-white shadow-md' : 'bg-gray-100 text-gray-600 hover:bg-primary/20'}
                `}
              >
                {formatTime(segment.start)}
              </button>

              {/* Text with Inline Highlights */}
              <p className="text-base leading-relaxed text-gray-800 flex-1">
                {claimsInSegment.length > 0 ? (
                  <SegmentWithHighlights
                    text={segment.text}
                    highlights={claimsInSegment}
                    onClaimClick={onClaimClick}
                  />
                ) : (
                  segment.text
                )}
              </p>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/**
 * Renders text with inline yellow highlights for claims
 * Retro highlighter marker style
 */
function SegmentWithHighlights({ text, highlights, onClaimClick }) {
  // Sort highlights by start position
  const sortedHighlights = [...highlights].sort((a, b) => a.start - b.start);

  const parts = [];
  let lastIndex = 0;

  sortedHighlights.forEach((highlight, idx) => {
    // Add text before highlight
    if (highlight.start > lastIndex) {
      parts.push({
        type: 'text',
        content: text.substring(lastIndex, highlight.start),
      });
    }

    // Add highlighted text
    parts.push({
      type: 'highlight',
      content: text.substring(highlight.start, highlight.end),
      claimIndex: highlight.claimIndex,
    });

    lastIndex = highlight.end;
  });

  // Add remaining text
  if (lastIndex < text.length) {
    parts.push({
      type: 'text',
      content: text.substring(lastIndex),
    });
  }

  return (
    <>
      {parts.map((part, index) => {
        if (part.type === 'text') {
          return <span key={index}>{part.content}</span>;
        }

        // Yellow marker highlight - retro style
        return (
          <mark
            key={index}
            onClick={(e) => onClaimClick && onClaimClick(part.claimIndex, e)}
            className="
              cursor-pointer px-1 mx-0.5 rounded-sm
              bg-yellow-200/70
              border-b-2 border-yellow-400/50
              transition-all duration-200
              hover:bg-yellow-300/80 hover:border-yellow-500
              hover:shadow-sm
              font-medium
            "
            style={{
              textDecoration: 'none',
              boxDecorationBreak: 'clone',
              WebkitBoxDecorationBreak: 'clone',
            }}
          >
            {part.content}
          </mark>
        );
      })}
    </>
  );
}
