/**
 * Groups raw transcript segments into larger, more readable chunks.
 * Uses a hybrid approach: respects sentence boundaries while targeting a character length.
 *
 * @param {Array} rawSegments - Array of {text, start, duration} objects
 * @param {Object} options - Configuration options
 * @param {number} options.targetChars - Target characters per group (default: 150 desktop, 80 mobile)
 * @param {number} options.minDuration - Minimum seconds per group (default: 5)
 * @param {number} options.maxDuration - Maximum seconds per group (default: 20)
 * @returns {Array} Grouped segments with {text, start, duration, segmentIds}
 */
export function groupTranscriptSegments(rawSegments, options = {}) {
  const isMobile = window.innerWidth < 768;
  const {
    targetChars = isMobile ? 80 : 150,
    minDuration = 5,
    maxDuration = 20,
  } = options;

  if (!rawSegments || rawSegments.length === 0) {
    return [];
  }

  const grouped = [];
  let currentGroup = {
    text: '',
    start: rawSegments[0].start,
    duration: 0,
    segmentIds: [],
  };

  for (let i = 0; i < rawSegments.length; i++) {
    const segment = rawSegments[i];

    // Add segment to current group
    if (currentGroup.text) {
      currentGroup.text += ' ' + segment.text;
    } else {
      currentGroup.text = segment.text;
      currentGroup.start = segment.start;
    }

    currentGroup.duration += segment.duration;
    currentGroup.segmentIds.push(i);

    // Check if we should break the group
    const endsWithPunctuation = /[.!?]$/.test(segment.text.trim());
    const isLongEnough = currentGroup.text.length >= targetChars;
    const tooLong = currentGroup.duration >= maxDuration;
    const isLastSegment = i === rawSegments.length - 1;

    // Break conditions:
    // 1. Hit sentence boundary AND long enough
    // 2. Hit max duration (force break)
    // 3. Last segment (flush remaining)
    if ((endsWithPunctuation && isLongEnough) || tooLong || isLastSegment) {
      grouped.push({ ...currentGroup });
      currentGroup = {
        text: '',
        start: 0,
        duration: 0,
        segmentIds: [],
      };
    }
  }

  return grouped;
}

/**
 * Find which grouped segment contains a specific timestamp
 * @param {Array} groupedSegments - Result from groupTranscriptSegments
 * @param {number} time - Time in seconds
 * @returns {number} Index of the segment containing this time, or -1
 */
export function findSegmentAtTime(groupedSegments, time) {
  for (let i = 0; i < groupedSegments.length; i++) {
    const segment = groupedSegments[i];
    const segmentEnd = segment.start + segment.duration;
    if (time >= segment.start && time < segmentEnd) {
      return i;
    }
  }
  return -1;
}

/**
 * Format time in seconds to MM:SS display format
 * @param {number} seconds - Time in seconds
 * @returns {string} Formatted time string (e.g., "1:05", "12:34")
 */
export function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}
