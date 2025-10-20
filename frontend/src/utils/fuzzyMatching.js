/**
 * Fuzzy matching utilities for finding claims within transcript text.
 * Handles paraphrasing, filler words, and text normalization.
 */

/**
 * Normalize text for fuzzy matching by:
 * - Converting to lowercase
 * - Removing punctuation
 * - Removing common filler words
 * - Normalizing numbers and percentages
 * - Collapsing whitespace
 */
function normalizeText(text) {
  if (!text) return '';

  return text
    .toLowerCase()
    .replace(/[,\.!?;:"""''`]/g, '') // Remove punctuation
    .replace(/\b(um|uh|like|you know|well|so|actually|basically|literally)\b/g, '') // Filler words
    .replace(/(\d+)\s*percent/g, '$1%') // Normalize "3 percent" to "3%"
    .replace(/\s+/g, ' ') // Collapse whitespace
    .trim();
}

/**
 * Calculate similarity score between two strings using token-based matching.
 * Returns a score from 0.0 to 1.0, where 1.0 is perfect match.
 */
function calculateSimilarity(str1, str2) {
  const tokens1 = str1.split(' ').filter((t) => t.length > 2); // Ignore very short words
  const tokens2 = str2.split(' ').filter((t) => t.length > 2);

  if (tokens1.length === 0 || tokens2.length === 0) return 0;

  let matchingTokens = 0;
  for (const token1 of tokens1) {
    for (const token2 of tokens2) {
      // Exact match or one contains the other (handles plural, tense variations)
      if (token1 === token2 || token1.includes(token2) || token2.includes(token1)) {
        matchingTokens++;
        break;
      }
    }
  }

  // Jaccard-like similarity: matches / total unique tokens
  const totalTokens = Math.max(tokens1.length, tokens2.length);
  return matchingTokens / totalTokens;
}

// Handle long claims that span multiple segments
const MAX_WINDOW_SEGMENTS = 8;

/**
 * Assemble a text window from consecutive raw transcript segments.
 * @param {Array} segments - Raw transcript segments [{text,start,duration}]
 * @param {number} startIndex - Index of the first segment in the window
 * @param {number} maxSegments - Maximum number of segments to include
 * @returns {Object} { text, start, duration, indices }
 */
function buildSegmentWindow(segments, startIndex, maxSegments = MAX_WINDOW_SEGMENTS) {
  const windowSegments = segments.slice(startIndex, startIndex + maxSegments);
  if (windowSegments.length === 0) {
    return { text: '', start: 0, duration: 0, indices: [] };
  }

  const text = windowSegments.map((segment) => segment.text).join(' ');
  const duration = windowSegments.reduce((total, segment) => total + (segment.duration ?? 0), 0);

  return {
    text,
    start: windowSegments[0].start ?? 0,
    duration,
    indices: windowSegments.map((segment, offset) => startIndex + offset),
  };
}

/**
 * Find the best matching raw transcript window for a claim text.
 * @param {string} claimText - The claim text to find
 * @param {Array} rawSegments - Raw transcript segments as returned by the backend
 * @param {number} threshold - Minimum similarity score (0.0-1.0)
 * @returns {Object|null} {segmentIndex, score, matchType, start, duration, excerpt, indices}
 */
export function findClaimInTranscript(claimText, rawSegments, threshold = 0.45) {
  if (!claimText || !rawSegments || rawSegments.length === 0) {
    return null;
  }

  const normalizedClaim = normalizeText(claimText);
  const claimWords = normalizedClaim.split(' ').filter((w) => w.length > 2);

  let bestMatch = null;
  let bestScore = 0;
  const debugCandidates = [];

  // Try exact substring match first (fastest)
  for (let i = 0; i < rawSegments.length; i++) {
    const window = buildSegmentWindow(rawSegments, i);
    const normalizedSegment = normalizeText(window.text);

    if (normalizedSegment.includes(normalizedClaim)) {
      console.log(`✓ Exact match found for claim at ${window.start}s:`, claimText.substring(0, 50));
      return {
        segmentIndex: i,
        score: 1.0,
        matchType: 'exact',
        start: window.start,
        duration: window.duration,
        excerpt: window.text.trim(),
        indices: window.indices,
      };
    }
  }

  // Try fuzzy matching over rolling windows of segments
  for (let i = 0; i < rawSegments.length; i++) {
    const window = buildSegmentWindow(rawSegments, i);
    const normalizedSegment = normalizeText(window.text);

    if (!normalizedSegment) {
      continue;
    }

    // Quick filter: segment must contain at least some claim words
    const segmentWords = normalizedSegment.split(' ');
    const commonWords = claimWords.filter((word) =>
      segmentWords.some((sw) => sw.includes(word) || word.includes(sw)),
    );

    if (commonWords.length < Math.min(3, Math.ceil(claimWords.length * 0.4))) {
      continue; // Skip if too few words in common
    }

    const score = calculateSimilarity(normalizedClaim, normalizedSegment);

    if (score > bestScore) {
      bestScore = score;
      bestMatch = {
        segmentIndex: i,
        score,
        matchType: 'fuzzy',
        start: window.start,
        duration: window.duration,
        excerpt: window.text.trim(),
        indices: window.indices,
      };
    }

    // Track top candidates for debugging
    if (score > 0.3) {
      debugCandidates.push({
        score,
        start: window.start,
        excerpt: window.text.substring(0, 80),
      });
    }
  }

  // Sort debug candidates by score
  debugCandidates.sort((a, b) => b.score - a.score);

  if (bestMatch) {
    if (bestScore >= threshold) {
      console.log(
        `✓ Fuzzy match found (score: ${bestScore.toFixed(2)}) at ${bestMatch.start}s:`,
        claimText.substring(0, 50)
      );
      return bestMatch;
    }

    if (bestScore >= 0.35) {
      console.log(
        `~ Approximate match found (score: ${bestScore.toFixed(2)}) at ${bestMatch.start}s:`,
        claimText.substring(0, 50)
      );
      return {
        ...bestMatch,
        matchType: 'approx',
      };
    }
  }

  // Log failure details
  console.warn('✗ No match found for claim:', claimText.substring(0, 80));
  console.warn(`  Best score: ${bestScore.toFixed(2)} (threshold: ${threshold})`);
  console.warn('  Top 3 candidates:', debugCandidates.slice(0, 3));

  return null;
}

/**
 * Match multiple claims to transcript segments.
 * @param {Array} claims - Array of claim report objects
 * @param {Array} rawSegments - Raw transcript segments with timestamps
 * @returns {Map} Map of claimIndex -> matchResult
 */
export function matchClaimsToSegments(claims, rawSegments) {
  const matches = new Map();

  claims.forEach((claim, index) => {
    // Handle both claim.text and claim.claim_text (backend uses claim_text)
    const claimText = claim.text || claim.claim_text;

    if (!claimText) {
      console.warn(`Claim ${index} has no text property`);
      return;
    }

    const match = findClaimInTranscript(claimText, rawSegments);
    if (match) {
      matches.set(index, match);
    } else {
      console.warn(`Could not match claim to transcript: "${claimText.substring(0, 50)}..."`);
    }
  });

  return matches;
}
