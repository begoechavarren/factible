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

/**
 * Find exact position of claim text within a segment using normalized matching
 * @param {string} claimText - The claim to find
 * @param {string} segmentText - The segment text to search in
 * @returns {Object|null} {start, end, matchedText} or null
 */
function findExactPosition(claimText, segmentText) {
  const normalizedClaim = normalizeText(claimText);
  const normalizedSegment = normalizeText(segmentText);

  const index = normalizedSegment.indexOf(normalizedClaim);
  if (index !== -1) {
    // Find approximate position in original text
    // This is approximate because normalization changes character positions
    const claimLength = claimText.length;
    const start = Math.max(0, index - 10);
    const end = Math.min(segmentText.length, index + claimLength + 10);
    return {
      start,
      end,
      matchedText: segmentText.substring(start, end),
    };
  }

  // Fallback: try to find the longest common substring
  const claimTokens = claimText.toLowerCase().split(/\s+/);
  const segmentLower = segmentText.toLowerCase();

  for (let i = claimTokens.length; i >= Math.min(3, claimTokens.length); i--) {
    for (let j = 0; j <= claimTokens.length - i; j++) {
      const phrase = claimTokens.slice(j, j + i).join(' ');
      const phraseIndex = segmentLower.indexOf(phrase);
      if (phraseIndex !== -1) {
        return {
          start: phraseIndex,
          end: phraseIndex + phrase.length,
          matchedText: segmentText.substring(phraseIndex, phraseIndex + phrase.length),
        };
      }
    }
  }

  return null;
}

/**
 * Find the best matching segment for a claim text.
 * @param {string} claimText - The claim text to find
 * @param {Array} groupedSegments - Grouped transcript segments
 * @param {number} threshold - Minimum similarity score (0.0-1.0), default 0.6
 * @returns {Object|null} {segmentIndex, score, matchedText, highlightStart, highlightEnd} or null if no match
 */
export function findClaimInTranscript(claimText, groupedSegments, threshold = 0.6) {
  if (!claimText || !groupedSegments || groupedSegments.length === 0) {
    return null;
  }

  const normalizedClaim = normalizeText(claimText);
  const claimWords = normalizedClaim.split(' ').filter((w) => w.length > 2);

  let bestMatch = null;
  let bestScore = 0;

  // Try exact substring match first (fastest)
  for (let i = 0; i < groupedSegments.length; i++) {
    const segment = groupedSegments[i];
    const normalizedSegment = normalizeText(segment.text);

    if (normalizedSegment.includes(normalizedClaim)) {
      const position = findExactPosition(claimText, segment.text);
      return {
        segmentIndex: i,
        score: 1.0,
        matchedText: segment.text,
        matchType: 'exact',
        highlightStart: position?.start || 0,
        highlightEnd: position?.end || segment.text.length,
      };
    }
  }

  // Try fuzzy matching
  for (let i = 0; i < groupedSegments.length; i++) {
    const segment = groupedSegments[i];
    const normalizedSegment = normalizeText(segment.text);

    // Quick filter: segment must contain at least some claim words
    const segmentWords = normalizedSegment.split(' ');
    const commonWords = claimWords.filter((word) =>
      segmentWords.some((sw) => sw.includes(word) || word.includes(sw)),
    );

    if (commonWords.length < Math.min(3, claimWords.length * 0.4)) {
      continue; // Skip if too few words in common
    }

    // Calculate similarity for this segment
    const score = calculateSimilarity(normalizedClaim, normalizedSegment);

    if (score > bestScore) {
      bestScore = score;
      const position = findExactPosition(claimText, segment.text);
      bestMatch = {
        segmentIndex: i,
        score: score,
        matchedText: segment.text,
        matchType: 'fuzzy',
        highlightStart: position?.start || 0,
        highlightEnd: position?.end || segment.text.length,
      };
    }
  }

  // Return best match if it exceeds threshold
  if (bestMatch && bestScore >= threshold) {
    return bestMatch;
  }

  return null;
}

/**
 * Match multiple claims to transcript segments.
 * @param {Array} claims - Array of claim objects with {text, ...} or {claim_text, ...}
 * @param {Array} groupedSegments - Grouped transcript segments
 * @returns {Map} Map of claimIndex -> matchResult
 */
export function matchClaimsToSegments(claims, groupedSegments) {
  const matches = new Map();

  claims.forEach((claim, index) => {
    // Handle both claim.text and claim.claim_text (backend uses claim_text)
    const claimText = claim.text || claim.claim_text;

    if (!claimText) {
      console.warn(`Claim ${index} has no text property`);
      return;
    }

    const match = findClaimInTranscript(claimText, groupedSegments);
    if (match) {
      matches.set(index, match);
    } else {
      console.warn(`Could not match claim to transcript: "${claimText.substring(0, 50)}..."`);
    }
  });

  return matches;
}
