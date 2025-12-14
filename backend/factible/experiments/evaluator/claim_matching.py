"""
Claim matching utilities for evaluation.

Provides fuzzy and semantic similarity matching between ground truth and system-extracted claims.
"""

from typing import Dict, List
from difflib import SequenceMatcher
import numpy as np

from .models import GroundTruthClaim


def semantic_similarity_match_claims(
    gt_claims: List[GroundTruthClaim],
    system_claims: List,
    threshold: float = 0.7,
    model=None,
) -> Dict[str, List]:
    """
    Match claims using semantic similarity (sentence-transformers).
    Falls back to fuzzy matching if sentence-transformers not available.

    Args:
        gt_claims: Ground truth claims
        system_claims: System extracted claims
        threshold: Similarity threshold for matching
        model: Pre-loaded SentenceTransformer model (optional, will load if not provided)

    Returns:
        {
            "true_positives": [(gt_claim, system_claim, similarity_score), ...],
            "false_positives": [system_claim, ...],
            "false_negatives": [gt_claim, ...],
        }
    """
    try:
        from sentence_transformers import SentenceTransformer, util

        # Use provided model or load new one
        if model is None:
            model = SentenceTransformer("all-MiniLM-L6-v2")

        # Encode all claims
        gt_texts = [c.claim_text for c in gt_claims]
        sys_texts = [c.text for c in system_claims]

        # Handle edge cases: no claims
        if not system_claims:
            return {
                "true_positives": [],
                "false_positives": [],
                "false_negatives": gt_claims,
            }

        if not gt_claims:
            return {
                "true_positives": [],
                "false_positives": system_claims,
                "false_negatives": [],
            }

        gt_embeddings = model.encode(gt_texts, convert_to_tensor=True)
        sys_embeddings = model.encode(sys_texts, convert_to_tensor=True)

        # Compute cosine similarities
        similarities = util.cos_sim(sys_embeddings, gt_embeddings)

        matched_pairs = []
        matched_gt_indices = set()
        matched_sys_indices = set()

        # Greedy matching: find best matches above threshold
        for sys_idx, sys_claim in enumerate(system_claims):
            best_gt_idx = None
            best_score = threshold

            for gt_idx in range(len(gt_claims)):
                if gt_idx in matched_gt_indices:
                    continue

                score = similarities[sys_idx][gt_idx].item()
                if score > best_score:
                    best_score = score
                    best_gt_idx = gt_idx

            if best_gt_idx is not None:
                matched_pairs.append((gt_claims[best_gt_idx], sys_claim, best_score))
                matched_gt_indices.add(best_gt_idx)
                matched_sys_indices.add(sys_idx)

        unmatched_gt = [
            c for i, c in enumerate(gt_claims) if i not in matched_gt_indices
        ]
        unmatched_sys = [
            c for i, c in enumerate(system_claims) if i not in matched_sys_indices
        ]

        return {
            "true_positives": matched_pairs,
            "false_positives": unmatched_sys,
            "false_negatives": unmatched_gt,
        }

    except ImportError:
        print("⚠️  sentence-transformers not installed. Falling back to fuzzy matching.")
        print("   Install with: uv pip install sentence-transformers")
        return fuzzy_match_claims(gt_claims, system_claims, threshold)


def fuzzy_match_claims(
    gt_claims: List[GroundTruthClaim],
    system_claims: List,
    threshold: float = 0.5,
) -> Dict[str, List]:
    """
    Match system-extracted claims to ground truth claims using fuzzy string matching.

    Args:
        gt_claims: Ground truth claims
        system_claims: System extracted claims
        threshold: Minimum similarity score for matching

    Returns:
        {
            "true_positives": [(gt_claim, system_claim), ...],
            "false_positives": [system_claim, ...],  # Extra claims
            "false_negatives": [gt_claim, ...],      # Missed claims
        }
    """
    matched_pairs = []
    unmatched_gt = list(gt_claims)
    unmatched_system = list(system_claims)

    # Find best matches
    for gt_claim in gt_claims:
        best_match = None
        best_score = threshold

        for sys_claim in unmatched_system:
            score = SequenceMatcher(
                None, gt_claim.claim_text.lower(), sys_claim.text.lower()
            ).ratio()

            if score > best_score:
                best_match = sys_claim
                best_score = score

        if best_match:
            matched_pairs.append((gt_claim, best_match))
            unmatched_gt.remove(gt_claim)
            unmatched_system.remove(best_match)

    return {
        "true_positives": matched_pairs,
        "false_positives": unmatched_system,
        "false_negatives": unmatched_gt,
    }


def calculate_mean_average_precision(
    system_claims: List,
    gt_claims: List[GroundTruthClaim],
    matches: Dict[str, List],
) -> float:
    """
    Calculate Mean Average Precision (MAP) for claim ranking quality.

    MAP rewards systems that rank matched claims higher in their output.

    Args:
        system_claims: List of system extracted claims (assumed ordered by importance)
        gt_claims: List of ground truth claims
        matches: Output from fuzzy_match_claims or semantic_similarity_match_claims

    Returns:
        MAP score between 0.0 and 1.0
    """
    if not system_claims or not matches["true_positives"]:
        return 0.0

    # Create a set of matched system claims for quick lookup
    matched_system_claims = {
        sys_claim for _, sys_claim, *_ in matches["true_positives"]
    }

    precisions_at_k = []
    num_matches_so_far = 0

    # Iterate through system claims in order (ranked by importance)
    for k, sys_claim in enumerate(system_claims, start=1):
        if sys_claim in matched_system_claims:
            num_matches_so_far += 1
            precision_at_k = num_matches_so_far / k
            precisions_at_k.append(precision_at_k)

    # MAP is the average of all precisions at positions where matches occurred
    if not precisions_at_k:
        return 0.0

    return np.mean(precisions_at_k)
