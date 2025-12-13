"""
Test LLM judge evaluation on a single video.

This script tests that the LLM judges are properly integrated into the evaluation pipeline.
"""

from pathlib import Path
from dotenv import load_dotenv

from factible.experiments.evaluator import GroundTruthEvaluator

load_dotenv()


def test_single_video_with_llm_judge():
    """Test LLM judge evaluation on a single video."""
    print("=" * 60)
    print("Testing LLM Judge Integration")
    print("=" * 60)

    # Setup paths
    base_dir = Path(__file__).parent.parent / "data"
    runs_dir = base_dir / "runs" / "20251213_145741_baseline"
    ground_truth_dir = base_dir / "ground_truth"
    output_dir = base_dir / "eval_results" / "test_llm_judge"

    print(f"\nRuns directory: {runs_dir}")
    print(f"Ground truth directory: {ground_truth_dir}")
    print(f"Output directory: {output_dir}")

    # Create evaluator with LLM judge enabled
    evaluator = GroundTruthEvaluator(
        runs_dir=runs_dir,
        ground_truth_dir=ground_truth_dir,
        output_dir=output_dir,
        enable_llm_judge=True,  # Enable LLM judges
    )

    # Find matching videos
    matching_videos = evaluator.find_matching_videos()
    print(f"\nFound {len(matching_videos)} matching videos")

    if not matching_videos:
        print("❌ No matching videos found!")
        return False

    # Test on first video only
    test_video = matching_videos[0]
    print(f"\nTesting LLM judges with video: {test_video}")
    print("-" * 60)

    try:
        result = evaluator.evaluate_video(test_video)

        print("\n" + "=" * 60)
        print("✓ LLM JUDGE TEST PASSED")
        print("=" * 60)
        print(f"Video: {result.video_id}")
        print(f"\nClaim Quality (LLM): {result.claim_extraction.claim_quality_avg:.2f}")
        print(
            f"Evidence Relevance (LLM): {result.evidence_search.evidence_relevance_avg:.2f}"
        )
        print(
            f"Query Relevance (LLM): {result.query_generation.query_claim_relevance_avg:.2f}"
        )
        print("\nOther Metrics:")
        print(f"  Claim F1: {result.claim_extraction.f1_score:.2%}")
        print(f"  Verdict Accuracy: {result.verdict_accuracy.overall_accuracy:.2%}")

        # Verify LLM judges actually ran
        if result.claim_extraction.claim_quality_avg > 0:
            print("\n✓ Claim quality judge ran successfully")
        else:
            print("\n⚠ Warning: Claim quality judge returned 0")

        if result.evidence_search.evidence_relevance_avg > 0:
            print("✓ Evidence relevance judge ran successfully")
        else:
            print("⚠ Warning: Evidence relevance judge returned 0 (may be no evidence)")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_single_video_with_llm_judge()

    if success:
        print("\n" + "=" * 60)
        print("✓✓✓ LLM JUDGE INTEGRATION TEST PASSED ✓✓✓")
        print("=" * 60)
        print("\nYou can now run evaluations with --enable-llm-judge flag!")
        print("\nExample:")
        print("  uv run factible-experiments evaluate \\")
        print(
            "      --runs-dir factible/experiments/data/runs/20251213_145741_baseline \\"
        )
        print("      --enable-llm-judge")
    else:
        print("\n❌ LLM judge integration test failed")
