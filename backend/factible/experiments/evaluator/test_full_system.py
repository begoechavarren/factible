"""
Test the full refactored evaluation system on real data.

This script validates that the modular evaluator works correctly
by running it on a single video from the baseline runs.
"""

from pathlib import Path
from dotenv import load_dotenv

from factible.experiments.evaluator import GroundTruthEvaluator


# Load environment variables
load_dotenv()


def test_single_video():
    """Test evaluation on a single video."""
    print("=" * 60)
    print("Testing Modular Evaluator on Real Data")
    print("=" * 60)

    # Setup paths
    base_dir = Path(__file__).parent.parent / "data"
    runs_dir = base_dir / "runs" / "20251213_145741_baseline"
    ground_truth_dir = base_dir / "ground_truth"
    output_dir = base_dir / "eval_results" / "test_modular_system"

    print(f"\nRuns directory: {runs_dir}")
    print(f"Ground truth directory: {ground_truth_dir}")
    print(f"Output directory: {output_dir}")

    # Create evaluator
    evaluator = GroundTruthEvaluator(
        runs_dir=runs_dir,
        ground_truth_dir=ground_truth_dir,
        output_dir=output_dir,
        enable_llm_judge=False,  # Disable for quick test
    )

    # Find matching videos
    matching_videos = evaluator.find_matching_videos()
    print(f"\nFound {len(matching_videos)} matching videos")

    if not matching_videos:
        print("❌ No matching videos found!")
        return False

    # Test on first video only
    test_video = matching_videos[0]
    print(f"\nTesting with video: {test_video}")

    try:
        result = evaluator.evaluate_video(test_video)

        print("\n" + "=" * 60)
        print("✓ TEST PASSED - Single Video Evaluation")
        print("=" * 60)
        print(f"Video: {result.video_id}")
        print(f"Claim Extraction F1: {result.claim_extraction.f1_score:.2%}")
        print(f"Verdict Accuracy: {result.verdict_accuracy.overall_accuracy:.2%}")
        print(f"Processing Time: {result.end_to_end.total_time_seconds:.1f}s")
        print(f"Cost: ${result.end_to_end.total_cost_usd:.4f}")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_full_evaluation():
    """Test full evaluation on all videos."""
    print("\n" + "=" * 60)
    print("Testing Full Evaluation (All Videos)")
    print("=" * 60)

    # Setup paths
    base_dir = Path(__file__).parent.parent / "data"
    runs_dir = base_dir / "runs" / "20251213_145741_baseline"
    ground_truth_dir = base_dir / "ground_truth"
    output_dir = base_dir / "eval_results" / "test_modular_full"

    # Create evaluator
    evaluator = GroundTruthEvaluator(
        runs_dir=runs_dir,
        ground_truth_dir=ground_truth_dir,
        output_dir=output_dir,
        enable_llm_judge=False,
    )

    try:
        results = evaluator.evaluate_all()

        print("\n" + "=" * 60)
        print(f"✓ FULL TEST PASSED - Evaluated {len(results)} Videos")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ FULL TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test 1: Single video
    success1 = test_single_video()

    if success1:
        # Test 2: Full evaluation
        print("\n" + "=" * 60)
        print("Single video test passed! Running full evaluation...")
        print("=" * 60)
        success2 = test_full_evaluation()

        if success2:
            print("\n" + "=" * 60)
            print("✓✓✓ ALL TESTS PASSED ✓✓✓")
            print("=" * 60)
            print("The modular evaluator is working correctly!")
        else:
            print("\n❌ Full evaluation failed")
    else:
        print("\n❌ Single video test failed, skipping full evaluation")
