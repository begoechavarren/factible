"""
Simple test script to validate LLM judge infrastructure.

Tests that the modular structure works correctly before full refactoring.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

from factible.experiments.evaluator.llm_judge import (
    ClaimQualityJudge,
    QueryRelevanceJudge,
    EvidenceRelevanceJudge,
    ExplanationQualityJudge,
)

# Load environment variables (including OPENAI_API_KEY)
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def test_claim_quality_judge():
    """Test claim quality evaluation."""
    print("\n" + "=" * 60)
    print("TEST 1: Claim Quality Judge")
    print("=" * 60)

    judge = ClaimQualityJudge()

    # Good claim
    claim1 = "The Earth's average temperature has increased by 1.1°C since 1880"
    print(f"\nClaim: {claim1}")
    result1 = judge.evaluate(claim1)
    print(f"  Clarity: {result1.clarity_score:.2f}")
    print(f"  Checkability: {result1.checkability_score:.2f}")
    print(f"  Completeness: {result1.completeness_score:.2f}")
    print(f"  Overall: {result1.overall_quality:.2f}")
    print(f"  Reasoning: {result1.reasoning}")

    # Poor claim
    claim2 = "Climate change is bad"
    print(f"\nClaim: {claim2}")
    result2 = judge.evaluate(claim2)
    print(f"  Clarity: {result2.clarity_score:.2f}")
    print(f"  Checkability: {result2.checkability_score:.2f}")
    print(f"  Completeness: {result2.completeness_score:.2f}")
    print(f"  Overall: {result2.overall_quality:.2f}")
    print(f"  Reasoning: {result2.reasoning}")


def test_query_relevance_judge():
    """Test query relevance evaluation."""
    print("\n" + "=" * 60)
    print("TEST 2: Query Relevance Judge")
    print("=" * 60)

    judge = QueryRelevanceJudge()

    claim = "The Earth's average temperature has increased by 1.1°C since 1880"

    # Good query
    query1 = "global average temperature increase since 1880 scientific data"
    print(f"\nClaim: {claim}")
    print(f"Query: {query1}")
    result1 = judge.evaluate(claim, query1)
    print(f"  Relevance: {result1.relevance_score:.2f}")
    print(f"  Specificity: {result1.specificity_score:.2f}")
    print(f"  Reasoning: {result1.reasoning}")

    # Poor query
    query2 = "climate change"
    print(f"\nClaim: {claim}")
    print(f"Query: {query2}")
    result2 = judge.evaluate(claim, query2)
    print(f"  Relevance: {result2.relevance_score:.2f}")
    print(f"  Specificity: {result2.specificity_score:.2f}")
    print(f"  Reasoning: {result2.reasoning}")


def test_evidence_relevance_judge():
    """Test evidence relevance evaluation."""
    print("\n" + "=" * 60)
    print("TEST 3: Evidence Relevance Judge")
    print("=" * 60)

    judge = EvidenceRelevanceJudge()

    claim = "The Earth's average temperature has increased by 1.1°C since 1880"
    evidence = "According to NASA's climate data, the global average temperature has risen approximately 1.1°C since the late 19th century, with most warming occurring in the past 40 years."
    source = "https://climate.nasa.gov"

    print(f"\nClaim: {claim}")
    print(f"Evidence: {evidence}")
    print(f"Source: {source}")
    result = judge.evaluate(claim, evidence, source)
    print(f"  Relevance: {result.relevance_score:.2f}")
    print(f"  Credibility: {result.credibility_score:.2f}")
    print(f"  Reasoning: {result.reasoning}")


def test_explanation_quality_judge():
    """Test explanation quality evaluation."""
    print("\n" + "=" * 60)
    print("TEST 4: Explanation Quality Judge")
    print("=" * 60)

    judge = ExplanationQualityJudge()

    claim = "The Earth's average temperature has increased by 1.1°C since 1880"
    verdict = "SUPPORTS"
    explanation = "NASA climate data confirms this claim. Their records show the global average temperature has risen approximately 1.1°C since the late 19th century, with most of the warming occurring in the past 40 years. This is consistent with multiple independent scientific measurements."

    print(f"\nClaim: {claim}")
    print(f"Verdict: {verdict}")
    print(f"Explanation: {explanation}")
    result = judge.evaluate(claim, verdict, explanation)
    print(f"  Clarity: {result.clarity_score:.2f}")
    print(f"  Evidence Support: {result.evidence_support_score:.2f}")
    print(f"  Completeness: {result.completeness_score:.2f}")
    print(f"  Overall: {result.overall_quality:.2f}")
    print(f"  Reasoning: {result.reasoning}")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TESTING LLM JUDGE INFRASTRUCTURE")
    print("=" * 60)
    print("\nThis tests the modular evaluation system before full refactoring.")
    print("Using GPT-4o-mini for cost-efficient evaluation.")

    try:
        test_claim_quality_judge()
        test_query_relevance_judge()
        test_evidence_relevance_judge()
        test_explanation_quality_judge()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nThe modular structure works! Ready for full refactoring.")

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
