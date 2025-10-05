#!/usr/bin/env python3
import sys
import requests  # type: ignore[import-untyped]


def test_health():
    """Test the health endpoint."""
    print("Testing health endpoint...")
    response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_fact_check_stream(video_url: str):
    """Test the fact-check streaming endpoint."""
    print(f"Testing fact-check stream for: {video_url}")
    print("=" * 80)

    url = "http://localhost:8000/api/v1/fact-check/stream"
    data = {
        "video_url": video_url,
        "max_claims": 2,
        "max_queries_per_claim": 1,
        "max_results_per_query": 2,
    }

    try:
        with requests.post(url, json=data, stream=True, timeout=300) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        import json

                        update = json.loads(line_str[6:])
                        progress = update.get("progress", 0)
                        message = update.get("message", "")
                        step = update.get("step", "")

                        bar_length = 40
                        filled = int(bar_length * progress / 100)
                        bar = "█" * filled + "░" * (bar_length - filled)

                        print(f"\r[{bar}] {progress}% - {message}", end="", flush=True)

                        if step == "complete":
                            print()
                            print("\n✅ Fact-checking complete!")
                            result = update.get("data", {}).get("result", {})
                            reports = result.get("claim_reports", [])
                            print(f"\nGenerated {len(reports)} fact-check report(s)")
                            break
                        elif step == "error":
                            print()
                            print(f"\n❌ Error: {message}")
                            break

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
        return False

    return True


if __name__ == "__main__":
    print("factible API Test Script")
    print("=" * 80)
    print()

    # Test health
    try:
        test_health()
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("Make sure the API server is running:")
        print("  cd backend")
        print("  python -m factible_api.main")
        sys.exit(1)

    # Test fact-check streaming
    video_url = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "https://www.youtube.com/watch?v=iGkLcqLWxMA"
    )

    success = test_fact_check_stream(video_url)
    sys.exit(0 if success else 1)
