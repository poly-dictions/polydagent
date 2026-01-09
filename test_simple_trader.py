"""
Test Simple Trader Setup
Verify everything is configured correctly without placing real trades
"""

import os
import sqlite3
import asyncio
from dotenv import load_dotenv

load_dotenv()


def check_env_vars():
    """Check required environment variables"""
    print("\n" + "="*60)
    print("Checking Environment Variables")
    print("="*60)

    required = {
        "ANTHROPIC_API_KEY": "Claude API for parsing predictions",
        "POLYMARKET_PRIVATE_KEY": "Polymarket trading key",
        "POLYMARKET_FUNDER_ADDRESS": "Polymarket wallet address"
    }

    optional = {
        "POLYMARKET_BUILDER_CODE": "Builder attribution code",
        "POLYMARKET_BUILDERS_KEY": "Builder API key (fallback)"
    }

    all_good = True

    for var, description in required.items():
        value = os.getenv(var)
        if value:
            masked = value[:10] + "..." if len(value) > 10 else value
            print(f"  [OK] {var}: {masked}")
        else:
            print(f"  [MISSING] {var}: {description}")
            all_good = False

    print("\nOptional:")
    for var, description in optional.items():
        value = os.getenv(var)
        if value:
            masked = value[:10] + "..." if len(value) > 10 else value
            print(f"  [OK] {var}: {masked}")
        else:
            print(f"  [-] {var}: not set - {description}")

    return all_good


def check_predictions_db():
    """Check if predictions database exists and has data"""
    print("\n" + "="*60)
    print("Checking Predictions Database")
    print("="*60)

    db_path = "polydictions-agent/calls.db"

    if not os.path.exists(db_path):
        print(f"  [FAIL] Database not found: {db_path}")
        print("  -> Make sure your prediction agent has created this file")
        return False

    print(f"  [OK] Database exists: {db_path}")

    try:
        conn = sqlite3.connect(db_path)

        # Check if calls table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='calls'"
        )
        if not cursor.fetchone():
            print("  [FAIL] 'calls' table not found")
            return False

        print("  [OK] 'calls' table exists")

        # Count total predictions
        total = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        print(f"  [OK] Total predictions: {total}")

        # Count unresolved
        unresolved = conn.execute("SELECT COUNT(*) FROM calls WHERE resolved = FALSE").fetchone()[0]
        print(f"  [OK] Unresolved predictions: {unresolved}")

        # Show recent predictions
        recent = conn.execute("""
            SELECT title, yes_odds_at_call, created_at
            FROM calls
            WHERE resolved = FALSE
            ORDER BY created_at DESC
            LIMIT 3
        """).fetchall()

        if recent:
            print("\n  Recent unresolved predictions:")
            for title, yes_odds, created_at in recent:
                print(f"    - {title[:50]}... ({yes_odds}% YES) - {created_at}")

        conn.close()
        return True

    except Exception as e:
        print(f"  [FAIL] Error reading database: {e}")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print("\n" + "="*60)
    print("Checking Dependencies")
    print("="*60)

    packages = {
        "aiohttp": "HTTP client for API calls",
        "requests": "HTTP client for Polymarket API",
        "py_clob_client": "Polymarket trading client"
    }

    all_good = True

    for package, description in packages.items():
        try:
            __import__(package)
            print(f"  [OK] {package}: installed")
        except ImportError:
            print(f"  [FAIL] {package}: MISSING - {description}")
            all_good = False

    return all_good


async def test_claude_api():
    """Test Claude API connection"""
    print("\n" + "="*60)
    print("Testing Claude API")
    print("="*60)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("  [FAIL] ANTHROPIC_API_KEY not set")
        return False

    try:
        import aiohttp

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }

        data = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "Say 'test successful' in JSON: {\"status\": \"...\"}"}]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=30
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    text = result.get("content", [{}])[0].get("text", "")
                    print(f"  [OK] Claude API working")
                    print(f"  Response: {text[:100]}")
                    return True
                else:
                    error_text = await resp.text()
                    print(f"  [FAIL] Claude API error: {resp.status}")
                    print(f"  {error_text[:200]}")
                    return False

    except Exception as e:
        print(f"  [FAIL] Error calling Claude: {e}")
        return False


def test_polymarket_connection():
    """Test Polymarket API connection (no trading)"""
    print("\n" + "="*60)
    print("Testing Polymarket API")
    print("="*60)

    try:
        import requests

        # Test public API endpoint
        resp = requests.get(
            "https://gamma-api.polymarket.com/events",
            params={"limit": 1},
            timeout=10
        )

        if resp.status_code == 200:
            events = resp.json()
            if events:
                print(f"  [OK] Polymarket API accessible")
                print(f"  Sample market: {events[0].get('title', 'Unknown')[:50]}")
                return True
            else:
                print(f"  [FAIL] No events returned")
                return False
        else:
            print(f"  [FAIL] API error: {resp.status_code}")
            return False

    except Exception as e:
        print(f"  [FAIL] Error connecting to Polymarket: {e}")
        return False


async def main():
    """Run all checks"""
    print("\n" + "="*60)
    print("SIMPLE TRADER - SETUP VERIFICATION")
    print("="*60)

    results = {}

    # Run checks
    results['env'] = check_env_vars()
    results['deps'] = check_dependencies()
    results['db'] = check_predictions_db()
    results['claude'] = await test_claude_api()
    results['polymarket'] = test_polymarket_connection()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for check, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}: {check}")

    all_passed = all(results.values())

    print("\n" + "="*60)
    if all_passed:
        print("[OK] ALL CHECKS PASSED - Ready to trade!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Review settings in simple_trader.py")
        print("  2. Run: python simple_trader.py")
        print("  3. Choose 'once' mode first to test")
    else:
        print("[FAIL] SOME CHECKS FAILED - Fix issues above")
        print("="*60)
        print("\nWhat to do:")

        if not results['env']:
            print("  - Add missing keys to .env file")
            print("    See SIMPLE_TRADER_GUIDE.md for instructions")

        if not results['deps']:
            print("  - Install missing packages:")
            print("    pip install -r requirements.txt")

        if not results['db']:
            print("  - Run your prediction agent to create database")
            print("    Or check path: polydictions-agent/calls.db")

        if not results['claude']:
            print("  - Check ANTHROPIC_API_KEY in .env")
            print("  - Verify API key is valid")

        if not results['polymarket']:
            print("  - Check internet connection")
            print("  - Polymarket API might be down")

    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
