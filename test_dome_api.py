#!/usr/bin/env python3
"""
Test Dome API SDK to understand response format
"""
import os
import json
from dotenv import load_dotenv
from dome_api_sdk import DomeClient

load_dotenv()

DOME_API_KEY = os.getenv('DOME_API_KEY')

print("=" * 60)
print("DOME API TEST")
print("=" * 60)
print(f"API Key: {DOME_API_KEY[:20]}...")
print()

try:
    # Try different initialization methods
    from dome_api_sdk.types import DomeSDKConfig, GetMarketsParams

    config = DomeSDKConfig(api_key=DOME_API_KEY)
    client = DomeClient(config=config)

    # Check available methods
    print(f"Available methods: {[m for m in dir(client) if not m.startswith('_')]}\n")

    # Test 1: Get Polymarket markets
    print("Test 1: Fetching 5 Polymarket markets...")
    params = GetMarketsParams(limit=5, status='open')
    poly_response = client.polymarket.markets.get_markets(params)

    print(f"Response type: {type(poly_response)}")
    if isinstance(poly_response, list):
        print(f"Got {len(poly_response)} markets")
        if poly_response:
            print("\nFirst market keys:")
            print(list(poly_response[0].keys())[:20])
            print("\nFirst market sample:")
            print(json.dumps(poly_response[0], indent=2, default=str)[:800])
    else:
        print(f"Response: {str(poly_response)[:500]}")

    print("\n" + "=" * 60)

    # Test 2: Get Kalshi markets
    print("Test 2: Fetching 10 Kalshi markets...")
    kalshi_params = GetMarketsParams(limit=10, status='open')
    kalshi_response = client.kalshi.markets.get_markets(kalshi_params)

    print(f"Response type: {type(kalshi_response)}")
    if hasattr(kalshi_response, 'markets'):
        markets = kalshi_response.markets
        print(f"Got {len(markets)} markets\n")

        # Show all fields for first market
        if markets:
            m = markets[0]
            print("=== ALL FIELDS FOR FIRST KALSHI MARKET ===")
            for attr in dir(m):
                if not attr.startswith('_'):
                    val = getattr(m, attr)
                    print(f"  {attr}: {val}")

            print("\n=== SAMPLE MARKETS ===")
            for i, m in enumerate(markets[:5]):
                print(f"\n[{i+1}] {m.title[:60]}...")
                print(f"    event_ticker: {m.event_ticker}")
                print(f"    market_ticker: {m.market_ticker}")
                print(f"    status: {m.status}")
                print(f"    last_price: {m.last_price}")
                print(f"    volume: {m.volume}")
                print(f"    volume_24h: {getattr(m, 'volume_24h', 'N/A')}")
                print(f"    start_time: {m.start_time}")
                print(f"    end_time: {m.end_time}")
                print(f"    close_time: {m.close_time}")
                print(f"    result: {m.result}")
    else:
        print(f"Response: {str(kalshi_response)[:500]}")

    print("\n" + "=" * 60)
    print("SUCCESS - Dome API works!")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)
