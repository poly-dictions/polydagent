"""
Debug script to see Polymarket API structure
"""
import requests
import json

print("Fetching from Polymarket API...")

resp = requests.get(
    "https://gamma-api.polymarket.com/events",
    params={"limit": 5},
    timeout=10
)

if resp.status_code == 200:
    events = resp.json()
    print(f"\nGot {len(events)} events\n")

    if events:
        # Show structure of first event
        print("="*70)
        print("FIRST EVENT STRUCTURE:")
        print("="*70)
        print(json.dumps(events[0], indent=2))

        print("\n" + "="*70)
        print("SUMMARY OF ALL EVENTS:")
        print("="*70)

        for i, event in enumerate(events, 1):
            print(f"\n{i}. {event.get('title', 'No title')[:60]}")
            print(f"   closed: {event.get('closed', 'N/A')}")
            print(f"   active: {event.get('active', 'N/A')}")

            markets = event.get('markets', [])
            print(f"   markets count: {len(markets)}")

            if markets:
                for m in markets:
                    outcome = m.get('outcome', 'N/A')
                    print(f"     - outcome: {outcome}")
    else:
        print("No events returned")
else:
    print(f"API error: {resp.status_code}")
    print(resp.text[:500])
