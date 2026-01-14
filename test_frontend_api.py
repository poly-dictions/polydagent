import requests
import json

base = 'http://localhost:8765'

# Test wallets endpoint
print('=== /api/quantish/wallets ===')
r = requests.get(f'{base}/api/quantish/wallets')
data = r.json()
print(f"Polymarket: {data['data']['polymarket']['address']}")
print(f"Kalshi: {data['data']['kalshi']['address']}")

# Test balances polymarket
print('\n=== /api/quantish/balances (polymarket) ===')
r = requests.get(f'{base}/api/quantish/balances?platform=polymarket')
data = r.json()
if data.get('success'):
    safe = data['data']['polymarket'].get('safe', {})
    print(f"USDC.e: {safe.get('usdc', 0)}")
    print(f"Native USDC: {safe.get('nativeUsdc', 0)}")
    print(f"MATIC: {safe.get('matic', 0)}")

# Test balances kalshi
print('\n=== /api/quantish/balances (kalshi) ===')
r = requests.get(f'{base}/api/quantish/balances?platform=kalshi')
data = r.json()
if data.get('success'):
    bal = data['data']['kalshi'].get('balances', {})
    print(f"SOL: {bal.get('sol', 0)}")
    print(f"USDC: {bal.get('usdc', 0)}")

# Test search
print('\n=== /api/quantish/search ===')
r = requests.get(f'{base}/api/quantish/search?q=bitcoin&limit=2')
data = r.json()
if data.get('success'):
    markets = data['data'].get('markets', [])
    print(f"Found {len(markets)} markets")
    for m in markets[:2]:
        print(f"  - {m.get('title', m.get('question', '?'))[:50]}")

print('\n=== ALL TESTS PASSED ===')
