import asyncio, aiohttp, json

URL = 'https://kalshi-mcp-production-7c2c.up.railway.app/mcp'
KEY = 'pk_kalshi_pvttXePc4vzFj1tVPNG3i82a81qB_QE1'

async def call(tool, args={}):
    payload = {'jsonrpc': '2.0', 'method': 'tools/call', 'params': {'name': tool, 'arguments': args}, 'id': 1}
    headers = {'Content-Type': 'application/json', 'x-api-key': KEY}
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(URL, json=payload, headers=headers) as resp:
            data = await resp.json()
            return json.loads(data.get('result', {}).get('content', [{}])[0].get('text', '{}'))

async def main():
    # 1. Search for a market
    print('=== SEARCHING MARKETS ===')
    result = await call('kalshi_search_markets', {'query': 'trump', 'limit': 5})
    
    # Extract markets from events
    events = result.get('events', [])
    if not events:
        print('No events found')
        return
    
    # Find an active market
    ticker = None
    market_title = None
    for event in events:
        for market in event.get('markets', []):
            status = market.get('status', '')
            if status == 'active' or status == 'open':
                ticker = market.get('ticker', '')
                market_title = market.get('title', event.get('title', ''))
                print(f"Found active: {market_title}")
                print(f"  Ticker: {ticker}")
                print(f"  Status: {status}")
                break
        if ticker:
            break
    
    if not ticker:
        print('No active markets found, trying to get live data...')
        # Try get_live_data for active markets
        live = await call('kalshi_get_live_data', {'limit': 5})
        print(json.dumps(live, indent=2)[:1000])
        return
    
    print(f"\nSelected: {market_title}")
    print(f"Ticker: {ticker}")
    
    # 2. Get market details
    print('\n=== MARKET DETAILS ===')
    details = await call('kalshi_get_market', {'ticker': ticker})
    print(json.dumps(details, indent=2)[:600])
    
    yes_mint = details.get('yesOutcomeMint', '')
    no_mint = details.get('noOutcomeMint', '')
    yes_price = details.get('yesPrice', 0.5)
    
    print(f"\nYES mint: {yes_mint}")
    print(f"NO mint: {no_mint}")
    print(f"YES price: {yes_price}")
    
    if not yes_mint:
        print('No outcome mint found')
        return
    
    # 3. Buy YES for $1
    print('\n=== BUYING $1 YES ===')
    result = await call('kalshi_buy_yes', {
        'marketTicker': ticker,
        'yesOutcomeMint': yes_mint,
        'usdcAmount': 1.0
    })
    print(json.dumps(result, indent=2))
    
    # 4. Check positions
    print('\n=== POSITIONS ===')
    positions = await call('kalshi_get_positions')
    print(json.dumps(positions, indent=2)[:500])
    
    # 5. Check balance
    print('\n=== FINAL BALANCE ===')
    bal = await call('kalshi_get_balances')
    print(f"SOL: {bal['balances']['sol']}")
    print(f"USDC: {bal['balances']['usdc']}")

asyncio.run(main())
