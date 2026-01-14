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
            text = data.get('result', {}).get('content', [{}])[0].get('text', '{}')
            return json.loads(text)

async def main():
    ticker = 'KXFEDCHAIRNOM-29-MBOW'
    
    # Check if market is initialized on DFlow
    print('=== CHECK MARKET INITIALIZATION ===')
    result = await call('kalshi_check_market_initialization', {'ticker': ticker})
    print(json.dumps(result, indent=2))
    
    # If not initialized, initialize it
    if not result.get('initialized'):
        print('\n=== INITIALIZING MARKET ===')
        init = await call('kalshi_initialize_market', {'ticker': ticker})
        print(json.dumps(init, indent=2))
        
        # Check again
        result = await call('kalshi_check_market_initialization', {'ticker': ticker})
        print(json.dumps(result, indent=2))
    
    yes_mint = result.get('yesMint', result.get('yesOutcomeMint', ''))
    no_mint = result.get('noMint', result.get('noOutcomeMint', ''))
    
    if not yes_mint:
        print('Still no mint found')
        return
    
    print(f"\nYES mint: {yes_mint}")
    print(f"NO mint: {no_mint}")
    
    # Get quote first
    print('\n=== GET QUOTE ===')
    quote = await call('kalshi_get_quote', {'marketTicker': ticker, 'side': 'yes', 'usdcAmount': 1})
    print(json.dumps(quote, indent=2))
    
    # Buy YES
    print('\n=== BUYING $1 YES ===')
    result = await call('kalshi_buy_yes', {
        'marketTicker': ticker,
        'yesOutcomeMint': yes_mint,
        'usdcAmount': 1.0
    })
    print(json.dumps(result, indent=2))
    
    # Check positions
    print('\n=== POSITIONS ===')
    positions = await call('kalshi_get_positions')
    print(json.dumps(positions, indent=2)[:800])
    
    # Check balance
    print('\n=== FINAL BALANCE ===')
    bal = await call('kalshi_get_balances')
    print(f"SOL: {bal['balances']['sol']}")
    print(f"USDC: {bal['balances']['usdc']}")

asyncio.run(main())
