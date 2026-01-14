"""Quick trade test"""
import asyncio
import aiohttp
import json

URL = "https://quantish-sdk-production.up.railway.app/mcp"
KEY = "pk_live_VlF9heMyzdUjTnQ6Zn1FHIkRvo27o8sC"

async def call_mcp(tool, args):
    payload = {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": tool, "arguments": args}, "id": 1}
    headers = {"Content-Type": "application/json", "x-api-key": KEY}
    async with aiohttp.ClientSession() as session:
        async with session.post(URL, json=payload, headers=headers) as resp:
            data = await resp.json()
            text = data.get("result", {}).get("content", [{}])[0].get("text", "{}")
            return json.loads(text)

async def main():
    # 1. Check balance
    print("=== BALANCE ===")
    bal = await call_mcp("get_balances", {})
    print(f"USDC: {bal['safe']['nativeUsdc']}")
    
    # 2. Search market
    print("\n=== SEARCH MARKET ===")
    result = await call_mcp("search_markets", {"query": "bitcoin", "limit": 3})
    print(f"Found: {len(result.get('markets', []))} markets")
    if not result.get("markets"):
        print("No markets found, trying trending...")
        result = await call_mcp("get_trending_markets", {"limit": 3})
        print(json.dumps(result, indent=2)[:500])
        return
    market = result["markets"][0]
    print(f"Market data: {json.dumps(market, indent=2)[:400]}")
    title = market.get('title') or market.get('question') or 'Unknown'
    print(f"Title: {title[:60]}...")
    cond_id = market.get('conditionId') or market.get('condition_id')
    print(f"conditionId: {cond_id}")
    print(f"tokens: {market.get('clobTokenIds', market.get('tokens'))}")
    print(f"yesPrice: {market.get('yesPrice', market.get('yes_price', 'N/A'))}")
    
    # 3. Get market details for token IDs
    print("\n=== MARKET DETAILS ===")
    details = await call_mcp("get_market", {"conditionId": market["conditionId"]})
    print(json.dumps(details, indent=2)[:500])
    
    # Extract token ID for YES
    tokens = details.get("clobTokenIds") or details.get("tokens") or []
    if isinstance(tokens, str):
        tokens = json.loads(tokens)
    
    # Get YES token - it's either a string or object with tokenId
    yes_token_data = tokens[0] if tokens else None
    if isinstance(yes_token_data, dict):
        yes_token = yes_token_data.get("tokenId")
        yes_price = yes_token_data.get("price", 0.5)
    else:
        yes_token = yes_token_data
        yes_price = 0.5
    print(f"\nYES token ID: {yes_token}")
    print(f"YES price: {yes_price}")
    
    if yes_token and input("\nPlace $1 BUY YES order? (y/n): ").lower() == 'y':
        print("\n=== PLACING ORDER ===")
        # Calculate size (shares) from amount and price
        amount_usd = 1.0
        size = amount_usd / yes_price  # shares = dollars / price_per_share
        order = await call_mcp("place_order", {
            "tokenId": yes_token,
            "conditionId": cond_id,
            "side": "BUY",
            "size": size,
            "price": yes_price
        })
        print(json.dumps(order, indent=2))

asyncio.run(main())
