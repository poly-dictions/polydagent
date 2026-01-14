import asyncio, aiohttp, json

URL = 'https://quantish-sdk-production.up.railway.app/mcp'
KEY = 'pk_live_VlF9heMyzdUjTnQ6Zn1FHIkRvo27o8sC'

async def call(tool, args={}):
    payload = {'jsonrpc': '2.0', 'method': 'tools/call', 'params': {'name': tool, 'arguments': args}, 'id': 1}
    headers = {'Content-Type': 'application/json', 'x-api-key': KEY}
    async with aiohttp.ClientSession() as session:
        async with session.post(URL, json=payload, headers=headers) as resp:
            data = await resp.json()
            return json.loads(data.get('result', {}).get('content', [{}])[0].get('text', '{}'))

async def main():
    bal = await call('get_balances')
    print('=== BALANCES ===')
    print(f"USDC.e: {bal['safe']['usdc']}")
    print(f"Native USDC: {bal['safe']['nativeUsdc']}")
    print(f"MATIC: {bal['safe']['matic']}")
    
    pos = await call('get_positions')
    print(f"\n=== POSITIONS ({pos['count']}) ===")
    for p in pos.get('positions', []):
        print(f"  {p['outcome']}: {p['size']} shares @ ${p['currentPrice']}")
    
    orders = await call('get_orders')
    print(f"\n=== OPEN ORDERS ({orders['count']}) ===")
    for o in orders.get('orders', []):
        print(f"  {o.get('side')}: {o.get('originalSize')} @ ${o.get('price')}")

asyncio.run(main())
