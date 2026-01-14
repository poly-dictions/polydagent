import asyncio, aiohttp, json

URL = 'https://kalshi-mcp-production-7c2c.up.railway.app/mcp'
KEY = 'pk_kalshi_pvttXePc4vzFj1tVPNG3i82a81qB_QE1'

async def call(tool, args={}):
    payload = {'jsonrpc': '2.0', 'method': 'tools/call', 'params': {'name': tool, 'arguments': args}, 'id': 1}
    headers = {'Content-Type': 'application/json', 'x-api-key': KEY}
    async with aiohttp.ClientSession() as session:
        async with session.post(URL, json=payload, headers=headers) as resp:
            data = await resp.json()
            return json.loads(data.get('result', {}).get('content', [{}])[0].get('text', '{}'))

async def main():
    bal = await call('kalshi_get_balances')
    print('=== KALSHI BALANCES ===')
    print(f"SOL: {bal['balances']['sol']}")
    print(f"USDC: {bal['balances']['usdc']}")

asyncio.run(main())
