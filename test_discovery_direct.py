import asyncio, aiohttp, json

async def test():
    url = 'https://quantish.live/mcp'
    key = 'qm_dyvSSGPYLcRwLXBNmW5VPa94sjCVAe-6'
    payload = {'jsonrpc': '2.0', 'method': 'tools/call', 'params': {'name': 'search_markets', 'arguments': {'query': 'bitcoin', 'limit': 2}}, 'id': 1}
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream', 'X-API-Key': key}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            print(f'Status: {resp.status}')
            ct = resp.headers.get('Content-Type', '')
            print(f'Content-Type: {ct}')
            text = await resp.text()
            print(f'Response: {text[:500]}')

asyncio.run(test())
