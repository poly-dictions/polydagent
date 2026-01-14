import asyncio, aiohttp, json

async def main():
    url = 'https://kalshi-mcp-production-7c2c.up.railway.app/mcp'
    key = 'pk_kalshi_pvttXePc4vzFj1tVPNG3i82a81qB_QE1'
    headers = {'Content-Type': 'application/json', 'x-api-key': key}
    
    payload = {'jsonrpc': '2.0', 'method': 'tools/list', 'id': 1}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            data = await resp.json()
            tools = data.get('result', {}).get('tools', [])
            print(f"Found {len(tools)} tools:\n")
            for t in tools:
                name = t.get('name', '')
                desc = t.get('description', '')[:80]
                print(f"  {name}")
                if 'buy' in name or 'sell' in name or 'order' in name:
                    print(f"    -> {desc}")
                    schema = t.get('inputSchema', {}).get('properties', {})
                    for k, v in schema.items():
                        print(f"       {k}: {v.get('description', v.get('type', ''))[:50]}")

asyncio.run(main())
