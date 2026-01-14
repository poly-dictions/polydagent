"""Test multi-user API endpoints"""
import asyncio
import aiohttp
import json

BASE = 'http://localhost:8765'

async def test():
    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        
        # Create wallet for new user
        print('=== CREATE POLYMARKET WALLET ===')
        async with session.post(f'{BASE}/api/quantish/user/create-wallet', 
                                json={'userId': 'telegram_888', 'platform': 'polymarket'}) as resp:
            data = await resp.json()
            print(json.dumps(data, indent=2))
        
        # Get wallet
        print('\n=== GET WALLET ===')
        async with session.get(f'{BASE}/api/quantish/user/wallet?userId=telegram_888&platform=polymarket') as resp:
            data = await resp.json()
            print(json.dumps(data, indent=2))
        
        # Get balances
        print('\n=== GET BALANCES ===')
        async with session.get(f'{BASE}/api/quantish/user/balances?userId=telegram_888&platform=polymarket') as resp:
            data = await resp.json()
            print(json.dumps(data, indent=2))
        
        # Create Kalshi wallet
        print('\n=== CREATE KALSHI WALLET ===')
        async with session.post(f'{BASE}/api/quantish/user/create-wallet',
                                json={'userId': 'telegram_888', 'platform': 'kalshi'}) as resp:
            data = await resp.json()
            print(json.dumps(data, indent=2))

asyncio.run(test())
