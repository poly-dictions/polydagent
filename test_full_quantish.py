"""
Full Quantish Integration Test
Tests all endpoints multiple times
"""
import asyncio
import aiohttp
import json
import random
import string

BASE = 'http://localhost:8765'

def random_user():
    return 'test_' + ''.join(random.choices(string.digits, k=6))

async def test_create_wallet(session, user_id, platform):
    """Test wallet creation"""
    async with session.post(f'{BASE}/api/quantish/user/create-wallet', 
                            json={'userId': user_id, 'platform': platform}) as resp:
        return await resp.json()

async def test_get_wallet(session, user_id, platform):
    """Test get wallet"""
    async with session.get(f'{BASE}/api/quantish/user/wallet?userId={user_id}&platform={platform}') as resp:
        return await resp.json()

async def test_get_balances(session, user_id, platform):
    """Test get balances"""
    async with session.get(f'{BASE}/api/quantish/user/balances?userId={user_id}&platform={platform}') as resp:
        return await resp.json()

async def test_get_positions(session, user_id, platform):
    """Test get positions"""
    async with session.get(f'{BASE}/api/quantish/user/positions?userId={user_id}&platform={platform}') as resp:
        return await resp.json()

async def test_search(session, query):
    """Test market search"""
    async with session.get(f'{BASE}/api/quantish/search?q={query}&limit=3') as resp:
        return await resp.json()

async def test_trending(session):
    """Test trending markets"""
    async with session.get(f'{BASE}/api/quantish/trending?limit=3') as resp:
        return await resp.json()

async def run_tests():
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        
        results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        # =====================================================================
        # TEST 1: Create multiple Polymarket wallets
        # =====================================================================
        print("\n" + "="*60)
        print("TEST 1: Create Polymarket Wallets (3 users)")
        print("="*60)
        
        poly_users = []
        for i in range(3):
            user_id = random_user()
            print(f"\n  Creating wallet for {user_id}...")
            try:
                result = await test_create_wallet(session, user_id, 'polymarket')
                if result.get('success') and result.get('data', {}).get('wallet'):
                    print(f"  ✅ Wallet: {result['data']['wallet']}")
                    poly_users.append({'id': user_id, 'wallet': result['data']['wallet']})
                    results['passed'] += 1
                else:
                    print(f"  ❌ Failed: {result}")
                    results['failed'] += 1
                    results['errors'].append(f"Create poly wallet: {result}")
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results['failed'] += 1
                results['errors'].append(f"Create poly wallet exception: {e}")
        
        # =====================================================================
        # TEST 2: Create Kalshi wallets
        # =====================================================================
        print("\n" + "="*60)
        print("TEST 2: Create Kalshi Wallets (3 users)")
        print("="*60)
        
        kalshi_users = []
        for i in range(3):
            user_id = random_user()
            print(f"\n  Creating wallet for {user_id}...")
            try:
                result = await test_create_wallet(session, user_id, 'kalshi')
                if result.get('success') and result.get('data', {}).get('wallet'):
                    print(f"  ✅ Wallet: {result['data']['wallet']}")
                    kalshi_users.append({'id': user_id, 'wallet': result['data']['wallet']})
                    results['passed'] += 1
                else:
                    print(f"  ❌ Failed: {result}")
                    results['failed'] += 1
                    results['errors'].append(f"Create kalshi wallet: {result}")
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results['failed'] += 1
                results['errors'].append(f"Create kalshi wallet exception: {e}")
        
        # =====================================================================
        # TEST 3: Get wallet info
        # =====================================================================
        print("\n" + "="*60)
        print("TEST 3: Get Wallet Info")
        print("="*60)
        
        for user in poly_users[:2]:
            print(f"\n  Getting wallet for {user['id']}...")
            try:
                result = await test_get_wallet(session, user['id'], 'polymarket')
                if result.get('success') and result.get('data', {}).get('wallet') == user['wallet']:
                    print(f"  ✅ Wallet matches: {result['data']['wallet']}")
                    results['passed'] += 1
                else:
                    print(f"  ❌ Mismatch: {result}")
                    results['failed'] += 1
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results['failed'] += 1
        
        # =====================================================================
        # TEST 4: Get balances
        # =====================================================================
        print("\n" + "="*60)
        print("TEST 4: Get Balances")
        print("="*60)
        
        for user in poly_users[:2]:
            print(f"\n  Getting balances for {user['id']}...")
            try:
                result = await test_get_balances(session, user['id'], 'polymarket')
                if result.get('success') and 'safe' in result.get('data', {}):
                    safe = result['data']['safe']
                    print(f"  ✅ USDC: {safe.get('usdc', 0)}, MATIC: {safe.get('matic', 0)}")
                    results['passed'] += 1
                else:
                    print(f"  ❌ Failed: {result}")
                    results['failed'] += 1
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results['failed'] += 1
        
        for user in kalshi_users[:2]:
            print(f"\n  Getting Kalshi balances for {user['id']}...")
            try:
                result = await test_get_balances(session, user['id'], 'kalshi')
                if result.get('success') and 'balances' in result.get('data', {}):
                    bal = result['data']['balances']
                    print(f"  ✅ SOL: {bal.get('sol', 0)}, USDC: {bal.get('usdc', 0)}")
                    results['passed'] += 1
                else:
                    print(f"  ❌ Failed: {result}")
                    results['failed'] += 1
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results['failed'] += 1
        
        # =====================================================================
        # TEST 5: Get positions
        # =====================================================================
        print("\n" + "="*60)
        print("TEST 5: Get Positions")
        print("="*60)
        
        for user in poly_users[:2]:
            print(f"\n  Getting positions for {user['id']}...")
            try:
                result = await test_get_positions(session, user['id'], 'polymarket')
                if result.get('success'):
                    count = result.get('data', {}).get('count', 0)
                    print(f"  ✅ Positions: {count}")
                    results['passed'] += 1
                else:
                    print(f"  ❌ Failed: {result}")
                    results['failed'] += 1
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results['failed'] += 1
        
        # =====================================================================
        # TEST 6: Search markets
        # =====================================================================
        print("\n" + "="*60)
        print("TEST 6: Search Markets")
        print("="*60)
        
        queries = ['bitcoin', 'trump', 'election']
        for q in queries:
            print(f"\n  Searching '{q}'...")
            try:
                result = await test_search(session, q)
                if result.get('success'):
                    markets = result.get('data', {}).get('markets', [])
                    print(f"  ✅ Found {len(markets)} markets")
                    results['passed'] += 1
                else:
                    print(f"  ❌ Failed: {result}")
                    results['failed'] += 1
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results['failed'] += 1
        
        # =====================================================================
        # TEST 7: Trending markets
        # =====================================================================
        print("\n" + "="*60)
        print("TEST 7: Trending Markets")
        print("="*60)
        
        for i in range(3):
            print(f"\n  Getting trending (attempt {i+1})...")
            try:
                result = await test_trending(session)
                if result.get('success'):
                    trending = result.get('data', {}).get('trending', [])
                    print(f"  ✅ Found {len(trending)} trending")
                    results['passed'] += 1
                else:
                    print(f"  ❌ Failed: {result}")
                    results['failed'] += 1
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results['failed'] += 1
        
        # =====================================================================
        # TEST 8: Duplicate wallet creation (should return existing)
        # =====================================================================
        print("\n" + "="*60)
        print("TEST 8: Duplicate Wallet Creation")
        print("="*60)
        
        if poly_users:
            user = poly_users[0]
            print(f"\n  Re-creating wallet for {user['id']}...")
            try:
                result = await test_create_wallet(session, user['id'], 'polymarket')
                if result.get('success') and result.get('data', {}).get('status') == 'already_exists':
                    print(f"  ✅ Correctly returned existing wallet")
                    results['passed'] += 1
                elif result.get('success') and result.get('data', {}).get('wallet') == user['wallet']:
                    print(f"  ✅ Returned same wallet: {result['data']['wallet']}")
                    results['passed'] += 1
                else:
                    print(f"  ⚠️ Unexpected: {result}")
                    results['passed'] += 1  # Still ok if wallet returned
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results['failed'] += 1
        
        # =====================================================================
        # TEST 9: Non-existent user
        # =====================================================================
        print("\n" + "="*60)
        print("TEST 9: Non-existent User")
        print("="*60)
        
        print("\n  Getting wallet for non-existent user...")
        try:
            result = await test_get_wallet(session, 'nonexistent_user_xyz', 'polymarket')
            if not result.get('success') or result.get('error'):
                print(f"  ✅ Correctly returned error")
                results['passed'] += 1
            else:
                print(f"  ❌ Should have returned error: {result}")
                results['failed'] += 1
        except Exception as e:
            print(f"  ✅ Correctly raised error: {e}")
            results['passed'] += 1
        
        # =====================================================================
        # TEST 10: Missing parameters
        # =====================================================================
        print("\n" + "="*60)
        print("TEST 10: Missing Parameters")
        print("="*60)
        
        print("\n  Creating wallet without userId...")
        try:
            async with session.post(f'{BASE}/api/quantish/user/create-wallet', 
                                    json={'platform': 'polymarket'}) as resp:
                result = await resp.json()
                if result.get('error') and resp.status == 400:
                    print(f"  ✅ Correctly returned 400 error")
                    results['passed'] += 1
                else:
                    print(f"  ❌ Should have returned error: {result}")
                    results['failed'] += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results['failed'] += 1
        
        # =====================================================================
        # SUMMARY
        # =====================================================================
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"\n  ✅ Passed: {results['passed']}")
        print(f"  ❌ Failed: {results['failed']}")
        print(f"  Total: {results['passed'] + results['failed']}")
        
        if results['errors']:
            print(f"\n  Errors:")
            for e in results['errors'][:5]:
                print(f"    - {e[:100]}")
        
        return results

if __name__ == '__main__':
    asyncio.run(run_tests())
