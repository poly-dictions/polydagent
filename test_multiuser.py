"""Test multi-user Quantish wallets"""
import asyncio
from quantish_users import quantish_users

async def main():
    user_id = "test_user_001"
    
    print("=== CREATE POLYMARKET WALLET ===")
    result = await quantish_users.create_polymarket_wallet(user_id)
    print(result)
    
    if result.get("success"):
        print(f"\nDeposit address: {result['wallet']}")
        print(f"Network: {result['network']}")
        print(f"Tokens: {result['deposit_tokens']}")
        
        print("\n=== CHECK BALANCES ===")
        balances = await quantish_users.get_balances(user_id, "polymarket")
        print(balances)
    
    print("\n=== CREATE KALSHI WALLET ===")
    result = await quantish_users.create_kalshi_wallet(user_id)
    print(result)
    
    if result.get("success"):
        print(f"\nDeposit address: {result['wallet']}")
        print(f"Network: {result['network']}")

asyncio.run(main())
