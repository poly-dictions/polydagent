"""
Test Quantish MCP Integration
Run: python test_quantish.py
"""
import asyncio
import aiohttp
import json

# Quantish API Configuration
QUANTISH_DISCOVERY_URL = "https://quantish.live/mcp"
QUANTISH_DISCOVERY_KEY = "qm_dyvSSGPYLcRwLXBNmW5VPa94sjCVAe-6"
QUANTISH_POLYMARKET_URL = "https://quantish-sdk-production.up.railway.app/mcp"
QUANTISH_POLYMARKET_KEY = "pk_live_VlF9heMyzdUjTnQ6Zn1FHIkRvo27o8sC"
QUANTISH_KALSHI_URL = "https://kalshi-mcp-production-7c2c.up.railway.app/mcp"
QUANTISH_KALSHI_KEY = "pk_kalshi_pvttXePc4vzFj1tVPNG3i82a81qB_QE1"


async def call_mcp(url: str, api_key: str, tool_name: str, arguments: dict) -> dict:
    """Make a JSON-RPC call to MCP endpoint"""
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        },
        "id": 1
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "x-api-key": api_key
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers, timeout=30) as resp:
            if resp.status != 200:
                text = await resp.text()
                return {"error": f"HTTP {resp.status}: {text[:200]}"}
            
            data = await resp.json()
            if "error" in data:
                return {"error": data["error"]}
            
            result = data.get("result", {})
            content = result.get("content", [])
            if content and len(content) > 0:
                text = content[0].get("text", "{}")
                try:
                    return json.loads(text)
                except:
                    return {"raw": text}
            return result


async def test_discovery():
    """Test Discovery API"""
    print("\n" + "="*60)
    print("🔍 DISCOVERY API TEST")
    print("="*60)
    
    # Test search
    print("\n1. Search markets for 'bitcoin'...")
    result = await call_mcp(
        QUANTISH_DISCOVERY_URL, QUANTISH_DISCOVERY_KEY,
        "search_markets", {"query": "bitcoin", "limit": 3}
    )
    if "error" in result:
        print(f"   ❌ Error: {result['error']}")
    else:
        markets = result.get("markets", result)
        if isinstance(markets, list):
            print(f"   ✅ Found {len(markets)} markets")
            for m in markets[:2]:
                title = m.get("title", m.get("question", "?"))[:50]
                platform = m.get("platform", "?")
                print(f"      - [{platform}] {title}...")
        else:
            print(f"   ✅ Response: {str(result)[:200]}")
    
    # Test trending
    print("\n2. Get trending markets...")
    result = await call_mcp(
        QUANTISH_DISCOVERY_URL, QUANTISH_DISCOVERY_KEY,
        "get_trending_markets", {"limit": 3}
    )
    if "error" in result:
        print(f"   ❌ Error: {result['error']}")
    else:
        markets = result.get("markets", result)
        if isinstance(markets, list):
            print(f"   ✅ Found {len(markets)} trending markets")
        else:
            print(f"   ✅ Response: {str(result)[:200]}")


async def test_polymarket():
    """Test Polymarket Trading API"""
    print("\n" + "="*60)
    print("🟣 POLYMARKET API TEST")
    print("="*60)
    
    # Test balances
    print("\n1. Get wallet balances...")
    result = await call_mcp(
        QUANTISH_POLYMARKET_URL, QUANTISH_POLYMARKET_KEY,
        "get_balances", {}
    )
    if "error" in result:
        print(f"   ❌ Error: {result['error']}")
    else:
        usdc = result.get("usdc", result.get("USDC", "?"))
        matic = result.get("matic", result.get("MATIC", "?"))
        print(f"   ✅ USDC: {usdc}, MATIC: {matic}")
    
    # Test positions
    print("\n2. Get open positions...")
    result = await call_mcp(
        QUANTISH_POLYMARKET_URL, QUANTISH_POLYMARKET_KEY,
        "get_positions", {}
    )
    if "error" in result:
        print(f"   ❌ Error: {result['error']}")
    else:
        positions = result.get("positions", result)
        if isinstance(positions, list):
            print(f"   ✅ {len(positions)} open positions")
            for p in positions[:3]:
                title = p.get("title", p.get("market", "?"))[:40]
                size = p.get("size", p.get("shares", "?"))
                print(f"      - {title}... ({size} shares)")
        else:
            print(f"   ✅ Response: {str(result)[:200]}")


async def test_kalshi():
    """Test Kalshi Trading API"""
    print("\n" + "="*60)
    print("🔵 KALSHI API TEST")
    print("="*60)
    
    # Test balances
    print("\n1. Get wallet balances...")
    result = await call_mcp(
        QUANTISH_KALSHI_URL, QUANTISH_KALSHI_KEY,
        "kalshi_get_balances", {}
    )
    if "error" in result:
        print(f"   ❌ Error: {result['error']}")
    else:
        sol = result.get("sol", result.get("SOL", "?"))
        usdc = result.get("usdc", result.get("USDC", "?"))
        print(f"   ✅ SOL: {sol}, USDC: {usdc}")
    
    # Test positions
    print("\n2. Get open positions...")
    result = await call_mcp(
        QUANTISH_KALSHI_URL, QUANTISH_KALSHI_KEY,
        "kalshi_get_positions", {}
    )
    if "error" in result:
        print(f"   ❌ Error: {result['error']}")
    else:
        positions = result.get("positions", result)
        if isinstance(positions, list):
            print(f"   ✅ {len(positions)} open positions")
        else:
            print(f"   ✅ Response: {str(result)[:200]}")


async def test_local_api():
    """Test local API server endpoints (if running)"""
    print("\n" + "="*60)
    print("🌐 LOCAL API SERVER TEST")
    print("="*60)
    
    base_url = "http://localhost:8765"
    
    async with aiohttp.ClientSession() as session:
        # Test search
        print("\n1. Test /api/quantish/search...")
        try:
            async with session.get(f"{base_url}/api/quantish/search?q=trump&limit=2", timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"   ✅ Success: {data.get('success', False)}")
                else:
                    print(f"   ❌ HTTP {resp.status}")
        except Exception as e:
            print(f"   ⚠️ Server not running or error: {e}")
        
        # Test balances
        print("\n2. Test /api/quantish/balances...")
        try:
            async with session.get(f"{base_url}/api/quantish/balances", timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"   ✅ Success: {data.get('success', False)}")
                    if data.get("data"):
                        print(f"      Polymarket: {data['data'].get('polymarket', {})}")
                        print(f"      Kalshi: {data['data'].get('kalshi', {})}")
                else:
                    print(f"   ❌ HTTP {resp.status}")
        except Exception as e:
            print(f"   ⚠️ Server not running or error: {e}")


async def main():
    print("\n" + "🚀 QUANTISH INTEGRATION TEST")
    print("="*60)
    
    await test_discovery()
    await test_polymarket()
    await test_kalshi()
    await test_local_api()
    
    print("\n" + "="*60)
    print("✅ TEST COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
