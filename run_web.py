"""
Run the web server for polyweb
Usage: python run_web.py
"""
import asyncio
import logging
import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from api_server import APIServer

async def main():
    server = APIServer(host="0.0.0.0", port=8765)
    runner = await server.start()

    print("\n" + "="*50)
    print("  Polydictions Web Server")
    print("="*50)
    print(f"  Local:   http://localhost:8765")
    print(f"  Network: http://0.0.0.0:8765")
    print("="*50)
    print("\n  Pages:")
    print("  - /              Landing page")
    print("  - /markets/      Markets browser")
    print("  - /whales/       Whale tracker")
    print("  - /arbitrage/    Arbitrage scanner")
    print("  - /roadmap/      Roadmap")
    print("\n  API Endpoints:")
    print("  - /api/events    Polymarket events")
    print("  - /api/arbitrage Arbitrage opportunities")
    print("  - /api/watchlist User watchlists")
    print("\n  Press Ctrl+C to stop\n")

    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        print("\nShutting down...")
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
