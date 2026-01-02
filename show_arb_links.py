"""Show arbitrage opportunities with correct links"""
import asyncio
from dome_tracker import DomeClient

def get_kalshi_url(ticker: str) -> str:
    """Convert Kalshi ticker to URL.

    Ticker format: KXNFLGAME-25DEC25DENKC-DEN
    URL format: https://kalshi.com/markets/kxnflgame/professional-football-game/kxnflgame-25dec25denkc
    """
    # Extract event ticker (without team suffix)
    parts = ticker.split('-')
    if len(parts) >= 2:
        # KXNFLGAME-25DEC25DENKC-DEN -> kxnflgame-25dec25denkc
        event_ticker = '-'.join(parts[:-1]).lower()
        series = parts[0].lower()  # kxnflgame

        # Map series to readable name
        name_map = {
            'kxnflgame': 'professional-football-game',
            'kxnbagame': 'professional-basketball-game',
            'kxnhlgame': 'nhl-game',
            'kxncaafgame': 'college-football-game',
            'kxncaambgame': 'mens-college-basketball-mens-game',
        }
        name = name_map.get(series, series)

        return f"https://kalshi.com/markets/{series}/{name}/{event_ticker}"
    return f"https://kalshi.com/markets/{ticker.lower()}"

async def main():
    client = DomeClient()
    opps = await client.find_sports_arbitrage(min_diff=5.0)

    print("TOP ARBITRAGE OPPORTUNITIES\n")
    print("=" * 70)

    for i, opp in enumerate(opps[:10], 1):
        team = opp.get('team', '')
        kalshi_url = get_kalshi_url(opp['kalshi_ticker'])

        print(f"\n{i}. {opp['sport']} - {team} ({opp['date']})")
        print(f"   Spread: {opp['difference']:.1f}%")
        print(f"   Kalshi: {opp['kalshi_price']*100:.0f}% | Polymarket: {opp['polymarket_price']*100:.0f}%")
        print(f"   -> {opp['action']}")
        print(f"\n   Polymarket: https://polymarket.com/event/{opp['polymarket_slug']}")
        print(f"   Kalshi: {kalshi_url}")

if __name__ == "__main__":
    asyncio.run(main())
