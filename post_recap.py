"""Post daily recap to Twitter"""
import sys
sys.path.insert(0, r'c:\Users\Username\Desktop\polydictions-agent')

import asyncio
import aiohttp
import json
import sqlite3
from config import *
import tweepy

# Twitter API setup
auth = tweepy.OAuth1UserHandler(
    TWITTER_API_KEY, TWITTER_API_SECRET,
    TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET
)
api_v1 = tweepy.API(auth)
client = tweepy.Client(
    consumer_key=TWITTER_API_KEY,
    consumer_secret=TWITTER_API_SECRET,
    access_token=TWITTER_ACCESS_TOKEN,
    access_token_secret=TWITTER_ACCESS_SECRET
)

POLYMARKET_API = 'https://gamma-api.polymarket.com'

async def fetch_current_odds(slug):
    """Fetch current odds for a market"""
    url = f"{POLYMARKET_API}/events/slug/{slug}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=15) as resp:
                if resp.status == 200:
                    event = await resp.json()
                    markets = event.get('markets', [])
                    if markets:
                        market = markets[0]
                        outcome_prices = market.get('outcomePrices')
                        if isinstance(outcome_prices, str):
                            outcome_prices = json.loads(outcome_prices)
                        if outcome_prices:
                            yes_pct = float(outcome_prices[0]) * 100
                            return yes_pct
    except Exception as e:
        print(f"Failed to fetch odds for {slug}: {e}")
    return None

async def generate_recap():
    """Generate recap data"""
    db_path = r'c:\Users\Username\Desktop\polydictions-agent\calls.db'
    conn = sqlite3.connect(db_path)

    # Get active calls
    calls = conn.execute('''
        SELECT id, slug, title, yes_odds_at_call, signal, created_at
        FROM calls WHERE resolved = FALSE
        ORDER BY created_at DESC
    ''').fetchall()

    # Get stats
    stats = conn.execute('SELECT total_calls, wins, losses, pending FROM stats').fetchone()

    conn.close()

    recap_data = {
        'total_active': len(calls),
        'stats': {
            'total': stats[0],
            'wins': stats[1],
            'losses': stats[2],
            'pending': stats[3]
        },
        'calls': []
    }

    for call in calls:
        call_id, slug, title, odds_at_call, signal, created_at = call
        current_odds = await fetch_current_odds(slug)

        if current_odds is not None:
            raw_change = current_odds - odds_at_call
            # Calculate change from our position's perspective
            # If we bet NO and YES odds dropped, that's positive for us
            if signal == 'NO':
                our_change = -raw_change  # Invert: YES dropping = good for NO
            else:
                our_change = raw_change   # YES rising = good for YES
            recap_data['calls'].append({
                'title': title,
                'slug': slug,
                'signal': signal,
                'odds_at_call': odds_at_call,
                'odds_now': current_odds,
                'raw_change': raw_change,
                'our_change': our_change
            })
        else:
            recap_data['calls'].append({
                'title': title,
                'slug': slug,
                'signal': signal,
                'odds_at_call': odds_at_call,
                'odds_now': None,
                'raw_change': None,
                'our_change': None
            })

    return recap_data

def format_recap_tweet(recap):
    """Format recap as main tweet"""
    stats = recap['stats']

    tweet = f"""🧡 daily recap

📊 tracking {recap['total_active']} active calls

"""

    # Sort by absolute our_change (how good for us)
    movers = [c for c in recap['calls'] if c['our_change'] is not None]
    movers.sort(key=lambda x: abs(x['our_change']), reverse=True)

    for i, call in enumerate(movers, 1):
        our_change = call['our_change']
        raw_change = call['raw_change']

        # Format: show raw market change, then "for us" change
        if our_change > 0:
            our_str = f"+{our_change:.1f} for us"
        elif our_change < 0:
            our_str = f"{our_change:.1f} for us"
        else:
            our_str = "0.0 for us"

        title = call['title']
        if len(title) > 50:
            title = title[:48] + ".."

        # Show: market moved X%, that's Y for us
        tweet += f"{i}. {title}\n{call['odds_at_call']:.0f}% → {call['odds_now']:.0f}% ({our_str})\n\n"

    tweet += "by @polydictions 🧡"

    return tweet

async def main():
    print("Generating recap...")
    recap = await generate_recap()

    print(f"\nActive calls: {recap['total_active']}")
    print(f"Stats: {recap['stats']}")

    for call in recap['calls']:
        our_change = call['our_change']
        change_str = f"{our_change:+.1f}" if our_change else "N/A"
        odds_now = f"{call['odds_now']:.1f}" if call['odds_now'] else "N/A"
        signal = call.get('signal', '?')
        print(f"  [{signal}] {call['title'][:35]}: {call['odds_at_call']:.1f}% -> {odds_now}% ({change_str} for us)")

    tweet = format_recap_tweet(recap)
    print("\n" + "="*50)
    print("RECAP TWEET:")
    print("="*50)
    # Write to file for full emoji support, print ASCII version to console
    with open('recap_preview.txt', 'w', encoding='utf-8') as f:
        f.write(tweet)
    print("[Tweet saved to recap_preview.txt - check file for emojis]")
    print(tweet.encode('ascii', errors='replace').decode('ascii'))
    print("="*50)
    print(f"Length: {len(tweet)} chars")

    # Post
    print("\nPosting...")
    try:
        response = client.create_tweet(text=tweet)
        tweet_id = response.data['id']
        print(f"Posted! https://twitter.com/polydagent/status/{tweet_id}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    asyncio.run(main())
