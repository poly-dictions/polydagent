"""Generate alpha call for manual posting - uses Gemini as backup"""
import asyncio
import aiohttp
import json
import random
import re
from datetime import datetime

# Config - import from config.py
from config import POLYFACTUAL_API_URL, POLYFACTUAL_API_KEY, GEMINI_API_KEY, MAIN_TWITTER_HANDLE

GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent'
POLYMARKET_API = 'https://gamma-api.polymarket.com'

SPAM_WORDS = [
    'up or down', 'higher or lower', 'above or below',
    'am-', 'pm-', 'am et', 'pm et', ':00', ':15', ':30', ':45',
    'december 19', 'december 20', 'december 21', 'december 22',
    'december 23', 'december 24', 'december 25'
]

async def fetch_events():
    url = f'{POLYMARKET_API}/events'
    params = {'limit': 100, 'active': 'true', 'closed': 'false', 'order': 'volume', 'ascending': 'false'}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, timeout=30) as resp:
            if resp.status == 200:
                return await resp.json()
    return []

def is_valid_market(event):
    title = event.get('title', '').lower()
    volume = float(event.get('volume', 0) or 0)
    markets = event.get('markets', [])

    if any(spam in title for spam in SPAM_WORDS):
        return False
    if volume < 100000:
        return False
    if not markets or len(markets) > 3:
        return False

    market = markets[0]
    outcomes = market.get('outcomes', [])
    if isinstance(outcomes, str):
        outcomes = json.loads(outcomes)
    if len(outcomes) != 2:
        return False

    outcome_names = [str(o.get('name', o) if isinstance(o, dict) else o).lower() for o in outcomes]
    if 'yes' not in outcome_names and 'no' not in outcome_names:
        return False

    end_date_str = market.get('endDate') or event.get('endDate')
    if end_date_str:
        try:
            end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
            now = datetime.now(end_date.tzinfo) if end_date.tzinfo else datetime.now()
            if (end_date - now).days < 14:
                return False
        except:
            pass

    outcome_prices = market.get('outcomePrices')
    if isinstance(outcome_prices, str):
        outcome_prices = json.loads(outcome_prices)
    if outcome_prices:
        yes_pct = float(outcome_prices[0]) * 100
        if yes_pct >= 90 or yes_pct <= 10:
            return False

    return True

def parse_market_data(event):
    market = event.get('markets', [{}])[0]
    outcome_prices = market.get('outcomePrices')
    if isinstance(outcome_prices, str):
        outcome_prices = json.loads(outcome_prices)
    yes_pct = float(outcome_prices[0]) * 100 if outcome_prices else 50
    no_pct = 100 - yes_pct

    end_date_str = market.get('endDate') or event.get('endDate')
    days_left = None
    if end_date_str:
        try:
            end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
            now = datetime.now(end_date.tzinfo) if end_date.tzinfo else datetime.now()
            days_left = (end_date - now).days
        except:
            pass

    return {
        'event_id': event.get('id', ''),
        'title': event.get('title', ''),
        'slug': event.get('slug', ''),
        'yes_odds': yes_pct,
        'no_odds': no_pct,
        'volume': float(event.get('volume', 0) or 0),
        'days_left': days_left
    }

async def analyze_polyfactual(title, yes_odds, no_odds, volume):
    """Try Polyfactual API first"""
    query = f"""Analyze this Polymarket prediction market: [{title}]
Current odds: YES {yes_odds:.0f}% / NO {no_odds:.0f}%
Volume: ${volume:,.0f}

Write analysis with this EXACT format (each reason on its own line, NO periods at end of sentences):

1) YES reasons:
[reason 1 - one sentence, no period at end]
[reason 2 - one sentence, no period at end]

2) NO reasons:
[reason 1 - one sentence, no period at end]
[reason 2 - one sentence, no period at end]

3) Key risk:
[one main uncertainty that could swing the outcome, no period at end]

4) Recommendation:
[YES or NO] with [low/mid/high] confidence - [brief reason, no period at end]

IMPORTANT: Never end sentences with periods. Each sentence on its own line. Plain text only, no markdown."""

    headers = {'Content-Type': 'application/json', 'X-API-Key': POLYFACTUAL_API_KEY}
    data = {'query': query, 'text': True}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(POLYFACTUAL_API_URL, headers=headers, json=data, timeout=120) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    if result.get('success') and result.get('data'):
                        answer = result['data'].get('answer', '') or result['data'].get('text', '')
                        if isinstance(answer, dict):
                            answer = answer.get('text', str(answer))
                        reasoning = str(answer).strip()
                        reasoning = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', reasoning)
                        reasoning = re.sub(r'https?://\S+', '', reasoning)
                        reasoning = re.sub(r'\([^)]*polymarket[^)]*\)', '', reasoning)
                        reasoning = re.sub(r'\([^)]*\.com[^)]*\)', '', reasoning)
                        reasoning = reasoning.replace('**', '')
                        lines = reasoning.split('\n')
                        lines = [' '.join(line.split()) for line in lines]
                        reasoning = '\n'.join(line for line in lines if line)
                        return reasoning
                else:
                    print(f'Polyfactual API error: {resp.status}')
    except Exception as e:
        print(f'Polyfactual error: {e}')
    return None

async def analyze_gemini(title, yes_odds, no_odds, volume):
    """Fallback to Gemini API"""
    prompt = f"""Analyze this Polymarket prediction market: [{title}]
Current odds: YES {yes_odds:.0f}% / NO {no_odds:.0f}%
Volume: ${volume:,.0f}

Write analysis with this EXACT format (each reason on its own line, NO periods at end of sentences):

1) YES reasons:
[reason 1 - one sentence, no period at end]
[reason 2 - one sentence, no period at end]

2) NO reasons:
[reason 1 - one sentence, no period at end]
[reason 2 - one sentence, no period at end]

3) Key risk:
[one main uncertainty that could swing the outcome, no period at end]

4) Recommendation:
[YES or NO] with [low/mid/high] confidence - [brief reason, no period at end]

IMPORTANT: Never end sentences with periods. Each sentence on its own line. Plain text only, no markdown."""

    url = f'{GEMINI_API_URL}?key={GEMINI_API_KEY}'
    headers = {'Content-Type': 'application/json'}
    data = {'contents': [{'parts': [{'text': prompt}]}]}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data, timeout=60) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                    text = text.replace('**', '')
                    return text.strip()
                else:
                    err = await resp.text()
                    print(f'Gemini error: {resp.status} - {err[:200]}')
    except Exception as e:
        print(f'Gemini error: {e}')
    return None

def format_volume(volume):
    if volume >= 1_000_000:
        return f'${volume/1_000_000:.1f}M'
    elif volume >= 1_000:
        return f'${volume/1_000:.0f}K'
    return f'${volume:.0f}'

def create_tweet(market):
    openers = [
        '🧡 found some edge here',
        "🧡 this one's interesting",
        '🧡 market might be off',
        '🧡 worth a look',
        '🧡 spotted something',
    ]
    opener = random.choice(openers)
    time_text = ''
    if market.get('days_left'):
        days = market['days_left']
        if days >= 60:
            time_text = f'~{days // 30}mo out'
        elif days >= 30:
            time_text = '~1mo out'
        else:
            time_text = f'{days}d left'

    tweet = f"""{opener}

{market['title']}

YES {market['yes_odds']:.1f}% / NO {market['no_odds']:.1f}%
vol: {format_volume(market['volume'])} • {time_text}

polymarket.com/event/{market['slug']}

{MAIN_TWITTER_HANDLE} 🧡"""
    return tweet

def create_reply_thread(reasoning):
    sections = []
    current_section = ''
    for line in reasoning.split('\n'):
        if re.match(r'^\d+[\)\.]\s', line.strip()):
            if current_section:
                sections.append(current_section.strip())
            current_section = line
        else:
            current_section += '\n' + line if current_section else line
    if current_section:
        sections.append(current_section.strip())

    if len(sections) <= 1:
        sections = [reasoning]

    tweets = []
    for i, section in enumerate(sections):
        text = re.sub(r'^\d+[\)\.]\s*', '', section.strip())
        # First tweet gets "deep research 🧠" header
        if i == 0:
            tweet = f'deep research 🧠\n\n{i+1}/\n\n{text}'
        else:
            tweet = f'{i+1}/\n\n{text}'
        if i == len(sections) - 1:
            tweet += '\n\nby @polydictions 🧡'
        tweets.append(tweet)
    return tweets

async def main():
    print('Fetching markets...')
    events = await fetch_events()

    valid_markets = []
    for event in events:
        if is_valid_market(event):
            market_data = parse_market_data(event)
            valid_markets.append(market_data)

    print(f'Found {len(valid_markets)} valid markets')

    if not valid_markets:
        print('No valid markets')
        return

    top_markets = sorted(valid_markets, key=lambda x: x['volume'], reverse=True)[:10]
    market = random.choice(top_markets)

    print(f'Selected: {market["title"]}')
    print(f'Volume: {format_volume(market["volume"])}')
    print()

    # Try Polyfactual first, fallback to Gemini
    print('Trying Polyfactual API...')
    reasoning = await analyze_polyfactual(market['title'], market['yes_odds'], market['no_odds'], market['volume'])

    if not reasoning:
        print('Polyfactual failed, trying Gemini...')
        reasoning = await analyze_gemini(market['title'], market['yes_odds'], market['no_odds'], market['volume'])

    if not reasoning:
        print('Failed to get AI analysis from any provider')
        return

    print()
    print('='*60)
    print('MAIN TWEET:')
    print('='*60)
    print(create_tweet(market))
    print()
    print('='*60)
    print('REPLY THREAD:')
    print('='*60)
    for tweet in create_reply_thread(reasoning):
        print(tweet)
        print('-'*40)

async def test_recap():
    """Test recap generation without posting"""
    from agent import Database, DailyRecap, TwitterPoster

    class FakePoster:
        def post(self, text): return "FAKE_ID"
        def post_reply(self, id, text): return "FAKE_REPLY"
        def format_volume(self, v):
            if v >= 1_000_000: return f'${v/1_000_000:.1f}M'
            elif v >= 1_000: return f'${v/1_000:.0f}K'
            return f'${v:.0f}'

    db = Database()
    recap = DailyRecap(db, FakePoster())

    print("Generating recap...")
    data = await recap.generate_recap()

    print(f"\nActive calls: {data['total_active']}")
    print(f"Calls today: {data['calls_today']}")
    print(f"Stats: {data['stats']}")

    if data['calls']:
        print("\n" + "="*60)
        print("RECAP TWEET:")
        print("="*60)
        print(recap.format_recap_tweet(data))

        if len(data['calls']) > 3:
            print("\n" + "="*60)
            print("RECAP THREAD:")
            print("="*60)
            for i, tweet in enumerate(recap.format_recap_thread(data)):
                print(f"--- Tweet {i+1} ---")
                print(tweet)
                print()
    else:
        print("\nNo active calls for recap")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'recap':
        asyncio.run(test_recap())
    else:
        asyncio.run(main())
