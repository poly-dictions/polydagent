"""
POLYDICTIONS AGENT v2.1
AI-powered Polymarket alpha calls on X/Twitter
Features:
- AI analysis via Polyfactual Deep Research API
- Daily recap with odds tracking
- Win/loss statistics
"""

import os
import asyncio
import aiohttp
import json
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import tweepy

# Config
try:
    from config import (
        TWITTER_API_KEY, TWITTER_API_SECRET,
        TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET,
        TWITTER_BEARER_TOKEN, MAIN_TWITTER_HANDLE,
        POLYFACTUAL_API_URL, POLYFACTUAL_API_KEY,
        POST_INTERVAL_HOURS
    )
except ImportError:
    TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
    TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
    TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
    TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")
    TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
    MAIN_TWITTER_HANDLE = os.getenv("MAIN_TWITTER_HANDLE", "@polydictions")
    POLYFACTUAL_API_URL = os.getenv("POLYFACTUAL_API_URL", "https://deep-research-api.thekid-solana.workers.dev/answer")
    POLYFACTUAL_API_KEY = os.getenv("POLYFACTUAL_API_KEY")
    POST_INTERVAL_HOURS = int(os.getenv("POST_INTERVAL_HOURS", "4"))

POLYMARKET_API = "https://gamma-api.polymarket.com"
DB_PATH = Path(__file__).parent / "calls.db"


class Database:
    """Track all calls and their outcomes"""

    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.setup()

    def setup(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY,
                tweet_id TEXT,
                event_id TEXT,
                slug TEXT,
                title TEXT,
                signal TEXT,
                yes_odds_at_call REAL,
                no_odds_at_call REAL,
                volume_at_call REAL,
                ai_reasoning TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved BOOLEAN DEFAULT FALSE,
                outcome TEXT,
                profit_loss REAL
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stats (
                id INTEGER PRIMARY KEY,
                total_calls INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                pending INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Initialize stats if empty
        if not self.conn.execute("SELECT * FROM stats").fetchone():
            self.conn.execute("INSERT INTO stats (total_calls, wins, losses, pending) VALUES (0, 0, 0, 0)")
        self.conn.commit()

    def add_call(self, tweet_id: str, event_id: str, slug: str, title: str,
                 signal: str, yes_odds: float, no_odds: float, volume: float, reasoning: str):
        self.conn.execute("""
            INSERT INTO calls (tweet_id, event_id, slug, title, signal,
                             yes_odds_at_call, no_odds_at_call, volume_at_call, ai_reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (tweet_id, event_id, slug, title, signal, yes_odds, no_odds, volume, reasoning))
        self.conn.execute("UPDATE stats SET total_calls = total_calls + 1, pending = pending + 1")
        self.conn.commit()

    def get_posted_event_ids(self) -> set:
        rows = self.conn.execute("SELECT event_id FROM calls").fetchall()
        return {row[0] for row in rows}

    def get_stats(self) -> dict:
        row = self.conn.execute("SELECT total_calls, wins, losses, pending FROM stats").fetchone()
        return {
            'total': row[0],
            'wins': row[1],
            'losses': row[2],
            'pending': row[3],
            'win_rate': (row[1] / (row[1] + row[2]) * 100) if (row[1] + row[2]) > 0 else 0
        }

    def get_recent_calls(self, limit: int = 5) -> list:
        rows = self.conn.execute("""
            SELECT title, signal, yes_odds_at_call, created_at
            FROM calls ORDER BY created_at DESC LIMIT ?
        """, (limit,)).fetchall()
        return rows

    def get_active_calls(self) -> list:
        """Get all unresolved calls with their slugs for odds checking"""
        rows = self.conn.execute("""
            SELECT id, event_id, slug, title, yes_odds_at_call, no_odds_at_call,
                   volume_at_call, created_at
            FROM calls
            WHERE resolved = FALSE
            ORDER BY created_at DESC
        """).fetchall()
        return [
            {
                'id': r[0], 'event_id': r[1], 'slug': r[2], 'title': r[3],
                'yes_odds_at_call': r[4], 'no_odds_at_call': r[5],
                'volume_at_call': r[6], 'created_at': r[7]
            }
            for r in rows
        ]

    def get_calls_last_24h(self) -> list:
        """Get calls made in last 24 hours"""
        rows = self.conn.execute("""
            SELECT id, slug, title, yes_odds_at_call, created_at
            FROM calls
            WHERE created_at >= datetime('now', '-1 day')
            ORDER BY created_at DESC
        """).fetchall()
        return [
            {'id': r[0], 'slug': r[1], 'title': r[2], 'yes_odds_at_call': r[3], 'created_at': r[4]}
            for r in rows
        ]

    def mark_resolved(self, call_id: int, outcome: str, profit_loss: float = 0):
        """Mark a call as resolved"""
        self.conn.execute("""
            UPDATE calls SET resolved = TRUE, outcome = ?, profit_loss = ?
            WHERE id = ?
        """, (outcome, profit_loss, call_id))

        # Update stats
        if outcome == 'WIN':
            self.conn.execute("UPDATE stats SET wins = wins + 1, pending = pending - 1")
        elif outcome == 'LOSS':
            self.conn.execute("UPDATE stats SET losses = losses + 1, pending = pending - 1")

        self.conn.commit()


class AIAnalyzer:
    """Polyfactual Deep Research API powered analysis"""

    @staticmethod
    async def analyze_market(title: str, yes_odds: float, no_odds: float, volume: float) -> dict | None:
        """Get AI analysis from Polyfactual Deep Research API"""
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

        headers = {
            'Content-Type': 'application/json',
            'X-API-Key': POLYFACTUAL_API_KEY
        }
        data = {
            'query': query,
            'text': True
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(POLYFACTUAL_API_URL, headers=headers, json=data, timeout=120) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        if result.get('success') and result.get('data'):
                            # Extract the answer text
                            answer = result['data'].get('answer', '') or result['data'].get('text', '')
                            if isinstance(answer, dict):
                                answer = answer.get('text', str(answer))

                            # Clean up the response
                            reasoning = str(answer).strip()

                            # Remove markdown links [text](url)
                            import re
                            reasoning = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', reasoning)

                            # Remove URLs
                            reasoning = re.sub(r'https?://\S+', '', reasoning)

                            # Remove citations like (polymarket.com...)
                            reasoning = re.sub(r'\([^)]*polymarket[^)]*\)', '', reasoning)
                            reasoning = re.sub(r'\([^)]*\.com[^)]*\)', '', reasoning)

                            # Remove ** markdown bold
                            reasoning = reasoning.replace('**', '')

                            # Clean up whitespace but preserve newlines
                            lines = reasoning.split('\n')
                            lines = [' '.join(line.split()) for line in lines]
                            reasoning = '\n'.join(line for line in lines if line)

                            return {'reasoning': reasoning}
                    else:
                        print(f"Polyfactual API error: {resp.status}")
        except asyncio.TimeoutError:
            print("Polyfactual API timeout")
        except Exception as e:
            print(f"AI analysis error: {e}")

        return None


class DailyRecap:
    """Generate daily recap of all active calls with current odds"""

    def __init__(self, db: Database, poster: 'TwitterPoster'):
        self.db = db
        self.poster = poster

    async def fetch_current_odds(self, slug: str) -> dict | None:
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
                                return {
                                    'yes_odds': yes_pct,
                                    'no_odds': 100 - yes_pct,
                                    'closed': market.get('closed', False)
                                }
        except Exception as e:
            print(f"Failed to fetch odds for {slug}: {e}")
        return None

    async def generate_recap(self) -> dict:
        """Generate recap data with current odds for all active calls"""
        active_calls = self.db.get_active_calls()
        stats = self.db.get_stats()
        calls_24h = self.db.get_calls_last_24h()

        recap_data = {
            'total_active': len(active_calls),
            'calls_today': len(calls_24h),
            'stats': stats,
            'calls': []
        }

        for call in active_calls[:10]:  # Limit to 10 for recap
            current = await self.fetch_current_odds(call['slug'])
            if current:
                odds_change = current['yes_odds'] - call['yes_odds_at_call']
                recap_data['calls'].append({
                    'title': call['title'],
                    'slug': call['slug'],
                    'odds_at_call': call['yes_odds_at_call'],
                    'odds_now': current['yes_odds'],
                    'odds_change': odds_change,
                    'closed': current['closed']
                })
            else:
                recap_data['calls'].append({
                    'title': call['title'],
                    'slug': call['slug'],
                    'odds_at_call': call['yes_odds_at_call'],
                    'odds_now': None,
                    'odds_change': None,
                    'closed': False
                })

        return recap_data

    def format_recap_tweet(self, recap: dict) -> str:
        """Format recap as tweet"""
        stats = recap['stats']

        # Header
        tweet = f"""🧡 daily recap

📊 {recap['total_active']} active calls
📈 {recap['calls_today']} new today
"""

        # Show win rate if we have resolved calls
        if stats['wins'] + stats['losses'] > 0:
            tweet += f"🎯 {stats['win_rate']:.0f}% win rate ({stats['wins']}W/{stats['losses']}L)\n"

        tweet += "\n"

        # Top movers (sorted by absolute change)
        movers = [c for c in recap['calls'] if c['odds_change'] is not None]
        movers.sort(key=lambda x: abs(x['odds_change']), reverse=True)

        for call in movers[:5]:
            change = call['odds_change']
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
            change_str = f"+{change:.1f}" if change > 0 else f"{change:.1f}"

            # Truncate title
            title = call['title']
            if len(title) > 30:
                title = title[:28] + ".."

            tweet += f"{arrow} {title}\n   {call['odds_at_call']:.0f}% → {call['odds_now']:.0f}% ({change_str})\n"

        tweet += f"\n{MAIN_TWITTER_HANDLE} 🧡"

        return tweet

    def format_recap_thread(self, recap: dict) -> list[str]:
        """Format full recap as thread for more detail"""
        tweets = []

        # Main tweet
        stats = recap['stats']
        main = f"""🧡 daily recap

📊 tracking {recap['total_active']} active calls
📈 {recap['calls_today']} new calls today"""

        if stats['wins'] + stats['losses'] > 0:
            main += f"\n🎯 {stats['win_rate']:.0f}% win rate ({stats['wins']}W / {stats['losses']}L)"

        main += f"\n\nthread with all positions 👇"
        tweets.append(main)

        # Group by performance
        movers = [c for c in recap['calls'] if c['odds_change'] is not None]

        # Winners (odds moved in our favor - assuming we call YES)
        winners = [c for c in movers if c['odds_change'] > 2]
        losers = [c for c in movers if c['odds_change'] < -2]
        flat = [c for c in movers if -2 <= c['odds_change'] <= 2]

        if winners:
            winners_tweet = "📈 moving up:\n\n"
            for c in sorted(winners, key=lambda x: x['odds_change'], reverse=True)[:4]:
                title = c['title'][:35] + ".." if len(c['title']) > 35 else c['title']
                winners_tweet += f"• {title}\n  {c['odds_at_call']:.0f}% → {c['odds_now']:.0f}% (+{c['odds_change']:.1f})\n\n"
            tweets.append(winners_tweet.strip())

        if losers:
            losers_tweet = "📉 moving down:\n\n"
            for c in sorted(losers, key=lambda x: x['odds_change'])[:4]:
                title = c['title'][:35] + ".." if len(c['title']) > 35 else c['title']
                losers_tweet += f"• {title}\n  {c['odds_at_call']:.0f}% → {c['odds_now']:.0f}% ({c['odds_change']:.1f})\n\n"
            tweets.append(losers_tweet.strip())

        if flat:
            flat_tweet = "➡️ holding steady:\n\n"
            for c in flat[:4]:
                title = c['title'][:35] + ".." if len(c['title']) > 35 else c['title']
                flat_tweet += f"• {title} ({c['odds_now']:.0f}%)\n"
            tweets.append(flat_tweet.strip())

        # Footer
        tweets.append(f"that's the daily update\n\nfollow for AI-powered polymarket alpha\n\n{MAIN_TWITTER_HANDLE} 🧡")

        return tweets

    async def post_recap(self, as_thread: bool = True) -> bool:
        """Generate and post daily recap"""
        print(f"\n{'='*50}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating daily recap...")

        recap = await self.generate_recap()

        if recap['total_active'] == 0:
            print("No active calls to recap")
            return False

        print(f"Active calls: {recap['total_active']}")
        print(f"Calls today: {recap['calls_today']}")

        if as_thread and len(recap['calls']) > 3:
            tweets = self.format_recap_thread(recap)
            print(f"\nRecap thread ({len(tweets)} tweets):")
            for i, t in enumerate(tweets):
                print(f"--- Tweet {i+1} ---\n{t}\n")

            # Post thread
            tweet_id = self.poster.post(tweets[0])
            if tweet_id:
                print(f"✓ Main recap posted! ID: {tweet_id}")
                last_id = tweet_id
                for i, reply in enumerate(tweets[1:], 1):
                    reply_id = self.poster.post_reply(last_id, reply)
                    if reply_id:
                        print(f"✓ Reply {i} posted!")
                        last_id = reply_id
                    else:
                        print(f"✗ Failed to post reply {i}")
                        break
                return True
        else:
            tweet = self.format_recap_tweet(recap)
            print(f"\nRecap tweet:\n{tweet}\n")

            tweet_id = self.poster.post(tweet)
            if tweet_id:
                print(f"✓ Recap posted! ID: {tweet_id}")
                return True

        print("✗ Failed to post recap")
        return False


class PolymarketScanner:
    """Scan Polymarket for opportunities"""

    SPAM_WORDS = [
        'up or down', 'higher or lower', 'above or below',
        'am-', 'pm-', 'am et', 'pm et', ':00', ':15', ':30', ':45',
        'december 19', 'december 20', 'december 21', 'december 22',
        'december 23', 'december 24', 'december 25'
    ]

    async def fetch_events(self, limit: int = 100) -> list:
        """Fetch events sorted by volume"""
        url = f"{POLYMARKET_API}/events"
        params = {
            'limit': limit,
            'active': 'true',
            'closed': 'false',
            'order': 'volume',
            'ascending': 'false'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception as e:
            print(f"Fetch error: {e}")
        return []

    def is_valid_market(self, event: dict, posted_ids: set) -> bool:
        """Check if market is valid for posting"""
        event_id = event.get('id', '')
        title = event.get('title', '').lower()
        volume = float(event.get('volume', 0) or 0)
        markets = event.get('markets', [])

        # Already posted
        if event_id in posted_ids:
            return False

        # Spam filter
        if any(spam in title for spam in self.SPAM_WORDS):
            return False

        # Need volume
        if volume < 100000:
            return False

        # Need markets
        if not markets or len(markets) > 3:
            return False

        market = markets[0]

        # YES/NO only
        outcomes = market.get('outcomes', [])
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if len(outcomes) != 2:
            return False

        outcome_names = [str(o.get('name', o) if isinstance(o, dict) else o).lower() for o in outcomes]
        if 'yes' not in outcome_names and 'no' not in outcome_names:
            return False

        # Check end date (14+ days)
        end_date_str = market.get('endDate') or event.get('endDate')
        if end_date_str:
            try:
                end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                now = datetime.now(end_date.tzinfo) if end_date.tzinfo else datetime.now()
                if (end_date - now).days < 14:
                    return False
            except:
                pass

        # Skip obvious markets (>90% or <10%)
        outcome_prices = market.get('outcomePrices')
        if isinstance(outcome_prices, str):
            outcome_prices = json.loads(outcome_prices)
        if outcome_prices:
            yes_pct = float(outcome_prices[0]) * 100
            if yes_pct >= 90 or yes_pct <= 10:
                return False

        return True

    def parse_market_data(self, event: dict) -> dict:
        """Extract market data"""
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


class TwitterPoster:
    """Post to Twitter/X"""

    def __init__(self):
        self.client = tweepy.Client(
            bearer_token=TWITTER_BEARER_TOKEN,
            consumer_key=TWITTER_API_KEY,
            consumer_secret=TWITTER_API_SECRET,
            access_token=TWITTER_ACCESS_TOKEN,
            access_token_secret=TWITTER_ACCESS_SECRET
        )

        auth = tweepy.OAuth1UserHandler(
            TWITTER_API_KEY, TWITTER_API_SECRET,
            TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET
        )
        self.api_v1 = tweepy.API(auth)

    def format_volume(self, volume: float) -> str:
        if volume >= 1_000_000:
            return f"${volume/1_000_000:.1f}M"
        elif volume >= 1_000:
            return f"${volume/1_000:.0f}K"
        return f"${volume:.0f}"

    def create_tweet(self, market: dict) -> str:
        """Create short main tweet (no AI reasoning - that goes in reply)"""
        openers = [
            "🧡 found some edge here",
            "🧡 this one's interesting",
            "🧡 market might be off",
            "🧡 worth a look",
            "🧡 spotted something",
        ]
        opener = random.choice(openers)

        # Time text
        time_text = ""
        if market.get('days_left'):
            days = market['days_left']
            if days >= 60:
                time_text = f"~{days // 30}mo out"
            elif days >= 30:
                time_text = "~1mo out"
            else:
                time_text = f"{days}d left"

        tweet = f"""{opener}

{market['title']}

YES {market['yes_odds']:.1f}% / NO {market['no_odds']:.1f}%
vol: {self.format_volume(market['volume'])} • {time_text}

polymarket.com/event/{market['slug']}

{MAIN_TWITTER_HANDLE} 🧡"""

        return tweet

    def create_reply(self, ai_reasoning: str) -> list[str]:
        """Create reply tweet(s) with full AI analysis - splits into thread"""
        # Split by numbered sections (1), 2), etc or 1., 2., etc)
        import re

        # Parse sections from AI response
        sections = []
        current_section = ""

        for line in ai_reasoning.split('\n'):
            # Check if line starts a new section (1), 2), 1., 2., etc)
            if re.match(r'^\d+[\)\.]\s', line.strip()):
                if current_section:
                    sections.append(current_section.strip())
                current_section = line
            else:
                current_section += "\n" + line if current_section else line

        if current_section:
            sections.append(current_section.strip())

        # If no sections found, just split by length
        if len(sections) <= 1:
            sections = [ai_reasoning]

        # Format as thread with "deep research 🧠" header + 1/ 2/ 3/ 4/ format
        tweets = []
        for i, section in enumerate(sections):
            # Clean the section text (remove original numbering)
            text = re.sub(r'^\d+[\)\.]\s*', '', section.strip())

            # First tweet gets "deep research 🧠" header
            if i == 0:
                tweet = f"deep research 🧠\n\n{i+1}/\n\n{text}"
            else:
                tweet = f"{i+1}/\n\n{text}"

            # Add footer to last tweet
            if i == len(sections) - 1:
                tweet += "\n\nby @polydictions 🧡"

            tweets.append(tweet)

        return tweets

    def post(self, tweet_text: str) -> str | None:
        """Post tweet with image, return tweet ID"""
        try:
            image_path = Path(__file__).parent / "image.png"
            if image_path.exists():
                media = self.api_v1.media_upload(filename=str(image_path))
                response = self.client.create_tweet(text=tweet_text, media_ids=[media.media_id])
            else:
                response = self.client.create_tweet(text=tweet_text)
            return str(response.data['id'])
        except Exception as e:
            print(f"Post error: {e}")
            return None

    def post_reply(self, tweet_id: str, reply_text: str) -> str | None:
        """Post reply to a tweet"""
        try:
            response = self.client.create_tweet(text=reply_text, in_reply_to_tweet_id=tweet_id)
            return str(response.data['id'])
        except Exception as e:
            print(f"Reply error: {e}")
            return None


class PolydictionsAgent:
    """Main agent orchestrator"""

    def __init__(self):
        self.db = Database()
        self.scanner = PolymarketScanner()
        self.poster = TwitterPoster()
        self.recap = DailyRecap(self.db, self.poster)
        self.last_recap_date = None

    async def find_and_post(self) -> bool:
        """Find best opportunity and post it"""
        print(f"\n{'='*50}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Scanning markets...")

        events = await self.scanner.fetch_events(limit=100)
        posted_ids = self.db.get_posted_event_ids()

        valid_markets = []
        for event in events:
            if self.scanner.is_valid_market(event, posted_ids):
                market_data = self.scanner.parse_market_data(event)
                valid_markets.append(market_data)

        print(f"Found {len(valid_markets)} valid markets")

        if not valid_markets:
            print("No valid markets to post")
            return False

        # Pick random from top 10 by volume
        top_markets = sorted(valid_markets, key=lambda x: x['volume'], reverse=True)[:10]
        market = random.choice(top_markets)

        print(f"Selected: {market['title']}")
        print(f"Odds: YES {market['yes_odds']:.1f}% / NO {market['no_odds']:.1f}%")
        print(f"Volume: {self.poster.format_volume(market['volume'])}")

        # Get AI analysis
        print("Getting AI analysis...")
        ai_result = await AIAnalyzer.analyze_market(
            market['title'], market['yes_odds'], market['no_odds'], market['volume']
        )

        reasoning = ai_result['reasoning'] if ai_result else "Interesting setup here. Worth watching."
        print(f"AI: {reasoning}")

        # Create and post main tweet (short, no analysis)
        tweet_text = self.poster.create_tweet(market)
        print(f"\nMain tweet preview:\n{tweet_text}\n")

        tweet_id = self.poster.post(tweet_text)

        if tweet_id:
            print(f"✓ Main tweet posted! ID: {tweet_id}")

            # Post reply thread with full AI analysis
            reply_tweets = self.poster.create_reply(reasoning)
            print(f"\nReply thread ({len(reply_tweets)} tweets):")
            for i, rt in enumerate(reply_tweets):
                print(f"--- Tweet {i+1} ---\n{rt}\n")

            # Post as thread
            last_id = tweet_id
            for i, reply_text in enumerate(reply_tweets):
                reply_id = self.poster.post_reply(last_id, reply_text)
                if reply_id:
                    print(f"✓ Reply {i+1} posted! ID: {reply_id}")
                    last_id = reply_id
                else:
                    print(f"✗ Failed to post reply {i+1}")
                    break

            # Save to database
            self.db.add_call(
                tweet_id=tweet_id,
                event_id=market['event_id'],
                slug=market['slug'],
                title=market['title'],
                signal='ALPHA',
                yes_odds=market['yes_odds'],
                no_odds=market['no_odds'],
                volume=market['volume'],
                reasoning=reasoning
            )
            return True
        else:
            print("✗ Failed to post")
            return False

    def should_post_recap(self) -> bool:
        """Check if it's time for daily recap (once per day at ~9am UTC)"""
        now = datetime.utcnow()
        today = now.date()

        # Already posted today
        if self.last_recap_date == today:
            return False

        # Post recap between 9-10 AM UTC
        if 9 <= now.hour < 10:
            return True

        return False

    async def maybe_post_recap(self) -> bool:
        """Post daily recap if conditions are met"""
        if not self.should_post_recap():
            return False

        print("\n" + "=" * 50)
        print("TIME FOR DAILY RECAP!")
        print("=" * 50)

        success = await self.recap.post_recap(as_thread=True)
        if success:
            self.last_recap_date = datetime.utcnow().date()
        return success

    async def generate_recap_now(self) -> dict:
        """Generate recap without posting (for manual use)"""
        return await self.recap.generate_recap()

    async def post_recap_now(self, as_thread: bool = True) -> bool:
        """Force post recap now (for manual use)"""
        return await self.recap.post_recap(as_thread=as_thread)

    async def run_forever(self):
        """Run agent continuously"""
        print("=" * 50)
        print("POLYDICTIONS AGENT v2.1")
        print(f"Posting every {POST_INTERVAL_HOURS} hours")
        print("Daily recap at 9 AM UTC")
        print("=" * 50)

        while True:
            try:
                # Check for daily recap first
                await self.maybe_post_recap()

                # Then do regular alpha call
                success = await self.find_and_post()

                # Show stats
                stats = self.db.get_stats()
                print(f"\nTotal calls: {stats['total']} | Pending: {stats['pending']}")
                if stats['wins'] + stats['losses'] > 0:
                    print(f"Win rate: {stats['win_rate']:.0f}% ({stats['wins']}W/{stats['losses']}L)")

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

            # Wait for next post
            print(f"\nNext post in {POST_INTERVAL_HOURS} hours...")
            print("=" * 50)
            await asyncio.sleep(POST_INTERVAL_HOURS * 3600)


async def main():
    agent = PolydictionsAgent()
    await agent.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
