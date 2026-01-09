"""
Polydictions Agent with Automated Trading
Combines AI predictions with real Polymarket trades
"""

import os
import asyncio
import aiohttp
import json
import random
import sqlite3
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
from dotenv import load_dotenv

# Import trading system
from polymarket_trader import PolymarketTrader

# Import original agent components
import sys
sys.path.append(str(Path(__file__).parent / "polydictions-agent"))

try:
    from agent import AIAnalyzer, PolymarketScanner, TwitterPoster
except ImportError:
    print("Warning: Could not import from agent.py")
    AIAnalyzer = PolymarketScanner = TwitterPoster = None

load_dotenv()

POLYMARKET_API = "https://gamma-api.polymarket.com"
POST_INTERVAL_HOURS = int(os.getenv("AGENT_POST_INTERVAL", "4"))
TRADING_ENABLED = os.getenv("TRADING_ENABLED", "false").lower() == "true"


class TradingDatabase:
    """Extended database with trading support"""

    def __init__(self, db_path: str = "agent_trading.db"):
        self.conn = sqlite3.connect(db_path)
        self.setup()

    def setup(self):
        """Create tables for calls and trades"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY,
                tweet_id TEXT,
                event_id TEXT,
                token_id TEXT,
                slug TEXT,
                title TEXT,
                signal TEXT,
                yes_odds_at_call REAL,
                no_odds_at_call REAL,
                volume_at_call REAL,
                ai_reasoning TEXT,
                ai_confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved BOOLEAN DEFAULT FALSE,
                outcome TEXT,
                profit_loss REAL,
                traded BOOLEAN DEFAULT FALSE,
                trade_amount REAL,
                trade_order_id TEXT
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stats (
                id INTEGER PRIMARY KEY,
                total_calls INTEGER DEFAULT 0,
                total_trades INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                pending INTEGER DEFAULT 0,
                total_invested REAL DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Initialize stats if empty
        if not self.conn.execute("SELECT * FROM stats").fetchone():
            self.conn.execute("INSERT INTO stats (total_calls) VALUES (0)")

        self.conn.commit()

    def add_call(
        self,
        tweet_id: str,
        event_id: str,
        token_id: str,
        slug: str,
        title: str,
        signal: str,
        yes_odds: float,
        no_odds: float,
        volume: float,
        reasoning: str,
        confidence: float = 0
    ):
        """Record a new call"""
        self.conn.execute("""
            INSERT INTO calls (tweet_id, event_id, token_id, slug, title, signal,
                             yes_odds_at_call, no_odds_at_call, volume_at_call,
                             ai_reasoning, ai_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (tweet_id, event_id, token_id, slug, title, signal, yes_odds, no_odds, volume, reasoning, confidence))
        self.conn.execute("UPDATE stats SET total_calls = total_calls + 1, pending = pending + 1")
        self.conn.commit()

    def mark_traded(self, tweet_id: str, trade_amount: float, order_id: str):
        """Mark a call as traded"""
        self.conn.execute("""
            UPDATE calls SET traded = TRUE, trade_amount = ?, trade_order_id = ?
            WHERE tweet_id = ?
        """, (trade_amount, order_id, tweet_id))
        self.conn.execute("UPDATE stats SET total_trades = total_trades + 1, total_invested = total_invested + ?", (trade_amount,))
        self.conn.commit()

    def get_posted_event_ids(self) -> set:
        """Get all posted event IDs"""
        rows = self.conn.execute("SELECT event_id FROM calls").fetchall()
        return {row[0] for row in rows}

    def get_stats(self) -> dict:
        """Get overall stats"""
        row = self.conn.execute("SELECT total_calls, total_trades, wins, losses, pending, total_invested, total_pnl FROM stats").fetchone()
        return {
            'total_calls': row[0],
            'total_trades': row[1],
            'wins': row[2],
            'losses': row[3],
            'pending': row[4],
            'total_invested': row[5],
            'total_pnl': row[6],
            'win_rate': (row[2] / (row[2] + row[3]) * 100) if (row[2] + row[3]) > 0 else 0
        }


class TradingAgent:
    """Agent that posts predictions AND trades on them"""

    def __init__(self, trading_enabled: bool = False):
        self.db = TradingDatabase()
        self.scanner = PolymarketScanner()
        self.poster = TwitterPoster() if TwitterPoster else None
        self.trading_enabled = trading_enabled

        # Initialize trader if enabled
        self.trader = None
        if self.trading_enabled:
            try:
                self.trader = PolymarketTrader()
                print("✓ Trading enabled")
            except Exception as e:
                print(f"✗ Could not initialize trader: {e}")
                self.trading_enabled = False

    def extract_confidence_from_reasoning(self, reasoning: str) -> float:
        """Extract confidence level from AI reasoning"""
        # Look for patterns like "high confidence", "mid confidence", "low confidence"
        reasoning_lower = reasoning.lower()

        if 'high confidence' in reasoning_lower:
            return 80.0
        elif 'mid confidence' in reasoning_lower or 'medium confidence' in reasoning_lower:
            return 65.0
        elif 'low confidence' in reasoning_lower:
            return 55.0

        # Try to extract percentage
        match = re.search(r'(\d+)%?\s*confidence', reasoning_lower)
        if match:
            return float(match.group(1))

        # Default to moderate confidence
        return 65.0

    def extract_signal_from_reasoning(self, reasoning: str, yes_odds: float, no_odds: float) -> str:
        """Extract YES/NO signal from AI reasoning"""
        reasoning_lower = reasoning.lower()

        # Look for explicit recommendation
        if 'recommendation:' in reasoning_lower:
            after_rec = reasoning_lower.split('recommendation:')[1]
            if 'yes' in after_rec[:50]:
                return 'YES'
            elif 'no' in after_rec[:50]:
                return 'NO'

        # Check if reasoning favors YES or NO
        yes_count = reasoning_lower.count(' yes ') + reasoning_lower.count('yes ') + reasoning_lower.count(' yes.') + reasoning_lower.count(' yes,')
        no_count = reasoning_lower.count(' no ') + reasoning_lower.count('no ') + reasoning_lower.count(' no.') + reasoning_lower.count(' no,')

        if yes_count > no_count:
            return 'YES'
        elif no_count > yes_count:
            return 'NO'

        # Default to side with better odds
        return 'YES' if yes_odds < no_odds else 'NO'

    async def find_and_post_with_trade(self) -> bool:
        """Find market, post about it, and optionally trade on it"""
        print(f"\n{'='*50}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Scanning markets...")

        # Fetch events
        events = await self.scanner.fetch_events(limit=100)
        posted_ids = self.db.get_posted_event_ids()

        valid_markets = []
        for event in events:
            if self.scanner.is_valid_market(event, posted_ids):
                market_data = self.scanner.parse_market_data(event)
                # Add token_id for trading
                if event.get('markets') and len(event['markets']) > 0:
                    market_data['token_id'] = event['markets'][0].get('clobTokenIds', [''])[0]
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
        print(f"Volume: ${market['volume']:,.0f}")

        # Get AI analysis
        print("Getting AI analysis...")
        ai_result = await AIAnalyzer.analyze_market(
            market['title'], market['yes_odds'], market['no_odds'], market['volume']
        )

        reasoning = ai_result['reasoning'] if ai_result else "Interesting setup here. Worth watching."
        print(f"AI: {reasoning[:200]}...")

        # Extract signal and confidence
        signal = self.extract_signal_from_reasoning(reasoning, market['yes_odds'], market['no_odds'])
        confidence = self.extract_confidence_from_reasoning(reasoning)

        print(f"\nExtracted signal: {signal} ({confidence:.0f}% confidence)")

        # Create and post tweet
        if self.poster:
            tweet_text = self.create_tweet(market, signal, confidence)
            print(f"\nTweet:\n{tweet_text}\n")

            tweet_id = self.poster.post(tweet_text)

            if tweet_id:
                print(f"✓ Tweet posted! ID: {tweet_id}")

                # Save to database
                self.db.add_call(
                    tweet_id=tweet_id,
                    event_id=market['event_id'],
                    token_id=market.get('token_id', ''),
                    slug=market['slug'],
                    title=market['title'],
                    signal=signal,
                    yes_odds=market['yes_odds'],
                    no_odds=market['no_odds'],
                    volume=market['volume'],
                    reasoning=reasoning,
                    confidence=confidence
                )

                # Execute trade if enabled
                if self.trading_enabled and self.trader and market.get('token_id'):
                    print(f"\n{'='*50}")
                    print("EXECUTING TRADE")
                    print(f"{'='*50}")

                    trade_result = self.trader.trade_with_ai_signal(
                        token_id=market['token_id'],
                        market_title=market['title'],
                        ai_signal=signal,
                        ai_confidence=confidence,
                        market_yes_price=market['yes_odds'] / 100,
                        market_no_price=market['no_odds'] / 100,
                        tweet_id=tweet_id
                    )

                    if trade_result and trade_result.get('success'):
                        order_id = trade_result.get('order_id')
                        amount = trade_result.get('amount')
                        self.db.mark_traded(tweet_id, amount, order_id)
                        print(f"✓ Trade executed: ${amount:.2f}")
                    else:
                        error = trade_result.get('error') if trade_result else 'unknown error'
                        print(f"✗ Trade not executed: {error}")

                return True
            else:
                print("✗ Failed to post tweet")
                return False
        else:
            print("✗ Twitter poster not available")
            return False

    def create_tweet(self, market: Dict, signal: str, confidence: float) -> str:
        """Create tweet with prediction and signal"""
        openers = [
            "🧡 found some edge",
            "🧡 AI sees value here",
            "🧡 market looks off",
            "🧡 spotted opportunity"
        ]
        opener = random.choice(openers)

        # Confidence indicator
        conf_emoji = "🔥" if confidence >= 75 else "💎" if confidence >= 65 else "👀"

        tweet = f"""{opener}

{market['title']}

📊 YES {market['yes_odds']:.1f}% / NO {market['no_odds']:.1f}%
vol: ${market['volume']/1000000:.1f}M

🤖 AI call: {signal} {conf_emoji}
confidence: {confidence:.0f}%

polymarket.com/event/{market['slug']}"""

        if self.trading_enabled:
            tweet += "\n\n💰 trading this live"

        tweet += "\n\n@polydictions 🧡"

        return tweet[:280]

    async def run_forever(self):
        """Run agent continuously with optional trading"""
        print("=" * 50)
        print("POLYDICTIONS TRADING AGENT")
        print(f"Posting every {POST_INTERVAL_HOURS} hours")
        print(f"Trading: {'ENABLED' if self.trading_enabled else 'DISABLED'}")
        print("=" * 50)

        while True:
            try:
                # Post and potentially trade
                success = await self.find_and_post_with_trade()

                # Show stats
                stats = self.db.get_stats()
                print(f"\n📊 Stats:")
                print(f"  Total calls: {stats['total_calls']}")
                print(f"  Total trades: {stats['total_trades']}")
                print(f"  Pending: {stats['pending']}")

                if self.trading_enabled and self.trader:
                    # Update positions and show portfolio
                    self.trader.update_all_position_prices()
                    self.trader.print_portfolio_status()

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

            # Wait for next post
            print(f"\nNext post in {POST_INTERVAL_HOURS} hours...")
            print("=" * 50)
            await asyncio.sleep(POST_INTERVAL_HOURS * 3600)


async def main():
    """Main entry point"""
    print("\n" + "=" * 50)
    print("POLYDICTIONS AI TRADING AGENT")
    print("=" * 50)
    print("\nThis agent will:")
    print("1. Scan Polymarket for opportunities")
    print("2. Get AI analysis via Polyfactual API")
    print("3. Post predictions on Twitter")

    if TRADING_ENABLED:
        print("4. Execute REAL TRADES on Polymarket")
        print("\n⚠️  WARNING: REAL MONEY TRADING IS ENABLED")
        print("   Make sure you have:")
        print("   - POLYMARKET_PRIVATE_KEY set")
        print("   - POLYMARKET_FUNDER_ADDRESS set")
        print("   - Sufficient USDC balance")
        print("   - Correct allowances (for EOA wallets)")
        print(f"\n{'='*50}\n")

        # Confirm
        confirmation = input("Type 'YES' to start with trading enabled: ")
        if confirmation != "YES":
            print("Aborted.")
            return
    else:
        print("4. NO TRADING (set TRADING_ENABLED=true to enable)")
        print(f"\n{'='*50}\n")

    agent = TradingAgent(trading_enabled=TRADING_ENABLED)
    await agent.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
