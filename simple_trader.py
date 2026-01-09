"""
Simple Auto-Trader for Polydictions
Reads existing predictions and trades on them automatically
No Twitter posting - just pure trading execution
"""

import os
import asyncio
import sqlite3
import json
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dotenv import load_dotenv

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import MarketOrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY
except ImportError:
    print("Error: py-clob-client not installed")
    ClobClient = None

load_dotenv()


class SmartParser:
    """Use Claude to parse AI reasoning into structured signal"""

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.api_url = "https://api.anthropic.com/v1/messages"

    async def parse_prediction(self, reasoning: str, market_title: str) -> Dict:
        """Parse AI reasoning into signal + confidence using Claude - NO FALLBACK"""

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set - cannot parse predictions")

        prompt = f"""Parse this prediction analysis for the market: "{market_title}"

AI Analysis:
{reasoning}

Extract:
1. Signal: Should we bet YES or NO?
2. Confidence: How confident is the AI? (0-100)

Respond ONLY with valid JSON in this exact format:
{{"signal": "YES", "confidence": 75}}

or

{{"signal": "NO", "confidence": 65}}"""

        try:
            import aiohttp

            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }

            data = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": prompt}]
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, json=data, timeout=30) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        text = result.get("content", [{}])[0].get("text", "")

                        # Parse JSON from Claude response
                        try:
                            parsed = json.loads(text.strip())
                            signal = parsed.get("signal", "YES").upper()
                            confidence = float(parsed.get("confidence", 0))

                            # Validate
                            if signal not in ["YES", "NO"]:
                                raise ValueError(f"Invalid signal: {signal}")
                            if confidence < 0 or confidence > 100:
                                raise ValueError(f"Invalid confidence: {confidence}")

                            return {"signal": signal, "confidence": confidence}
                        except (json.JSONDecodeError, ValueError) as e:
                            print(f"[ERROR] Claude response invalid: {text[:200]}")
                            print(f"[ERROR] Parse error: {e}")
                            return None
                    else:
                        error_text = await resp.text()
                        print(f"[ERROR] Claude API error {resp.status}: {error_text[:200]}")
                        return None

        except Exception as e:
            print(f"[ERROR] Failed to call Claude: {e}")
            return None


class ConservativeRiskManager:
    """Ultra-conservative risk management for $100 budget"""

    def __init__(
        self,
        total_budget: float = 100.0,
        max_bet: float = 5.0,
        max_positions: int = 10,  # Max 10 open positions
        min_confidence: float = 70.0,  # Only high confidence
        min_edge: float = 10.0  # Only strong edge
    ):
        self.total_budget = total_budget
        self.max_bet = max_bet
        self.max_positions = max_positions
        self.min_confidence = min_confidence
        self.min_edge = min_edge

    def calculate_bet_size(
        self,
        confidence: float,
        current_capital: float,
        open_positions: int
    ) -> float:
        """Calculate bet size - scales with confidence"""

        # Don't bet if too many positions
        if open_positions >= self.max_positions:
            return 0

        if confidence < self.min_confidence:
            return 0

        # Scale linearly with confidence
        # 70% confidence → $1.00 (20% of max)
        # 80% confidence → $2.00 (40% of max)
        # 90% confidence → $3.50 (70% of max)
        # 100% confidence → $5.00 (100% of max)

        confidence_normalized = (confidence - self.min_confidence) / (100 - self.min_confidence)
        bet = self.max_bet * (0.2 + 0.8 * confidence_normalized)

        # Never bet more than 5% of remaining capital
        max_risk = current_capital * 0.05
        bet = min(bet, max_risk)

        # Round to nearest $0.50
        bet = round(bet * 2) / 2

        return max(bet, 0)

    def should_trade(
        self,
        signal: str,
        confidence: float,
        ai_price: float,
        market_price: float,
        current_capital: float,
        open_positions: int
    ) -> tuple[bool, str]:
        """Ultra-strict trading criteria"""

        # Check confidence
        if confidence < self.min_confidence:
            return False, f"confidence too low: {confidence:.1f}% < {self.min_confidence}%"

        # Check edge
        edge = abs(ai_price - market_price) / market_price * 100
        if edge < self.min_edge:
            return False, f"edge too small: {edge:.1f}% < {self.min_edge}%"

        # Check capital
        if current_capital < self.max_bet:
            return False, f"insufficient capital: ${current_capital:.2f}"

        # Check positions
        if open_positions >= self.max_positions:
            return False, f"max positions reached: {open_positions}"

        return True, "all checks passed"


class SimpleTrader:
    """Simple trader that reads predictions and executes trades"""

    def __init__(self):
        self.parser = SmartParser()
        self.risk = ConservativeRiskManager()

        # Setup trading
        private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
        funder = os.getenv("POLYMARKET_FUNDER_ADDRESS")
        builder_code = os.getenv("POLYMARKET_BUILDER_CODE") or os.getenv("POLYMARKET_BUILDERS_KEY")

        if not private_key or not funder:
            raise ValueError("Missing POLYMARKET_PRIVATE_KEY or POLYMARKET_FUNDER_ADDRESS")

        if not ClobClient:
            raise ImportError("py-clob-client not installed")

        self.client = ClobClient(
            "https://clob.polymarket.com",
            key=private_key,
            chain_id=137,
            signature_type=1,  # Adjust based on wallet type
            funder=funder
        )

        self.client.set_api_creds(self.client.create_or_derive_api_creds())

        # Set builder code for attribution
        if builder_code:
            try:
                # Try to set builder code (method may vary by version)
                if hasattr(self.client, 'set_builder_code'):
                    self.client.set_builder_code(builder_code)
                    print(f"[OK] Builder code set: {builder_code}")
                else:
                    print("⚠ Builder code not supported in this client version")
            except Exception as e:
                print(f"⚠ Could not set builder code: {e}")

        # Setup DB for tracking
        self.setup_trading_db()

        print(f"[OK] Trader initialized")
        print(f"  Budget: ${self.risk.total_budget}")
        print(f"  Max bet: ${self.risk.max_bet}")
        print(f"  Min confidence: {self.risk.min_confidence}%")

    def setup_trading_db(self):
        """Create DB for tracking trades"""
        self.conn = sqlite3.connect("simple_trading.db")

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                token_id TEXT,
                market_title TEXT,
                signal TEXT,
                confidence REAL,
                bet_size REAL,
                market_price REAL,
                order_id TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS capital (
                id INTEGER PRIMARY KEY,
                current_capital REAL,
                total_invested REAL,
                total_pnl REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Initialize capital
        if not self.conn.execute("SELECT * FROM capital").fetchone():
            self.conn.execute(
                "INSERT INTO capital (id, current_capital, total_invested, total_pnl) VALUES (1, ?, 0, 0)",
                (self.risk.total_budget,)
            )

        self.conn.commit()

    def get_current_capital(self) -> float:
        """Get current available capital"""
        row = self.conn.execute("SELECT current_capital FROM capital WHERE id = 1").fetchone()
        return row[0] if row else self.risk.total_budget

    def get_open_positions(self) -> int:
        """Count open positions"""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM trades WHERE status IN ('pending', 'filled')"
        ).fetchone()
        return row[0] if row else 0

    def get_market_price(self, token_id: str) -> Optional[float]:
        """Get current market price"""
        try:
            price = self.client.get_price(token_id, side="BUY")
            return float(price)
        except Exception as e:
            print(f"Error getting price: {e}")
            return None

    async def trade_on_prediction(
        self,
        prediction_id: int,
        token_id: str,
        market_title: str,
        reasoning: str,
        market_yes_odds: float,
        market_no_odds: float
    ) -> bool:
        """Execute trade based on prediction"""

        print(f"\n{'='*60}")
        print(f"Processing prediction: {market_title[:50]}...")

        # Show AI reasoning preview
        print(f"\n  AI Reasoning (preview):")
        reasoning_preview = reasoning[:300] + "..." if len(reasoning) > 300 else reasoning
        for line in reasoning_preview.split('\n')[:5]:  # Show first 5 lines
            if line.strip():
                print(f"    {line[:70]}")

        # Parse with Claude
        parsed = await self.parser.parse_prediction(reasoning, market_title)

        if not parsed:
            print(f"  [SKIP] Claude failed to parse - need valid AI reasoning from Polyfactual")
            return False

        signal = parsed['signal']
        confidence = parsed['confidence']

        print(f"\n  [OK] Parsed Signal: {signal}")
        print(f"  [OK] Confidence: {confidence:.1f}%")

        # Determine AI price and market price
        if signal == "YES":
            ai_price = confidence / 100
            market_price = market_yes_odds / 100
        else:
            ai_price = (100 - confidence) / 100
            market_price = market_no_odds / 100

        # Get current state
        current_capital = self.get_current_capital()
        open_positions = self.get_open_positions()

        print(f"  AI price: {ai_price:.2%}")
        print(f"  Market price: {market_price:.2%}")
        print(f"  Edge: {abs(ai_price - market_price) / market_price * 100:.1f}%")
        print(f"  Capital: ${current_capital:.2f}")
        print(f"  Open positions: {open_positions}")

        # Risk check
        should_trade, reason = self.risk.should_trade(
            signal=signal,
            confidence=confidence,
            ai_price=ai_price,
            market_price=market_price,
            current_capital=current_capital,
            open_positions=open_positions
        )

        if not should_trade:
            print(f"  [X] SKIP: {reason}")
            return False

        # Calculate bet size
        bet_size = self.risk.calculate_bet_size(confidence, current_capital, open_positions)

        if bet_size < 1.0:
            print(f"  [X] SKIP: bet too small (${bet_size:.2f})")
            return False

        print(f"  [OK] TRADE: ${bet_size:.2f}")

        # Execute trade
        try:
            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=bet_size,
                side=BUY,
                order_type=OrderType.FOK
            )

            signed = self.client.create_market_order(order_args)
            resp = self.client.post_order(signed, OrderType.FOK)

            if resp.get('success'):
                order_id = resp.get('orderId', 'unknown')

                print(f"  [OK] ORDER FILLED: {order_id}")

                # Record trade
                self.conn.execute("""
                    INSERT INTO trades (prediction_id, token_id, market_title, signal, confidence, bet_size, market_price, order_id, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'filled')
                """, (prediction_id, token_id, market_title, signal, confidence, bet_size, market_price, order_id))

                # Update capital
                self.conn.execute(
                    "UPDATE capital SET current_capital = current_capital - ?, total_invested = total_invested + ? WHERE id = 1",
                    (bet_size, bet_size)
                )

                self.conn.commit()

                return True
            else:
                error = resp.get('errorMsg', 'Unknown error')
                print(f"  [X] ORDER FAILED: {error}")

                # Record failed trade
                self.conn.execute("""
                    INSERT INTO trades (prediction_id, token_id, market_title, signal, confidence, bet_size, market_price, order_id, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'failed')
                """, (prediction_id, token_id, market_title, signal, confidence, bet_size, market_price, error))

                self.conn.commit()

                return False

        except Exception as e:
            print(f"  [X] ERROR: {e}")
            return False

    def get_untraded_predictions(self) -> List[Dict]:
        """Get predictions from polydictions-agent DB that haven't been traded yet"""

        # Connect to predictions DB
        try:
            pred_conn = sqlite3.connect("polydictions-agent/calls.db")
        except:
            print("Could not find polydictions-agent/calls.db")
            return []

        # Get predictions from last 30 days that we haven't traded yet
        cutoff = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')

        rows = pred_conn.execute("""
            SELECT id, event_id, slug, title, yes_odds_at_call, no_odds_at_call, ai_reasoning, created_at
            FROM calls
            WHERE created_at > ?
            AND resolved = FALSE
            ORDER BY created_at DESC
        """, (cutoff,)).fetchall()

        pred_conn.close()

        # Filter out already traded
        traded_pred_ids = set(
            r[0] for r in self.conn.execute("SELECT prediction_id FROM trades").fetchall()
        )

        predictions = []
        for row in rows:
            pred_id = row[0]
            if pred_id in traded_pred_ids:
                continue

            # Need to get token_id - fetch from Polymarket API
            slug = row[2]
            token_id = self.get_token_id_from_slug(slug)

            if not token_id:
                continue

            predictions.append({
                'id': pred_id,
                'event_id': row[1],
                'slug': slug,
                'title': row[3],
                'yes_odds': row[4],
                'no_odds': row[5],
                'reasoning': row[6],
                'created_at': row[7],
                'token_id': token_id
            })

        return predictions

    def get_token_id_from_slug(self, slug: str) -> Optional[str]:
        """Fetch token_id from Polymarket API"""
        try:
            resp = requests.get(
                f"https://gamma-api.polymarket.com/events",
                params={"slug": slug},
                timeout=10
            )

            if resp.status_code == 200:
                events = resp.json()
                if events and len(events) > 0:
                    markets = events[0].get('markets', [])
                    if markets and len(markets) > 0:
                        token_ids = markets[0].get('clobTokenIds', [])
                        if token_ids and len(token_ids) > 0:
                            return token_ids[0]
        except Exception as e:
            print(f"Error fetching token_id for {slug}: {e}")

        return None

    async def run_once(self):
        """Process all untraded predictions once"""

        print("\n" + "="*60)
        print("SIMPLE TRADER - Processing predictions")
        print("="*60)

        predictions = self.get_untraded_predictions()

        print(f"\nFound {len(predictions)} untraded predictions")

        if not predictions:
            print("Nothing to trade")
            return

        traded_count = 0

        for pred in predictions:
            success = await self.trade_on_prediction(
                prediction_id=pred['id'],
                token_id=pred['token_id'],
                market_title=pred['title'],
                reasoning=pred['reasoning'],
                market_yes_odds=pred['yes_odds'],
                market_no_odds=pred['no_odds']
            )

            if success:
                traded_count += 1

            # Small delay between trades
            await asyncio.sleep(2)

        print(f"\n{'='*60}")
        print(f"Session complete: {traded_count}/{len(predictions)} trades executed")
        print(f"Capital remaining: ${self.get_current_capital():.2f}")
        print("="*60)

    async def run_forever(self, interval_hours: int = 2):
        """Run continuously, checking for new predictions"""

        print("\n" + "="*60)
        print("SIMPLE TRADER - Continuous Mode")
        print(f"Checking every {interval_hours} hour(s)")
        print("Press Ctrl+C to stop")
        print("="*60)

        iteration = 1

        while True:
            try:
                print(f"\n{'='*60}")
                print(f"Iteration #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*60}")

                await self.run_once()

                iteration += 1

                print(f"\nNext check in {interval_hours} hour(s)...")
                await asyncio.sleep(interval_hours * 3600)

            except KeyboardInterrupt:
                print("\n\nStopped by user")
                break
            except Exception as e:
                print(f"Error in run cycle: {e}")
                import traceback
                traceback.print_exc()
                print(f"\nRetrying in {interval_hours} hour(s)...")
                await asyncio.sleep(interval_hours * 3600)


async def main():
    """Main entry point"""

    print("\n" + "="*60)
    print("POLYDICTIONS SIMPLE AUTO-TRADER")
    print("="*60)
    print("\nThis trader:")
    print("  • Reads predictions from polydictions-agent/calls.db")
    print("  • Uses Claude to parse signal + confidence")
    print("  • Executes trades automatically")
    print("  • Conservative: $100 budget, $5 max bet")
    print("="*60 + "\n")

    trader = SimpleTrader()

    # Run once or continuously?
    mode = input("Run mode? (once/continuous): ").strip().lower()

    if mode == "continuous":
        await trader.run_forever()  # Uses default 2 hours
    else:
        await trader.run_once()


if __name__ == "__main__":
    asyncio.run(main())
