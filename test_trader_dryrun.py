"""
Dry-run Simple Trader
Shows what would be traded WITHOUT placing real trades
"""

import os
import asyncio
import sqlite3
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Import the parser from simple_trader
import sys
sys.path.insert(0, os.path.dirname(__file__))

try:
    from simple_trader import SmartParser, ConservativeRiskManager
except ImportError:
    print("Error: Could not import from simple_trader.py")
    print("Make sure simple_trader.py is in the same directory")
    sys.exit(1)


class DryRunTrader:
    """Test version that shows what would be traded"""

    def __init__(self):
        self.parser = SmartParser()
        self.risk = ConservativeRiskManager()

        print("\n" + "="*60)
        print("DRY-RUN MODE - NO REAL TRADES")
        print("="*60)
        print(f"Budget: ${self.risk.total_budget}")
        print(f"Max bet: ${self.risk.max_bet}")
        print(f"Min confidence: {self.risk.min_confidence}%")
        print(f"Min edge: {self.risk.min_edge}%")
        print("="*60 + "\n")

    def get_predictions(self):
        """Get unresolved predictions from database"""
        db_path = "polydictions-agent/calls.db"

        if not os.path.exists(db_path):
            print(f"Error: Database not found: {db_path}")
            return []

        conn = sqlite3.connect(db_path)

        cutoff = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')

        rows = conn.execute("""
            SELECT id, slug, title, yes_odds_at_call, no_odds_at_call, ai_reasoning, created_at
            FROM calls
            WHERE created_at > ?
            AND resolved = FALSE
            ORDER BY created_at DESC
        """, (cutoff,)).fetchall()

        conn.close()

        predictions = []
        for row in rows:
            predictions.append({
                'id': row[0],
                'slug': row[1],
                'title': row[2],
                'yes_odds': row[3],
                'no_odds': row[4],
                'reasoning': row[5],
                'created_at': row[6]
            })

        return predictions

    async def analyze_prediction(self, pred, capital, positions):
        """Analyze a prediction without trading"""

        print("\n" + "="*60)
        print(f"Prediction: {pred['title'][:55]}")
        print("="*60)

        # Show AI reasoning
        print(f"\nAI Reasoning:")
        print("-" * 60)
        reasoning = pred.get('reasoning', '') or ''
        if reasoning:
            reasoning_preview = reasoning[:500] + "..." if len(reasoning) > 500 else reasoning
            print(reasoning_preview)
        else:
            print("(No reasoning available)")
        print("-" * 60)

        # Parse with Claude
        parsed = await self.parser.parse_prediction(
            reasoning or "No reasoning provided",
            pred['title']
        )

        if not parsed:
            print(f"\n[SKIP] Claude failed to parse")
            print(f"Need valid AI reasoning from Polyfactual API")
            return None

        signal = parsed['signal']
        confidence = parsed['confidence']

        print(f"\n[OK] Parsed Signal: {signal}")
        print(f"[OK] Confidence: {confidence:.1f}%")

        # Determine prices
        if signal == "YES":
            ai_price = confidence / 100
            market_price = pred['yes_odds'] / 100
        else:
            ai_price = (100 - confidence) / 100
            market_price = pred['no_odds'] / 100

        edge = abs(ai_price - market_price) / market_price * 100

        print(f"\n  AI price: {ai_price:.2%}")
        print(f"  Market price: {market_price:.2%}")
        print(f"  Edge: {edge:.1f}%")
        print(f"\n  Current capital: ${capital:.2f}")
        print(f"  Open positions: {positions}")

        # Risk check
        should_trade, reason = self.risk.should_trade(
            signal=signal,
            confidence=confidence,
            ai_price=ai_price,
            market_price=market_price,
            current_capital=capital,
            open_positions=positions
        )

        if not should_trade:
            print(f"\n  [SKIP] {reason}")
            return None

        # Calculate bet
        bet_size = self.risk.calculate_bet_size(confidence, capital, positions)

        if bet_size < 1.0:
            print(f"\n  [SKIP] Bet too small: ${bet_size:.2f}")
            return None

        print(f"\n  [WOULD TRADE]")
        print(f"    Amount: ${bet_size:.2f}")
        print(f"    Side: {signal}")
        print(f"    Expected shares: ~{bet_size / market_price:.2f}")

        return {
            'signal': signal,
            'confidence': confidence,
            'bet_size': bet_size,
            'market_price': market_price,
            'ai_price': ai_price,
            'edge': edge
        }

    async def run(self):
        """Run dry-run analysis"""

        predictions = self.get_predictions()

        if not predictions:
            print("No predictions found in database")
            return

        print(f"Found {len(predictions)} unresolved predictions\n")

        capital = self.risk.total_budget
        positions = 0

        would_trade = []

        for pred in predictions:
            result = await self.analyze_prediction(pred, capital, positions)

            if result:
                would_trade.append({
                    'title': pred['title'],
                    **result
                })
                capital -= result['bet_size']
                positions += 1

        # Summary
        print("\n" + "="*60)
        print("DRY-RUN SUMMARY")
        print("="*60)

        if would_trade:
            print(f"\nWould execute {len(would_trade)} trades:\n")

            total_invested = 0

            for i, trade in enumerate(would_trade, 1):
                print(f"{i}. {trade['title'][:50]}")
                print(f"   Signal: {trade['signal']}")
                print(f"   Confidence: {trade['confidence']:.1f}%")
                print(f"   Edge: {trade['edge']:.1f}%")
                print(f"   Bet: ${trade['bet_size']:.2f}")
                print()

                total_invested += trade['bet_size']

            print(f"Total would invest: ${total_invested:.2f}")
            print(f"Capital remaining: ${capital:.2f}")
        else:
            print("\nWould NOT execute any trades")
            print("All predictions failed risk checks")

        print("\n" + "="*60)
        print("This was a DRY-RUN - no real trades were placed")
        print("="*60 + "\n")


async def run_continuous():
    """Run continuously every minute for testing"""

    trader = DryRunTrader()

    print("\n" + "="*60)
    print("CONTINUOUS MODE - Checking every 1 minute")
    print("Press Ctrl+C to stop")
    print("="*60)

    iteration = 1

    while True:
        try:
            print(f"\n{'='*60}")
            print(f"Iteration #{iteration} - {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*60}")

            await trader.run()

            iteration += 1

            print(f"\n{'='*60}")
            print(f"Next check in 1 minute... (Ctrl+C to stop)")
            print(f"{'='*60}")

            await asyncio.sleep(60)  # 1 minute

        except KeyboardInterrupt:
            print("\n\nStopped by user")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Continuing...")
            await asyncio.sleep(60)


async def main():
    """Main entry point"""

    print("\n" + "="*60)
    print("SIMPLE TRADER - DRY-RUN TEST")
    print("="*60)
    print("\nThis will analyze predictions and show what WOULD be traded")
    print("WITHOUT placing any real trades\n")

    # Check if ANTHROPIC_API_KEY is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set")
        print("Will use simple parsing instead of Claude")
        print()

    mode = input("Run mode? (once/continuous): ").strip().lower()

    if mode == "continuous":
        await run_continuous()
    else:
        trader = DryRunTrader()
        await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
