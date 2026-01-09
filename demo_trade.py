"""
Demo Trade - Beautiful analysis + fake bet for screenshots
"""

import os
import asyncio
import random
from datetime import datetime
from dotenv import load_dotenv
import aiohttp
import requests

load_dotenv()


class DemoTrader:
    """Shows beautiful trade analysis for screenshots"""

    def __init__(self):
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.polyfactual_key = os.getenv("POLYFACTUAL_API_KEY")
        self.polyfactual_url = os.getenv("POLYFACTUAL_API_URL",
                                         "https://deep-research-api.thekid-solana.workers.dev/answer")

        if not self.anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        if not self.polyfactual_key:
            raise ValueError("POLYFACTUAL_API_KEY not set")

    def get_market(self, slug=None):
        """Get a specific market or random market - only ACTIVE ones"""

        if slug:
            # Get specific market by slug
            resp = requests.get(
                "https://gamma-api.polymarket.com/events",
                params={"slug": slug},
                timeout=10
            )
            if resp.status_code == 200:
                events = resp.json()
                return events[0] if events else None
            return None

        # Get active markets only
        resp = requests.get(
            "https://gamma-api.polymarket.com/events",
            params={"limit": 100, "active": True, "closed": False},
            timeout=10
        )

        if resp.status_code != 200:
            return None

        events = resp.json()
        if not events:
            return None

        # Filter: only open markets (not closed, not resolved)
        active_markets = []
        for event in events:
            if event.get('closed') == True:
                continue
            if event.get('resolved') == True:
                continue
            # Check end date is in future
            end_date = event.get('endDate', '')
            if end_date and end_date < datetime.now().isoformat():
                continue
            active_markets.append(event)

        if not active_markets:
            print("[X] No active markets found, using any open market...")
            # Fallback - just skip closed ones
            active_markets = [e for e in events if not e.get('closed')]

        return random.choice(active_markets) if active_markets else None

    async def get_polyfactual_analysis(self, question):
        """Get deep research from Polyfactual"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.polyfactual_url,
                    json={
                        "question": question,
                        "api_key": self.polyfactual_key
                    },
                    timeout=60
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get('answer', '')
                    else:
                        error = await resp.text()
                        print(f"[X] Polyfactual error {resp.status}: {error[:200]}")
                        return None
        except asyncio.TimeoutError:
            print("[X] Polyfactual timeout (60s)")
            return None
        except Exception as e:
            print(f"[X] Polyfactual error: {e}")
            return None

    def generate_fake_analysis(self, market_title, yes_price, no_price):
        """Generate fake analysis for demo when Polyfactual fails"""
        # For demo purposes - create plausible analysis
        if yes_price > 60:
            signal = "YES"
            confidence = min(95, yes_price + random.randint(5, 15))
        elif no_price > 60:
            signal = "NO"
            confidence = min(95, 100 - yes_price + random.randint(5, 15))
        else:
            signal = random.choice(["YES", "NO"])
            confidence = random.randint(72, 88)

        return {
            "signal": signal,
            "confidence": confidence,
            "reasoning": f"Based on current market sentiment and recent developments, the probability of {signal} outcome appears undervalued. Market pricing suggests {yes_price:.0f}% for YES, but our analysis indicates {confidence}% confidence in {signal}.",
            "key_factors": [
                "Recent news and market sentiment analysis",
                "Historical pattern recognition",
                "Risk-adjusted probability assessment"
            ]
        }

    async def analyze_with_claude(self, market_title, polyfactual_reasoning, yes_price, no_price):
        """Analyze with Claude Sonnet"""

        prompt = f"""You are analyzing a prediction market for trading.

Market: {market_title}
Current Prices: YES {yes_price:.1f}% | NO {no_price:.1f}%

Research Data:
{polyfactual_reasoning}

Provide your analysis in JSON format:
{{
    "signal": "YES or NO",
    "confidence": 85,
    "reasoning": "2-3 sentences explaining your position clearly and concisely",
    "key_factors": ["Factor 1", "Factor 2", "Factor 3"]
}}

Be decisive and confident in your analysis."""

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.anthropic_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "claude-3-5-sonnet-20241022",
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": prompt}]
                    },
                    timeout=30
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        text = data.get("content", [{}])[0].get("text", "")

                        # Parse JSON
                        import json
                        if "```json" in text:
                            text = text.split("```json")[1].split("```")[0].strip()
                        elif "```" in text:
                            text = text.split("```")[1].split("```")[0].strip()

                        return json.loads(text)
                    else:
                        return None
        except Exception as e:
            print(f"[X] Claude error: {e}")
            return None

    def display_trade(self, market, analysis):
        """Beautiful display for screenshot"""

        title = market.get('title', 'Unknown')
        slug = market.get('slug', '')

        # Get prices
        outcomes = market.get('markets', [])
        yes_price = 50.0
        no_price = 50.0

        for outcome in outcomes:
            outcome_name = outcome.get('outcome', '').upper()
            prices = outcome.get('outcomePrices', '0.5')
            # Handle both string and list formats
            if isinstance(prices, list):
                price_str = prices[0] if prices else '0.5'
            else:
                price_str = prices
            try:
                price = float(price_str) * 100
            except:
                price = 50.0

            if outcome_name == 'YES':
                yes_price = price
            elif outcome_name == 'NO':
                no_price = price

        signal = analysis.get('signal', 'YES')
        confidence = analysis.get('confidence', 50)
        reasoning = analysis.get('reasoning', '')
        key_factors = analysis.get('key_factors', [])

        # Calculate prices
        if signal == "YES":
            ai_price = confidence
            market_price = yes_price
        else:
            ai_price = 100 - confidence
            market_price = no_price

        edge = abs(ai_price - market_price) / market_price * 100 if market_price > 0 else 0

        # Calculate bet size based on confidence
        # 70% -> $1, 80% -> $2, 90% -> $3.50, 100% -> $5
        if confidence >= 70:
            confidence_normalized = (confidence - 70) / 30
            bet_size = 5.0 * (0.2 + 0.8 * confidence_normalized)
            bet_size = round(bet_size * 2) / 2  # Round to $0.50
        else:
            bet_size = 0

        # Display
        print("\n" + "="*70)
        print("POLYDICTIONS AUTO-TRADER")
        print("="*70)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print(f"MARKET: {title}")
        print(f"URL: https://polymarket.com/event/{slug}")
        print("="*70)
        print()
        print(f"CURRENT PRICES:")
        print(f"  YES: {yes_price:.1f}%")
        print(f"  NO:  {no_price:.1f}%")
        print()
        print("="*70)
        print("AI ANALYSIS")
        print("="*70)
        print()
        print(f"SIGNAL:     {signal}")
        print(f"CONFIDENCE: {confidence}%")
        print()
        print("REASONING:")
        print("-" * 70)
        print(reasoning)
        print("-" * 70)
        print()

        if key_factors:
            print("KEY FACTORS:")
            for i, factor in enumerate(key_factors, 1):
                print(f"  {i}. {factor}")
            print()

        print("="*70)
        print("TRADE ANALYSIS")
        print("="*70)
        print()
        print(f"AI Price:     {ai_price:.1f}%")
        print(f"Market Price: {market_price:.1f}%")
        print(f"Edge:         {edge:.1f}%")
        print()

        if bet_size > 0 and edge >= 10:
            print(f"[TRADE SIGNAL]")
            print(f"  Side:   {signal}")
            print(f"  Amount: ${bet_size:.2f}")
            print(f"  Edge:   {edge:.1f}%")
            print()
            print("[OK] Placing order...")
            print(f"[OK] Order filled: order_abc123def456")
            print(f"[OK] Position opened: ${bet_size:.2f} on {signal}")
        else:
            print(f"[SKIP] No trade")
            if confidence < 70:
                print(f"  Reason: Confidence too low ({confidence}% < 70%)")
            elif edge < 10:
                print(f"  Reason: Edge too small ({edge:.1f}% < 10%)")
            elif bet_size < 1:
                print(f"  Reason: Bet size too small (${bet_size:.2f})")

        print()
        print("="*70)
        print()

    async def run(self, slug=None):
        """Run demo trade analysis"""

        print("\n" + "="*70)
        print("DEMO TRADER - Analysis for Screenshot")
        print("="*70)
        print()

        # Get market
        print("[*] Fetching market...")
        market = self.get_market(slug=slug)

        if not market:
            print("[X] Could not fetch market")
            return

        title = market.get('title', 'Unknown')
        print(f"[OK] Selected: {title[:60]}")

        # Get prices
        outcomes = market.get('markets', [])
        yes_price = 50.0
        no_price = 50.0

        for outcome in outcomes:
            outcome_name = outcome.get('outcome', '').upper()
            prices = outcome.get('outcomePrices', '0.5')
            # Handle both string and list formats
            if isinstance(prices, list):
                price_str = prices[0] if prices else '0.5'
            else:
                price_str = prices
            try:
                price = float(price_str) * 100
            except:
                price = 50.0

            if outcome_name == 'YES':
                yes_price = price
            elif outcome_name == 'NO':
                no_price = price

        # Get Polyfactual analysis
        print("[*] Getting deep research from Polyfactual API...")
        polyfactual_reasoning = await self.get_polyfactual_analysis(title)

        if polyfactual_reasoning:
            print("[OK] Got research analysis")
            # Analyze with Claude
            print("[*] Analyzing with Claude Sonnet 3.5...")
            analysis = await self.analyze_with_claude(title, polyfactual_reasoning, yes_price, no_price)
        else:
            print("[!] Polyfactual unavailable, using demo mode...")
            analysis = None

        if not analysis:
            # Fallback to fake analysis for demo
            print("[*] Generating demo analysis...")
            analysis = self.generate_fake_analysis(title, yes_price, no_price)

        print("[OK] Analysis complete")

        # Display
        self.display_trade(market, analysis)


async def main():
    """Main entry point"""

    print("\n" + "="*70)
    print("POLYDICTIONS DEMO TRADER")
    print("="*70)
    print("\nGenerates beautiful trade analysis for screenshots")
    print("(No real trades - just for demo)")
    print()

    # Check keys
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[X] ANTHROPIC_API_KEY not set")
        return

    if not os.getenv("POLYFACTUAL_API_KEY"):
        print("[X] POLYFACTUAL_API_KEY not set")
        return

    # Ask for slug or random
    use_slug = input("Enter market slug (or press Enter for random): ").strip()

    trader = DemoTrader()

    if use_slug:
        await trader.run(slug=use_slug)
    else:
        await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
