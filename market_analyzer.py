"""
Market Analyzer - Random Market Analysis Display
Fetches random Polymarket markets and displays detailed AI reasoning
"""

import os
import asyncio
import random
from datetime import datetime
from dotenv import load_dotenv
import aiohttp
import requests

load_dotenv()


class MarketAnalyzer:
    """Analyzes random markets and displays reasoning in terminal"""

    def __init__(self):
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.polyfactual_key = os.getenv("POLYFACTUAL_API_KEY")
        self.polyfactual_url = os.getenv("POLYFACTUAL_API_URL",
                                         "https://deep-research-api.thekid-solana.workers.dev/answer")

        if not self.anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")
        if not self.polyfactual_key:
            raise ValueError("POLYFACTUAL_API_KEY not set in .env")

    def get_random_markets(self, limit=5):
        """Fetch random active markets from Polymarket"""
        print("\n[*] Fetching active markets from Polymarket...")

        try:
            # Get active markets - don't use active/closed params, just get recent ones
            resp = requests.get(
                "https://gamma-api.polymarket.com/events",
                params={
                    "limit": 100
                },
                timeout=10
            )

            if resp.status_code != 200:
                print(f"[X] Failed to fetch markets: {resp.status_code}")
                return []

            markets = resp.json()

            print(f"[DEBUG] Got {len(markets)} total events from API")

            # Filter for binary markets with YES/NO outcomes
            binary_markets = []
            for market in markets:
                # Skip closed markets
                if market.get('closed', False):
                    continue

                # Check if it has 2 outcomes (binary market)
                outcomes = market.get('markets', [])

                if len(outcomes) == 2:
                    outcome_names = [o.get('outcome', '').upper() for o in outcomes]
                    # Check if it's YES/NO market
                    if 'YES' in outcome_names and 'NO' in outcome_names:
                        binary_markets.append(market)

            # Select random markets
            if len(binary_markets) > limit:
                selected = random.sample(binary_markets, limit)
            else:
                selected = binary_markets

            print(f"[OK] Found {len(binary_markets)} binary markets, selected {len(selected)}")

            return selected

        except Exception as e:
            print(f"[X] Error fetching markets: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def get_polyfactual_analysis(self, question):
        """Get deep research analysis from Polyfactual API"""
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
                        error_text = await resp.text()
                        print(f"[X] Polyfactual API error {resp.status}: {error_text[:200]}")
                        return None
        except Exception as e:
            print(f"[X] Error calling Polyfactual: {e}")
            return None

    async def format_reasoning_with_claude(self, market_title, polyfactual_reasoning, yes_price, no_price):
        """Use Claude Sonnet to format and enhance the reasoning"""

        prompt = f"""You are analyzing a prediction market. Format a clear, concise reasoning for this market.

Market: {market_title}
Current Prices: YES {yes_price:.1f}% | NO {no_price:.1f}%

Research Data:
{polyfactual_reasoning}

Provide:
1. Your signal (YES or NO)
2. Confidence level (0-100)
3. Clear reasoning (2-4 sentences explaining WHY)
4. Key factors supporting your position

Format as JSON:
{{
    "signal": "YES or NO",
    "confidence": 85,
    "reasoning": "Clear explanation here...",
    "key_factors": ["Factor 1", "Factor 2", "Factor 3"]
}}"""

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
                        # Extract JSON from response
                        if "```json" in text:
                            text = text.split("```json")[1].split("```")[0].strip()
                        elif "```" in text:
                            text = text.split("```")[1].split("```")[0].strip()

                        parsed = json.loads(text)
                        return parsed
                    else:
                        error_text = await resp.text()
                        print(f"[X] Claude API error {resp.status}: {error_text[:200]}")
                        return None
        except Exception as e:
            print(f"[X] Error calling Claude: {e}")
            return None

    def display_analysis(self, market, analysis):
        """Display formatted analysis in terminal"""

        title = market.get('title', 'Unknown Market')
        slug = market.get('slug', '')

        # Get prices
        outcomes = market.get('markets', [])
        yes_price = 0
        no_price = 0
        for outcome in outcomes:
            if outcome.get('outcome', '').upper() == 'YES':
                yes_price = float(outcome.get('outcomePrices', [0.5])[0]) * 100
            elif outcome.get('outcome', '').upper() == 'NO':
                no_price = float(outcome.get('outcomePrices', [0.5])[0]) * 100

        signal = analysis.get('signal', 'UNKNOWN')
        confidence = analysis.get('confidence', 0)
        reasoning = analysis.get('reasoning', 'No reasoning provided')
        key_factors = analysis.get('key_factors', [])

        # Calculate edge
        if signal == "YES":
            ai_price = confidence
            market_price = yes_price
        else:
            ai_price = 100 - confidence
            market_price = no_price

        edge = abs(ai_price - market_price) / market_price * 100 if market_price > 0 else 0

        # Display
        print("\n" + "="*70)
        print(f"MARKET: {title}")
        print("="*70)
        print(f"URL: https://polymarket.com/event/{slug}")
        print(f"Current Prices: YES {yes_price:.1f}% | NO {no_price:.1f}%")
        print()
        print(f"SIGNAL: {signal}")
        print(f"CONFIDENCE: {confidence}%")
        print(f"AI PRICE: {ai_price:.1f}%")
        print(f"MARKET PRICE: {market_price:.1f}%")
        print(f"EDGE: {edge:.1f}%")
        print()
        print("REASONING:")
        print("-" * 70)
        print(reasoning)
        print("-" * 70)

        if key_factors:
            print()
            print("KEY FACTORS:")
            for i, factor in enumerate(key_factors, 1):
                print(f"  {i}. {factor}")

        print("="*70)

    async def analyze_market(self, market):
        """Analyze a single market"""

        title = market.get('title', 'Unknown Market')

        print(f"\n[*] Analyzing: {title[:60]}...")

        # Get prices
        outcomes = market.get('markets', [])
        yes_price = 0
        no_price = 0
        for outcome in outcomes:
            if outcome.get('outcome', '').upper() == 'YES':
                yes_price = float(outcome.get('outcomePrices', [0.5])[0]) * 100
            elif outcome.get('outcome', '').upper() == 'NO':
                no_price = float(outcome.get('outcomePrices', [0.5])[0]) * 100

        # Get Polyfactual analysis
        print("[*] Getting deep research from Polyfactual...")
        polyfactual_reasoning = await self.get_polyfactual_analysis(title)

        if not polyfactual_reasoning:
            print("[X] Failed to get Polyfactual analysis, skipping...")
            return None

        print("[OK] Got Polyfactual research")

        # Format with Claude
        print("[*] Formatting with Claude Sonnet...")
        analysis = await self.format_reasoning_with_claude(
            title, polyfactual_reasoning, yes_price, no_price
        )

        if not analysis:
            print("[X] Failed to format with Claude, skipping...")
            return None

        print("[OK] Analysis complete")

        # Display
        self.display_analysis(market, analysis)

        return analysis

    async def run(self, num_markets=5):
        """Main run loop"""

        print("\n" + "="*70)
        print("MARKET ANALYZER - Random Market Analysis")
        print("="*70)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Target: Analyze {num_markets} random markets")
        print("="*70)

        # Get random markets
        markets = self.get_random_markets(limit=num_markets)

        if not markets:
            print("\n[X] No markets found, exiting...")
            return

        # Analyze each market
        analyzed = 0
        for i, market in enumerate(markets, 1):
            print(f"\n{'='*70}")
            print(f"Market {i}/{len(markets)}")
            print(f"{'='*70}")

            result = await self.analyze_market(market)

            if result:
                analyzed += 1

            # Small delay between markets
            if i < len(markets):
                print("\n[*] Waiting 3 seconds before next market...")
                await asyncio.sleep(3)

        # Summary
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"Successfully analyzed: {analyzed}/{len(markets)} markets")
        print("="*70 + "\n")


async def run_continuous():
    """Run continuously - analyze new random markets every 5 minutes"""

    analyzer = MarketAnalyzer()

    print("\n" + "="*70)
    print("CONTINUOUS MODE - Analyzing random markets every 5 minutes")
    print("Press Ctrl+C to stop")
    print("="*70)

    iteration = 1

    while True:
        try:
            print(f"\n{'='*70}")
            print(f"Iteration #{iteration} - {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*70}")

            await analyzer.run(num_markets=3)

            iteration += 1

            print(f"\n{'='*70}")
            print(f"Next analysis in 5 minutes... (Ctrl+C to stop)")
            print(f"{'='*70}")

            await asyncio.sleep(300)  # 5 minutes

        except KeyboardInterrupt:
            print("\n\nStopped by user")
            break
        except Exception as e:
            print(f"\n[X] Error: {e}")
            print("[*] Continuing...")
            await asyncio.sleep(60)


async def main():
    """Main entry point"""

    print("\n" + "="*70)
    print("MARKET ANALYZER")
    print("="*70)
    print("\nRandomly selects Polymarket markets and displays AI reasoning")
    print()

    # Check API keys
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[X] Error: ANTHROPIC_API_KEY not set in .env")
        return

    if not os.getenv("POLYFACTUAL_API_KEY"):
        print("[X] Error: POLYFACTUAL_API_KEY not set in .env")
        return

    mode = input("Run mode? (once/continuous): ").strip().lower()

    if mode == "continuous":
        await run_continuous()
    else:
        num = input("How many markets to analyze? (default: 5): ").strip()
        num_markets = int(num) if num.isdigit() else 5

        analyzer = MarketAnalyzer()
        await analyzer.run(num_markets=num_markets)


if __name__ == "__main__":
    asyncio.run(main())
