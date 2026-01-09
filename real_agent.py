"""
polydictions ai trade agent
polymarket -> factsai research -> claude opus 4.5 decision
"""

import os
import sys
import io
import asyncio
import random
from datetime import datetime
from dotenv import load_dotenv
import aiohttp
import requests
import json

# Fix Windows encoding for Unicode
os.system('chcp 65001 >nul 2>&1')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

load_dotenv()


class AITradeAgent:
    """polydictions ai trade agent"""

    def __init__(self):
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.factsai_key = os.getenv("FACTSAI_API_KEY")
        self.factsai_url = "https://factsai.org/answer"
        self.polymarket_key = os.getenv("POLYMARKET_BUILDERS_KEY")

        print("\n" + "="*60)
        print("polydictions ai trade agent")
        print("="*60)
        print(f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        if not self.anthropic_key:
            print("[x] anthropic api key missing")
            raise ValueError("ANTHROPIC_API_KEY required")
        print("[ok] anthropic key")

        if not self.factsai_key:
            print("[x] factsai api key missing")
            raise ValueError("FACTSAI_API_KEY required")
        print("[ok] factsai key")

        if self.polymarket_key:
            print("[ok] polymarket builders key")
        print()

    def step1_get_active_market(self):
        """fetch active market from polymarket"""
        print("="*60)
        print("step 1: fetching market from polymarket")
        print("="*60)

        try:
            headers = {}
            if self.polymarket_key:
                headers["Authorization"] = f"Bearer {self.polymarket_key}"

            resp = requests.get(
                "https://gamma-api.polymarket.com/events",
                headers=headers,
                params={
                    "limit": 100,
                    "active": True,
                    "closed": False
                },
                timeout=15
            )

            if resp.status_code != 200:
                print(f"[x] api error: {resp.status_code}")
                return None

            events = resp.json()
            print(f"[ok] got {len(events)} events")

            active = [e for e in events if e.get('markets')]

            # filter out markets with prices > 90% (no edge)
            tradeable = []
            for e in active:
                markets_data = e.get('markets', [])
                if not markets_data:
                    continue

                # get yes price from outcomePrices [yes, no]
                m = markets_data[0]
                outcome_prices = m.get('outcomePrices')
                if isinstance(outcome_prices, str):
                    try:
                        outcome_prices = json.loads(outcome_prices)
                    except:
                        outcome_prices = None

                if outcome_prices and len(outcome_prices) >= 2:
                    yes_pct = float(outcome_prices[0]) * 100
                else:
                    yes_pct = 50.0

                # skip if yes > 90 or yes < 10 (no edge)
                if yes_pct > 90 or yes_pct < 10:
                    continue
                tradeable.append(e)

            print(f"[ok] {len(tradeable)} tradeable (10-90% range)")

            if not tradeable:
                print("[x] no tradeable markets")
                return None

            market = random.choice(tradeable)

            title = market.get('title', 'Unknown')
            slug = market.get('slug', '')

            print(f"\n[selected] {title}")
            print(f"polymarket.com/event/{slug}")

            # Get prices from outcomePrices [yes, no]
            markets_data = market.get('markets', [])
            m = markets_data[0] if markets_data else {}
            outcome_prices = m.get('outcomePrices')

            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json.loads(outcome_prices)
                except:
                    outcome_prices = None

            if outcome_prices and len(outcome_prices) >= 2:
                yes_price = float(outcome_prices[0]) * 100
                no_price = float(outcome_prices[1]) * 100
            else:
                yes_price = 50.0
                no_price = 50.0

            print(f"prices: yes {yes_price:.1f}% / no {no_price:.1f}%")

            return {
                'title': title,
                'slug': slug,
                'yes_price': yes_price or 50.0,
                'no_price': no_price or 50.0,
                'raw': market
            }

        except Exception as e:
            print(f"[x] error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def step2_factsai_research(self, market_title):
        """deep research via factsai"""
        print()
        print("="*60)
        print("step 2: deep research via factsai")
        print("="*60)

        query = f"What is the likelihood of: {market_title}? Analyze recent news, data, and provide a probability estimate."

        print(f"query: {query[:70]}...")
        print()
        print("[*] calling factsai api...")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.factsai_url,
                    headers={
                        "Authorization": f"Bearer {self.factsai_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "query": query,
                        "text": True
                    },
                    timeout=aiohttp.ClientTimeout(total=90)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()

                        if data.get('success'):
                            answer = data.get('data', {}).get('answer', '')
                            citations = data.get('data', {}).get('citations', [])

                            if answer:
                                print(f"[ok] got research ({len(answer)} chars, {len(citations)} sources)")
                                print()
                                print("research:")
                                print("-" * 60)
                                preview = answer[:400] + "..." if len(answer) > 400 else answer
                                print(preview)
                                print("-" * 60)
                                return answer
                            else:
                                print("[x] empty answer from factsai")
                                return None
                        else:
                            error = data.get('error', 'Unknown error')
                            print(f"[x] factsai error: {error}")
                            return None
                    else:
                        error = await resp.text()
                        print(f"[x] factsai error {resp.status}")
                        return None

        except asyncio.TimeoutError:
            print("[x] factsai timeout")
            return None
        except Exception as e:
            print(f"[x] factsai error: {e}")
            return None

    async def step3_claude_decision(self, market_title, research, yes_price, no_price):
        """claude opus 4.5 trading decision"""
        print()
        print("="*60)
        print("step 3: claude opus 4.5 decision")
        print("="*60)

        prompt = f"""You are a prediction market trader. Analyze this market and make a trading decision.

MARKET: {market_title}
CURRENT PRICES: YES {yes_price:.1f}% | NO {no_price:.1f}%

RESEARCH:
{research}

Based on the research above, provide your trading decision in this exact JSON format:
{{
    "signal": "YES" or "NO",
    "confidence": <number 0-100>,
    "reasoning": "<2-3 sentences explaining your decision>"
}}

Be decisive. If the research suggests YES is more likely, bet YES. If NO is more likely, bet NO.
Only output the JSON, nothing else."""

        print("[*] calling claude opus 4.5...")

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
                        "model": "claude-opus-4-20250514",
                        "max_tokens": 500,
                        "messages": [{"role": "user", "content": prompt}]
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        text = data.get("content", [{}])[0].get("text", "")

                        print(f"[ok] got claude response")
                        print()
                        print("claude decision:")
                        print("-" * 60)
                        print(text)
                        print("-" * 60)

                        try:
                            clean = text.strip()
                            if "```json" in clean:
                                clean = clean.split("```json")[1].split("```")[0].strip()
                            elif "```" in clean:
                                clean = clean.split("```")[1].split("```")[0].strip()

                            parsed = json.loads(clean)
                            return parsed
                        except json.JSONDecodeError as e:
                            print(f"[x] json parse error: {e}")
                            return None
                    else:
                        error = await resp.text()
                        print(f"[x] claude error {resp.status}")
                        return None

        except Exception as e:
            print(f"[x] error: {e}")
            return None

    def step4_display_result(self, market, decision):
        """display final result"""
        print()
        print("="*60)
        print("trade decision")
        print("="*60)

        signal = decision.get('signal', 'UNKNOWN')
        confidence = decision.get('confidence', 0)
        reasoning = decision.get('reasoning', '')

        yes_price = market['yes_price']
        no_price = market['no_price']

        if signal == "YES":
            ai_price = confidence
            market_price = yes_price
        else:
            ai_price = 100 - confidence
            market_price = no_price

        edge = abs(ai_price - market_price) / market_price * 100 if market_price > 0 else 0

        print()
        print(f"{market['title']}")
        print(f"polymarket.com/event/{market['slug']}")
        print()
        print(f"market: yes {yes_price:.1f}% / no {no_price:.1f}%")
        print(f"signal: {signal.lower()}")
        print(f"confidence: {confidence}%")
        print(f"edge: {edge:.1f}%")
        print()
        print(f"reasoning: {reasoning}")
        print()

        if confidence >= 70 and edge >= 10:
            bet_normalized = (confidence - 70) / 30
            bet_size = 5.0 * (0.2 + 0.8 * bet_normalized)
            bet_size = round(bet_size * 2) / 2

            print(f"[trade]")
            print(f"side: {signal.lower()}")
            print(f"size: ${bet_size:.2f}")
        else:
            print(f"[no trade]")
            if confidence < 70:
                print(f"confidence too low ({confidence}% < 70%)")
            if edge < 10:
                print(f"edge too small ({edge:.1f}% < 10%)")

        print()
        print("="*60)

    async def run(self):
        """run full pipeline"""

        market = self.step1_get_active_market()
        if not market:
            print("\n[failed] could not get market")
            return False

        research = await self.step2_factsai_research(market['title'])
        if not research:
            print("\n[failed] factsai unavailable")
            return False

        decision = await self.step3_claude_decision(
            market['title'],
            research,
            market['yes_price'],
            market['no_price']
        )
        if not decision:
            print("\n[failed] claude error")
            return False

        self.step4_display_result(market, decision)
        return True


async def main():
    try:
        agent = AITradeAgent()
        success = await agent.run()

        if success:
            print("\n[ok] done")
        else:
            print("\n[x] failed")

    except Exception as e:
        print(f"\n[x] fatal: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
