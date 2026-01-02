"""
Dome API Integration for Polydictions
- Whale Tracker
- Cross-platform Arbitrage (Polymarket vs Kalshi)
- Order Flow (Large orders alerts)
"""
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict

# Dome API Config (Dev Tier - 100 QPS)
DOME_API_KEY = '0552f4458d67462cc3c87c40b9edb739a267bfaa'
DOME_BASE_URL = 'https://api.domeapi.io/v1'

# Polymarket Builders API Key
POLYMARKET_BUILDERS_KEY = '019b3895-f661-7002-9826-6d8615dd63cb'

# Thresholds
LARGE_ORDER_THRESHOLD = 1000  # $1000+ orders
WHALE_PNL_THRESHOLD = 10000   # $10K+ total PnL = whale
ARBITRAGE_THRESHOLD = 3.0     # 3%+ difference between platforms


class DomeClient:
    """Async client for Dome API"""

    def __init__(self, api_key: str = DOME_API_KEY):
        self.api_key = api_key
        self.headers = {'Authorization': f'Bearer {api_key}'}

    async def _request(self, endpoint: str, params: dict = None) -> dict:
        """Make async request to Dome API"""
        url = f'{DOME_BASE_URL}{endpoint}'
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers, params=params, timeout=30) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    error = await resp.text()
                    raise Exception(f'Dome API error {resp.status}: {error}')

    # ============ WHALE TRACKER ============

    async def get_wallet_info(self, address: str) -> dict:
        """Get wallet info (EOA or proxy)"""
        return await self._request('/polymarket/wallet', {'eoa': address})

    async def get_wallet_pnl(self, address: str, granularity: str = 'day') -> dict:
        """Get wallet PnL history"""
        return await self._request(f'/polymarket/wallet/pnl/{address}', {'granularity': granularity})

    async def get_recent_orders(self, limit: int = 100, wallet: str = None) -> List[dict]:
        """Get recent orders, optionally filtered by wallet"""
        params = {'limit': limit}
        if wallet:
            params['maker'] = wallet
        data = await self._request('/polymarket/orders', params)
        return data.get('orders', [])

    async def get_leaderboard(self, period: str = 'MONTH', limit: int = 30) -> List[dict]:
        """
        Get official Polymarket leaderboard
        period: DAY, WEEK, MONTH, ALL
        """
        url = f'https://data-api.polymarket.com/v1/leaderboard'
        params = {
            'timePeriod': period,
            'orderBy': 'PNL',
            'limit': limit
        }
        headers = {
            'Authorization': f'Bearer {POLYMARKET_BUILDERS_KEY}'
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers, timeout=30) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [
                        {
                            'rank': int(t.get('rank', 0)),
                            'address': t.get('proxyWallet', ''),
                            'username': t.get('userName', ''),
                            'pnl': t.get('pnl', 0),
                            'volume': t.get('vol', 0),
                            'verified': t.get('verifiedBadge', False)
                        }
                        for t in data
                    ]
                return []

    async def find_whales_from_orders(self, min_pnl: float = WHALE_PNL_THRESHOLD) -> List[dict]:
        """Find whales by scanning recent orders and checking their PnL"""
        orders = await self.get_recent_orders(limit=1000)

        # Get unique wallets
        wallets = list(set(o['user'] for o in orders))

        whales = []
        # Check up to 150 wallets (Dev tier: 100 QPS, this takes ~3-5 seconds)
        for wallet in wallets[:150]:
            try:
                pnl_data = await self.get_wallet_pnl(wallet)
                pnl_history = pnl_data.get('pnl_over_time', [])
                if pnl_history:
                    current_pnl = pnl_history[-1].get('pnl_to_date', 0)
                    if current_pnl >= min_pnl:
                        whales.append({
                            'address': wallet,
                            'pnl': current_pnl,
                            'pnl_history': pnl_history[-30:]  # Last 30 days
                        })
            except:
                continue
            await asyncio.sleep(0.02)  # Rate limit (100 QPS)

        return sorted(whales, key=lambda x: x['pnl'], reverse=True)

    # ============ ORDER FLOW ============

    async def get_large_orders(self, min_value: float = LARGE_ORDER_THRESHOLD, limit: int = 100) -> List[dict]:
        """Get large orders above threshold"""
        orders = await self.get_recent_orders(limit=limit)

        large_orders = []
        for o in orders:
            shares = o.get('shares_normalized', 0)
            price = o.get('price', 0)
            value = shares * price

            if value >= min_value:
                large_orders.append({
                    'side': o['side'],
                    'shares': shares,
                    'price': price,
                    'value': value,
                    'title': o.get('title', ''),
                    'market_slug': o.get('market_slug', ''),
                    'token_label': o.get('token_label', ''),
                    'user': o['user'],
                    'timestamp': o.get('timestamp', 0),
                    'tx_hash': o.get('tx_hash', '')
                })

        return sorted(large_orders, key=lambda x: x['value'], reverse=True)

    async def get_order_flow_sentiment(self, market_slug: str = None, limit: int = 100) -> dict:
        """Analyze order flow sentiment (buy vs sell pressure)"""
        params = {'limit': limit}
        if market_slug:
            params['market_slug'] = market_slug

        data = await self._request('/polymarket/orders', params)
        orders = data.get('orders', [])

        buy_volume = 0
        sell_volume = 0
        buy_count = 0
        sell_count = 0

        for o in orders:
            value = o.get('shares_normalized', 0) * o.get('price', 0)
            if o['side'] == 'BUY':
                buy_volume += value
                buy_count += 1
            else:
                sell_volume += value
                sell_count += 1

        total = buy_volume + sell_volume
        sentiment = 'neutral'
        if total > 0:
            buy_pct = (buy_volume / total) * 100
            if buy_pct > 60:
                sentiment = 'bullish'
            elif buy_pct < 40:
                sentiment = 'bearish'

        return {
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'buy_percentage': (buy_volume / total * 100) if total > 0 else 50,
            'sentiment': sentiment
        }

    # ============ CROSS-PLATFORM ARBITRAGE (SPORTS) ============

    # Supported sports for Dome matching API
    SPORTS = ['nfl', 'nba', 'nhl', 'mlb', 'cfb', 'cbb']

    async def get_sports_matches(self, sport: str, date: str) -> Dict:
        """
        Get matched markets for a sport on a specific date.
        Uses Dome's /matching-markets/sports/{sport} endpoint.

        Args:
            sport: nfl, nba, nhl, mlb, cfb, cbb
            date: YYYY-MM-DD format

        Returns:
            Dict with matched markets containing both Polymarket and Kalshi data
        """
        try:
            data = await self._request(f'/matching-markets/sports/{sport}', {'date': date})
            return data.get('markets', {})
        except Exception as e:
            print(f"Dome matching API error: {e}")
            return {}

    async def get_kalshi_market_price(self, market_ticker: str) -> Optional[dict]:
        """Get Kalshi market details including price - DIRECT from Kalshi API"""
        try:
            url = f'https://api.elections.kalshi.com/trade-api/v2/markets/{market_ticker}'
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        m = data.get('market', {})
                        # Use yes_ask (price to buy YES) - matches what UI shows
                        yes_ask = m.get('yes_ask', 0)
                        if not yes_ask:
                            yes_ask = m.get('last_price', 0)
                        return {
                            'title': m.get('title', ''),
                            'price': yes_ask,  # in cents
                            'volume': m.get('volume', 0),
                            'ticker': market_ticker
                        }
        except Exception as e:
            print(f"Kalshi API error for {market_ticker}: {e}")
        return None

    async def get_polymarket_price(self, token_id: str, slug: str = None) -> Optional[float]:
        """Get current price for a Polymarket token.

        First tries Gamma API (more reliable), falls back to CLOB API.
        """
        # Try Gamma API first if we have the slug
        if slug:
            try:
                url = f'https://gamma-api.polymarket.com/events?slug={slug}'
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=10) as resp:
                        if resp.status == 200:
                            events = await resp.json()
                            for event in events:
                                for m in event.get('markets', []):
                                    clob_tokens = m.get('clobTokenIds', '[]')
                                    if isinstance(clob_tokens, str):
                                        clob_tokens = json.loads(clob_tokens)

                                    if token_id in clob_tokens:
                                        idx = clob_tokens.index(token_id)
                                        prices_raw = m.get('outcomePrices', '[]')
                                        if isinstance(prices_raw, str):
                                            prices = json.loads(prices_raw)
                                        else:
                                            prices = prices_raw
                                        if idx < len(prices):
                                            return float(prices[idx])
            except Exception as e:
                print(f"Polymarket Gamma error for {token_id}: {e}")

        # Fallback to CLOB API
        try:
            url = f'https://clob.polymarket.com/price?token_id={token_id}&side=buy'
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return float(data.get('price', 0))
        except Exception as e:
            print(f"Polymarket CLOB error for {token_id}: {e}")
        return None

    async def find_sports_arbitrage(self, sports: List[str] = None, days_ahead: int = 7, min_diff: float = ARBITRAGE_THRESHOLD) -> List[dict]:
        """
        Find arbitrage opportunities in sports markets using Dome matching API.

        Args:
            sports: List of sports to check (default: all)
            days_ahead: How many days ahead to scan
            min_diff: Minimum spread percentage to report

        Returns:
            List of arbitrage opportunities with prices from both platforms
        """
        if sports is None:
            sports = self.SPORTS

        opportunities = []
        today = datetime.now()

        for sport in sports:
            for day_offset in range(days_ahead):
                date = (today + timedelta(days=day_offset)).strftime('%Y-%m-%d')

                try:
                    matches = await self.get_sports_matches(sport, date)

                    for game_key, platforms in matches.items():
                        # Get platform data
                        kalshi_data = None
                        poly_data = None

                        for p in platforms:
                            if p.get('platform') == 'KALSHI':
                                kalshi_data = p
                            elif p.get('platform') == 'POLYMARKET':
                                poly_data = p

                        # Need both platforms for arbitrage
                        if not kalshi_data or not poly_data:
                            continue

                        # Get prices for each outcome
                        kalshi_tickers = kalshi_data.get('market_tickers', [])
                        poly_tokens = poly_data.get('token_ids', [])
                        poly_slug = poly_data.get('market_slug', '')

                        if len(kalshi_tickers) < 2 or len(poly_tokens) < 2:
                            continue

                        # Get prices for BOTH teams on BOTH platforms
                        kalshi_prices = []
                        kalshi_titles = []
                        for ticker in kalshi_tickers[:2]:
                            market = await self.get_kalshi_market_price(ticker)
                            if market:
                                kalshi_prices.append(market['price'] / 100)
                                kalshi_titles.append(market['title'])
                            else:
                                kalshi_prices.append(None)
                                kalshi_titles.append('')

                        poly_prices = []
                        for token in poly_tokens[:2]:
                            price = await self.get_polymarket_price(token, slug=poly_slug)
                            poly_prices.append(price)

                        # Skip if missing prices
                        if None in kalshi_prices or None in poly_prices:
                            continue

                        # Skip if any price is 0 (likely invalid/stale data)
                        if 0 in kalshi_prices or 0 in poly_prices:
                            continue

                        # Sanity check: prices should roughly sum to 100% on each platform
                        kalshi_sum = sum(kalshi_prices)
                        poly_sum = sum(poly_prices)
                        if kalshi_sum < 0.9 or kalshi_sum > 1.1 or poly_sum < 0.9 or poly_sum > 1.1:
                            continue

                        # Max spread sanity check - spreads > 18% are likely bad data
                        MAX_REALISTIC_SPREAD = 0.18

                        # Determine correct pairing by checking which combo sums closer to 100%
                        # Option A: kalshi[0] matches poly[0]
                        # Option B: kalshi[0] matches poly[1]

                        # For option A: same team prices should be similar
                        diff_a = abs(kalshi_prices[0] - poly_prices[0]) + abs(kalshi_prices[1] - poly_prices[1])
                        # For option B: kalshi[0] matches poly[1]
                        diff_b = abs(kalshi_prices[0] - poly_prices[1]) + abs(kalshi_prices[1] - poly_prices[0])

                        # Use the pairing with smaller total difference (more likely correct)
                        if diff_a <= diff_b:
                            # Same order: kalshi[0]=poly[0], kalshi[1]=poly[1]
                            pairs = [(0, 0), (1, 1)]
                        else:
                            # Swapped: kalshi[0]=poly[1], kalshi[1]=poly[0]
                            pairs = [(0, 1), (1, 0)]

                        # Check each team for arbitrage
                        for kalshi_idx, poly_idx in pairs:
                            k_price = kalshi_prices[kalshi_idx]
                            p_price = poly_prices[poly_idx]
                            k_title = kalshi_titles[kalshi_idx]
                            k_ticker = kalshi_tickers[kalshi_idx]
                            p_token = poly_tokens[poly_idx]

                            # Skip extreme prices
                            if k_price <= 0.03 or k_price >= 0.97:
                                continue
                            if p_price <= 0.03 or p_price >= 0.97:
                                continue

                            # Calculate spread (can be negative if poly > kalshi)
                            diff = (k_price - p_price) * 100  # positive = kalshi higher (profitable)

                            # Skip unrealistic spreads (likely bad data)
                            if abs(k_price - p_price) > MAX_REALISTIC_SPREAD:
                                continue

                            # Only include profitable spreads from +1% to +18%
                            if diff >= 1.0 and diff <= 18.0:
                                # Determine direction
                                if p_price < k_price:
                                    direction = 'buy_poly_sell_kalshi'
                                    action = f"Buy YES on Polymarket ({p_price*100:.1f}%), Sell YES on Kalshi ({k_price*100:.1f}%)"
                                    is_profitable = True
                                else:
                                    direction = 'buy_kalshi_sell_poly'
                                    action = f"Buy YES on Kalshi ({k_price*100:.1f}%), Sell YES on Polymarket ({p_price*100:.1f}%)"
                                    is_profitable = False

                                # Extract team name from ticker (e.g., KXNFLGAME-25DEC25DENKC-DEN -> DEN)
                                team_code = k_ticker.split('-')[-1] if k_ticker else ''

                                opportunities.append({
                                    'sport': sport.upper(),
                                    'game': game_key,
                                    'team': team_code,
                                    'date': date,
                                    'polymarket_slug': poly_slug,
                                    'polymarket_price': p_price,
                                    'polymarket_token': p_token,
                                    'kalshi_title': k_title,
                                    'kalshi_ticker': k_ticker,
                                    'kalshi_price': k_price,
                                    'difference': diff,
                                    'direction': direction,
                                    'action': action,
                                    'confidence': 1.0,
                                    'is_profitable': is_profitable
                                })

                        # Small delay to respect rate limits
                        await asyncio.sleep(0.05)

                except Exception as e:
                    print(f"Error scanning {sport} {date}: {e}")
                    continue

        # Sort by spread (highest first)
        opportunities = sorted(opportunities, key=lambda x: x['difference'], reverse=True)

        return opportunities  # Return all opportunities

    async def find_arbitrage_opportunities(self, min_diff: float = ARBITRAGE_THRESHOLD) -> List[dict]:
        """
        Main arbitrage function - now uses sports matching.
        For backwards compatibility with existing bot commands.
        """
        return await self.find_sports_arbitrage(min_diff=min_diff)

    # ============ GENERAL MARKETS ARBITRAGE (FUZZY MATCHING) ============

    def _normalize_title(self, title: str) -> str:
        """Normalize title for matching"""
        import re
        title = title.lower()
        # Remove common words and punctuation
        title = re.sub(r'[^a-z0-9\s]', '', title)
        title = re.sub(r'\b(will|the|a|an|in|on|at|by|to|of|for|be|is|are|was|were)\b', '', title)
        title = ' '.join(title.split())  # Normalize whitespace
        return title

    def _calculate_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles (0-1)"""
        t1 = set(self._normalize_title(title1).split())
        t2 = set(self._normalize_title(title2).split())
        if not t1 or not t2:
            return 0.0
        intersection = len(t1 & t2)
        union = len(t1 | t2)
        return intersection / union if union > 0 else 0.0

    async def get_kalshi_general_markets(self, limit: int = 100) -> List[dict]:
        """Get general (non-sports) Kalshi markets from Dome API"""
        try:
            data = await self._request('/kalshi/markets', {
                'limit': limit,
                'status': 'open'
            })
            markets = data.get('markets', [])

            # Filter out sports markets
            sports_keywords = ['nfl', 'nba', 'nhl', 'mlb', 'game', 'vs', 'winner?', 'cfb', 'cbb',
                             'playoff', 'super bowl', 'championship', 'match']

            general = []
            for m in markets:
                title_lower = m.get('title', '').lower()
                ticker = m.get('market_ticker', '').lower()

                is_sports = any(kw in title_lower or kw in ticker for kw in sports_keywords)
                if not is_sports:
                    general.append(m)

            return general
        except Exception as e:
            print(f"Error getting Kalshi markets: {e}")
            return []

    async def get_polymarket_general_markets(self, limit: int = 200) -> List[dict]:
        """Get general Polymarket markets"""
        try:
            url = 'https://gamma-api.polymarket.com/events'
            params = {'limit': limit, 'active': 'true', 'closed': 'false'}
            headers = {'Authorization': f'Bearer {POLYMARKET_BUILDERS_KEY}'}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers, timeout=30) as resp:
                    if resp.status == 200:
                        events = await resp.json()

                        # Filter out sports
                        sports_keywords = ['nfl', 'nba', 'nhl', 'mlb', 'vs', 'game', 'winner',
                                         'playoff', 'super bowl', 'cfb', 'cbb']

                        general = []
                        for event in events:
                            title_lower = (event.get('title') or '').lower()
                            slug = (event.get('slug') or '').lower()

                            is_sports = any(kw in title_lower or kw in slug for kw in sports_keywords)
                            if not is_sports and event.get('markets'):
                                general.append(event)

                        return general
            return []
        except Exception as e:
            print(f"Error getting Polymarket markets: {e}")
            return []

    async def find_general_arbitrage(self, min_diff: float = 3.0, min_similarity: float = 0.5) -> List[dict]:
        """
        Find arbitrage in general (non-sports) markets using fuzzy title matching.

        Args:
            min_diff: Minimum spread percentage to report
            min_similarity: Minimum title similarity (0-1) to consider a match

        Returns:
            List of arbitrage opportunities
        """
        print("Fetching Kalshi markets...")
        kalshi_markets = await self.get_kalshi_general_markets(limit=100)  # Dome API max is 100
        print(f"Got {len(kalshi_markets)} Kalshi markets")

        print("Fetching Polymarket markets...")
        poly_events = await self.get_polymarket_general_markets(limit=200)
        print(f"Got {len(poly_events)} Polymarket events")

        opportunities = []
        matched_pairs = set()

        for k_market in kalshi_markets:
            k_title = k_market.get('title', '')
            k_ticker = k_market.get('market_ticker', '')

            if not k_title:
                continue

            best_match = None
            best_similarity = 0

            for p_event in poly_events:
                p_title = p_event.get('title', '')
                if not p_title:
                    continue

                similarity = self._calculate_similarity(k_title, p_title)

                if similarity > best_similarity and similarity >= min_similarity:
                    best_similarity = similarity
                    best_match = p_event

            if best_match and best_similarity >= min_similarity:
                pair_key = f"{k_ticker}_{best_match.get('slug', '')}"
                if pair_key in matched_pairs:
                    continue
                matched_pairs.add(pair_key)

                # Get prices
                k_price_cents = k_market.get('last_price', 0)
                k_price = k_price_cents / 100 if k_price_cents > 1 else k_price_cents

                # Get Polymarket price from first market
                p_markets = best_match.get('markets', [])
                if not p_markets:
                    continue

                p_market = p_markets[0]
                try:
                    prices_raw = p_market.get('outcomePrices')
                    if isinstance(prices_raw, str):
                        import json
                        prices = json.loads(prices_raw)
                    else:
                        prices = prices_raw or []
                    p_price = float(prices[0]) if prices else 0
                except:
                    continue

                if p_price <= 0.03 or p_price >= 0.97:
                    continue
                if k_price <= 0.03 or k_price >= 0.97:
                    continue

                # Calculate spread
                diff = (k_price - p_price) * 100

                if abs(diff) >= min_diff:
                    if p_price < k_price:
                        direction = 'buy_poly_sell_kalshi'
                        action = f"Buy YES on Polymarket ({p_price*100:.1f}%), Sell YES on Kalshi ({k_price*100:.1f}%)"
                    else:
                        direction = 'buy_kalshi_sell_poly'
                        action = f"Buy YES on Kalshi ({k_price*100:.1f}%), Sell YES on Polymarket ({p_price*100:.1f}%)"

                    # Parse clobTokenIds (can be string or list)
                    clob_tokens = p_market.get('clobTokenIds', [])
                    if isinstance(clob_tokens, str):
                        try:
                            clob_tokens = json.loads(clob_tokens)
                        except:
                            clob_tokens = []
                    p_token = clob_tokens[0] if clob_tokens else ''

                    opportunities.append({
                        'sport': 'GENERAL',
                        'game': k_title[:50],
                        'team': '',
                        'date': '',
                        'polymarket_slug': best_match.get('slug', ''),
                        'polymarket_price': p_price,
                        'polymarket_token': p_token,
                        'kalshi_title': k_title,
                        'kalshi_ticker': k_ticker,
                        'kalshi_price': k_price,
                        'difference': diff,
                        'direction': direction,
                        'action': action,
                        'confidence': best_similarity,
                        'is_profitable': diff > 0
                    })

        # Sort by absolute spread
        opportunities = sorted(opportunities, key=lambda x: abs(x['difference']), reverse=True)
        return opportunities

    # Legacy methods for backwards compatibility
    async def get_polymarket_markets(self, limit: int = 100) -> List[dict]:
        """Get Polymarket markets with prices from gamma API"""
        url = 'https://gamma-api.polymarket.com/events'
        params = {'limit': limit, 'active': 'true', 'closed': 'false'}

        markets = []
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=30) as resp:
                if resp.status == 200:
                    events = await resp.json()
                    for event in events:
                        for m in event.get('markets', []):
                            prices_raw = m.get('outcomePrices')
                            if not prices_raw:
                                continue

                            try:
                                if isinstance(prices_raw, str):
                                    prices = json.loads(prices_raw)
                                else:
                                    prices = prices_raw

                                if prices and len(prices) >= 1:
                                    yes_price = float(prices[0]) if prices[0] else 0

                                    if yes_price > 0:
                                        markets.append({
                                            'title': m.get('question', event.get('title', '')),
                                            'market_slug': m.get('slug', event.get('slug', '')),
                                            'condition_id': m.get('conditionId', ''),
                                            'token_id': m.get('clobTokenIds', [''])[0] if m.get('clobTokenIds') else '',
                                            'price': yes_price,
                                            'volume': m.get('volume', 0),
                                            'event_title': event.get('title', ''),
                                            'end_date': m.get('endDate', event.get('endDate', ''))
                                        })
                            except:
                                continue
        return markets

    async def get_kalshi_markets(self, limit: int = 100) -> List[dict]:
        """Get Kalshi markets from Dome API"""
        try:
            data = await self._request('/kalshi/markets', {
                'limit': limit,
                'status': 'open'
            })
            return data.get('markets', [])
        except Exception as e:
            print(f"Dome Kalshi API error: {e}")
            return []


# ============ ALERT FORMATTERS ============

def format_whale_alert(whale: dict, order: dict) -> str:
    """Format whale alert for TG/Twitter"""
    addr_short = f"{whale['address'][:6]}...{whale['address'][-4:]}"
    pnl = whale['pnl']

    side_emoji = "🟢" if order['side'] == 'BUY' else "🔴"

    return f"""🐋 whale alert

{addr_short} (PnL: ${pnl:,.0f})
{side_emoji} {order['side']} ${order['value']:,.0f}

{order['title']}
{order['token_label']} @ {order['price']:.2f}

polymarket.com/event/{order['market_slug']}"""


def format_large_order_alert(order: dict) -> str:
    """Format large order alert"""
    addr_short = f"{order['user'][:6]}...{order['user'][-4:]}"
    side_emoji = "🟢" if order['side'] == 'BUY' else "🔴"

    return f"""📊 large order

{side_emoji} {order['side']} ${order['value']:,.0f}
{order['shares']:.0f} shares @ {order['price']:.2f}

{order['title']}
outcome: {order['token_label']}

by {addr_short}"""


def format_arbitrage_alert(opp: dict) -> str:
    """Format arbitrage opportunity alert for sports markets"""
    sport = opp.get('sport', '')
    game = opp.get('game', '')
    team = opp.get('team', '')
    date = opp.get('date', '')

    # Build title from game info
    title = f"{sport} {game}"
    if team:
        title = f"{sport} - {team}"

    return f"""arbitrage opportunity

{title}
{date}

Polymarket: {opp['polymarket_price']*100:.1f}%
Kalshi: {opp['kalshi_price']*100:.1f}%
Spread: {opp['difference']:.1f}%

{opp['action']}

polymarket.com/event/{opp['polymarket_slug']}"""


def format_sentiment_alert(market: str, sentiment: dict) -> str:
    """Format order flow sentiment"""
    emoji = "🟢" if sentiment['sentiment'] == 'bullish' else "🔴" if sentiment['sentiment'] == 'bearish' else "⚪"

    return f"""📈 order flow update

{market}

{emoji} Sentiment: {sentiment['sentiment'].upper()}

Buy: ${sentiment['buy_volume']:,.0f} ({sentiment['buy_count']} orders)
Sell: ${sentiment['sell_volume']:,.0f} ({sentiment['sell_count']} orders)

Buy pressure: {sentiment['buy_percentage']:.1f}%"""


# ============ MAIN TEST ============

async def test_all():
    """Test all features"""
    client = DomeClient()

    print("=" * 50)
    print("DOME API INTEGRATION TEST")
    print("=" * 50)

    # Test 1: Large Orders
    print("\n[1] LARGE ORDERS (>$500)")
    try:
        large_orders = await client.get_large_orders(min_value=500, limit=50)
        print(f"Found {len(large_orders)} large orders")
        for o in large_orders[:3]:
            print(f"  {o['side']} ${o['value']:.0f} | {o['title'][:40]}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Order Flow Sentiment
    print("\n[2] ORDER FLOW SENTIMENT")
    try:
        sentiment = await client.get_order_flow_sentiment(limit=100)
        print(f"Sentiment: {sentiment['sentiment']}")
        print(f"Buy: ${sentiment['buy_volume']:.0f} ({sentiment['buy_percentage']:.1f}%)")
        print(f"Sell: ${sentiment['sell_volume']:.0f}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 3: Whale Detection
    print("\n[3] WHALE DETECTION (PnL > $5K)")
    try:
        whales = await client.find_whales_from_orders(min_pnl=5000)
        print(f"Found {len(whales)} whales")
        for w in whales[:3]:
            addr = f"{w['address'][:6]}...{w['address'][-4:]}"
            print(f"  {addr}: ${w['pnl']:,.0f}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 4: Kalshi Markets
    print("\n[4] KALSHI MARKETS")
    try:
        kalshi = await client.get_kalshi_markets(limit=5)
        print(f"Found {len(kalshi)} Kalshi markets")
        for m in kalshi[:3]:
            print(f"  {m.get('title', m.get('market_ticker'))[:50]}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 50)
    print("TEST COMPLETE")


if __name__ == '__main__':
    asyncio.run(test_all())
