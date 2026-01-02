import os
import json
import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, Dict, List, Set
from pathlib import Path

import aiohttp
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message, BotCommand, MenuButtonCommands
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv


# FSM States for waiting user input
class UserStates(StatesGroup):
    waiting_for_deal_link = State()
    waiting_for_watch_link = State()

# Import new features
from features import Watchlist, Categories, Alerts, NewsTracker
from dome_tracker import DomeClient, format_whale_alert, format_large_order_alert, format_sentiment_alert
from payments import payments, PRICES, format_prices, format_balance, format_deposit_instructions

env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

POLYMARKET_API = "https://gamma-api.polymarket.com"
POLYMARKET_DATA_API = "https://data-api.polymarket.com"

# Polymarket Builders API Key (for authenticated requests)
POLYMARKET_BUILDERS_KEY = "019b3895-f661-7002-9826-6d8615dd63cb"
USERS_FILE = "users.json"
SEEN_EVENTS_FILE = "seen_events.json"
KEYWORDS_FILE = "keywords.json"
PAUSED_USERS_FILE = "paused_users.json"
POSTED_EVENTS_FILE = "posted_events.json"
CHECK_INTERVAL = 60  # Check every 60 seconds (1 minute)
NEWS_CHECK_INTERVAL = 300  # Check watchlist news every 5 minutes

# Channel for broadcasting
CHANNEL_ID = -1003253857796

subscribed_users: Set[int] = set()
seen_events: Set[str] = set()
user_keywords: Dict[int, List[str]] = {}
paused_users: Set[int] = set()


class Storage:

    @staticmethod
    def load_users() -> Set[int]:
        if Path(USERS_FILE).exists():
            try:
                with open(USERS_FILE, 'r') as f:
                    data = json.load(f)
                    return set(data.get('users', []))
            except Exception as e:
                logger.error(f"Error loading users: {e}")
        return set()

    @staticmethod
    def save_users(users: Set[int]):
        try:
            with open(USERS_FILE, 'w') as f:
                json.dump({'users': list(users)}, f, indent=2)
            logger.info(f"Saved {len(users)} users")
        except Exception as e:
            logger.error(f"Error saving users: {e}")

    @staticmethod
    def load_seen_events() -> Set[str]:
        if Path(SEEN_EVENTS_FILE).exists():
            try:
                with open(SEEN_EVENTS_FILE, 'r') as f:
                    data = json.load(f)
                    return set(data.get('events', []))
            except Exception as e:
                logger.error(f"Error loading events: {e}")
        return set()

    @staticmethod
    def save_seen_events(events: Set[str]):
        try:
            with open(SEEN_EVENTS_FILE, 'w') as f:
                json.dump({'events': list(events)}, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving events: {e}")

    @staticmethod
    def load_keywords() -> Dict[int, List[str]]:
        if Path(KEYWORDS_FILE).exists():
            try:
                with open(KEYWORDS_FILE, 'r') as f:
                    data = json.load(f)
                    # Convert string keys back to integers
                    return {int(k): v for k, v in data.items()}
            except Exception as e:
                logger.error(f"Error loading keywords: {e}")
        return {}

    @staticmethod
    def save_keywords(keywords: Dict[int, List[str]]):
        try:
            with open(KEYWORDS_FILE, 'w') as f:
                json.dump(keywords, f, indent=2)
            logger.info(f"Saved keywords for {len(keywords)} users")
        except Exception as e:
            logger.error(f"Error saving keywords: {e}")

    @staticmethod
    def load_paused_users() -> Set[int]:
        if Path(PAUSED_USERS_FILE).exists():
            try:
                with open(PAUSED_USERS_FILE, 'r') as f:
                    data = json.load(f)
                    return set(data.get('users', []))
            except Exception as e:
                logger.error(f"Error loading paused users: {e}")
        return set()

    @staticmethod
    def save_paused_users(users: Set[int]):
        try:
            with open(PAUSED_USERS_FILE, 'w') as f:
                json.dump({'users': list(users)}, f, indent=2)
            logger.info(f"Saved {len(users)} paused users")
        except Exception as e:
            logger.error(f"Error saving paused users: {e}")

    @staticmethod
    def load_posted_events() -> List[Dict]:
        """Load recently posted events for extension sync"""
        if Path(POSTED_EVENTS_FILE).exists():
            try:
                with open(POSTED_EVENTS_FILE, 'r') as f:
                    data = json.load(f)
                    return data.get('events', [])
            except Exception as e:
                logger.error(f"Error loading posted events: {e}")
        return []

    @staticmethod
    def save_posted_event(event_data: Dict):
        """Save posted event for extension sync (keep last 50)"""
        try:
            events = Storage.load_posted_events()

            # Add new event with timestamp
            posted_event = {
                'id': event_data.get('id'),
                'slug': event_data.get('slug'),
                'title': event_data.get('title'),
                'volume': event_data.get('volume'),
                'liquidity': event_data.get('liquidity'),
                'markets': event_data.get('markets', []),
                'posted_at': datetime.now().isoformat()
            }

            # Add to beginning (newest first)
            events.insert(0, posted_event)

            # Keep only last 50
            events = events[:50]

            with open(POSTED_EVENTS_FILE, 'w') as f:
                json.dump({'events': events}, f, indent=2)

            logger.info(f"Saved posted event: {event_data.get('title', 'N/A')[:50]}")
        except Exception as e:
            logger.error(f"Error saving posted event: {e}")


class PolymarketAPI:

    @staticmethod
    def matches_keywords(event_data: Dict, keywords: List[str]) -> bool:
        """
        Check if event matches any of the user's keywords.
        Supports:
        - Simple word matching (case-insensitive)
        - Phrase matching with quotes
        - OR logic (comma-separated keywords)

        Examples:
        - btc, eth -> matches events with 'btc' OR 'eth'
        - "united states", election -> matches events with phrase "united states" OR word "election"
        """
        if not keywords:
            return True  # No filters = show all events

        title = event_data.get('title', '').lower()

        # Also check market questions
        markets = event_data.get('markets', [])
        market_text = ' '.join([m.get('question', '').lower() for m in markets])

        # Combined searchable text
        searchable = f"{title} {market_text}"

        # Check each keyword (OR logic)
        for keyword in keywords:
            keyword = keyword.strip()
            if not keyword:
                continue

            # Check if it's a phrase (has quotes)
            if (keyword.startswith('"') and keyword.endswith('"')) or \
               (keyword.startswith("'") and keyword.endswith("'")):
                # Phrase matching - remove quotes
                phrase = keyword[1:-1].lower()
                if phrase in searchable:
                    return True
            else:
                # Simple word matching
                if keyword.lower() in searchable:
                    return True

        return False

    @staticmethod
    async def fetch_market_context(event_slug: str, market_question: str = None, retry: int = 0) -> Optional[str]:
        """
        Fetch Market Context from Polymarket API.
        This provides AI-generated context about the market/event.
        IMPORTANT: Must use event_slug in the prompt parameter, not market_question.
        """
        if not event_slug:
            logger.error("Cannot fetch Market Context: event_slug is empty")
            return None

        # The API only accepts event slugs, not market questions
        url = f"https://polymarket.com/api/grok/event-summary?prompt={event_slug}"

        # Create a timeout object with 120 seconds total timeout
        timeout = aiohttp.ClientTimeout(total=120)

        # Create SSL context that doesn't verify certificates (fixes Windows SSL issues)
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        try:
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                logger.info(f"Fetching Market Context for: {event_slug} (attempt {retry + 1}/2)")
                async with session.post(
                    url,
                    headers={
                        'Content-Type': 'application/json',
                        'Accept': '*/*',
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Authorization': f'Bearer {POLYMARKET_BUILDERS_KEY}'
                    }
                ) as response:
                    logger.info(f"Market Context API status: {response.status}")
                    if response.status == 200:
                        text = await response.text()
                        logger.info(f"Received response of length: {len(text)} chars")
                        if text and len(text) > 50:
                            # Remove sources block if present
                            if '__SOURCES__' in text:
                                text = text.split('__SOURCES__')[0].strip()
                            logger.info(f"✓ Got Market Context (length: {len(text)} chars)")
                            return text
                        else:
                            logger.warning(f"Market Context response too short: {len(text)} chars")
                            logger.warning(f"Response: {text}")
                            # Retry if response is too short and we haven't retried yet
                            if retry < 1 and len(text) < 50:
                                logger.info("Retrying due to short response...")
                                await asyncio.sleep(2)
                                return await PolymarketAPI.fetch_market_context(event_slug, market_question, retry + 1)
                    elif response.status == 400:
                        logger.error(f"Bad Request (400) - Invalid event slug: {event_slug}")
                        error_text = await response.text()
                        logger.error(f"Error response: {error_text}")
                    else:
                        logger.warning(f"Market Context API returned status: {response.status}")
                        error_text = await response.text()
                        logger.warning(f"Error response: {error_text}")
                        # Retry on 5xx errors
                        if retry < 1 and response.status >= 500:
                            logger.info("Retrying due to server error...")
                            await asyncio.sleep(3)
                            return await PolymarketAPI.fetch_market_context(event_slug, market_question, retry + 1)
        except asyncio.TimeoutError:
            logger.error(f"Market Context request timed out after 120 seconds (attempt {retry + 1}/2)")
            # Retry once on timeout
            if retry < 1:
                logger.info("Retrying after timeout...")
                await asyncio.sleep(2)
                return await PolymarketAPI.fetch_market_context(event_slug, market_question, retry + 1)
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching Market Context: {e}")
        except Exception as e:
            logger.error(f"Error fetching Market Context: {e}")
            import traceback
            logger.error(traceback.format_exc())

        return None

    @staticmethod
    async def fetch_ai_analysis(event_slug: str, event_id: str = None) -> Optional[str]:
        """Fetch AI analysis from Polymarket Grok API (legacy method, now uses fetch_market_context)"""
        return await PolymarketAPI.fetch_market_context(event_slug)

    @staticmethod
    async def fetch_event_by_slug(slug: str) -> Optional[Dict]:
        url = f"{POLYMARKET_API}/events"
        params = {"slug": slug}
        headers = {"Authorization": f"Bearer {POLYMARKET_BUILDERS_KEY}"}

        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        try:
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(url, params=params, headers=headers, timeout=15) as response:
                    if response.status == 200:
                        events = await response.json()
                        if isinstance(events, list) and len(events) > 0:
                            return events[0]
                    else:
                        logger.error(f"Failed to fetch event: {response.status}")
        except Exception as e:
            logger.error(f"Error fetching event '{slug}': {e}")

        return None

    @staticmethod
    async def fetch_recent_events(limit: int = 20) -> List[Dict]:
        url = f"{POLYMARKET_API}/events"
        params = {
            "limit": limit,
            "offset": 0,
            "closed": "false",
            "active": "true",
            "order": "createdAt",
            "ascending": "false"
        }
        headers = {"Authorization": f"Bearer {POLYMARKET_BUILDERS_KEY}"}

        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        try:
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(url, params=params, headers=headers, timeout=15) as response:
                    if response.status == 200:
                        events = await response.json()
                        return events if isinstance(events, list) else []
                    else:
                        logger.error(f"Failed to fetch events: {response.status}")
        except Exception as e:
            logger.error(f"Error fetching events: {e}")

        return []

    @staticmethod
    def parse_polymarket_url(url: str) -> Optional[str]:
        pattern = r'polymarket\.com/event/([a-zA-Z0-9\-]+)'
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def format_money(value) -> str:
        try:
            num = float(value) if value else 0
            return f"${num:,.0f}"
        except:
            return "$0"

    @staticmethod
    def format_date(date_str: str) -> str:
        if not date_str:
            return "N/A"
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime('%B %d, %Y at %H:%M UTC')
        except:
            return date_str

    @staticmethod
    def calculate_totals(markets: List[Dict]) -> tuple:
        total_liquidity = 0.0
        total_volume = 0.0
        for market in markets:
            try:
                liquidity = float(market.get('liquidityNum', market.get('liquidity', 0)) or 0)
                total_liquidity += liquidity
            except:
                pass

            try:
                volume = float(market.get('volumeNum', market.get('volume', 0)) or 0)
                total_volume += volume
            except:
                pass

        return total_liquidity, total_volume

    @staticmethod
    def format_event(event_data: Dict) -> str:
        try:
            title = event_data.get('title', 'Unknown Event')
            slug = event_data.get('slug', '')
            markets = event_data.get('markets', [])

            if not markets:
                return "No market data available"

            event_liquidity = event_data.get('liquidity')
            event_volume = event_data.get('volume')

            if event_liquidity is not None and event_volume is not None:
                total_liquidity = float(event_liquidity)
                total_volume = float(event_volume)
            else:
                total_liquidity, total_volume = PolymarketAPI.calculate_totals(markets)

            end_date = event_data.get('endDate')
            if not end_date and markets:
                end_date = markets[0].get('endDate') or markets[0].get('end_date_iso')

            formatted_date = PolymarketAPI.format_date(end_date)

            msg = []
            msg.append(f"🟠 <b>{title}</b>\n")
            msg.append(f"🔗 <b>Link:</b> https://polymarket.com/event/{slug}\n")
            msg.append(f"🟠 <b>Market stats:</b>")
            msg.append(f"<b>Closes:</b> {formatted_date}")
            msg.append(f"<b>Total Liquidity:</b> {PolymarketAPI.format_money(total_liquidity)}")
            msg.append(f"<b>Total Volume:</b> {PolymarketAPI.format_money(total_volume)}\n")

            if len(markets) == 1:
                market = markets[0]
                outcomes = market.get('outcomes', [])

                if outcomes and isinstance(outcomes, str):
                    try:
                        outcomes = json.loads(outcomes)
                    except:
                        outcomes = []

                outcome_prices = market.get('outcomePrices')
                if isinstance(outcome_prices, str):
                    try:
                        outcome_prices = json.loads(outcome_prices)
                    except:
                        outcome_prices = []

                if len(outcomes) == 2:
                    msg.append("🟠 <b>Current Odds:</b>")
                    for idx, outcome in enumerate(outcomes):
                        name = outcome.get('name', outcome) if isinstance(outcome, dict) else outcome
                        if outcome_prices and idx < len(outcome_prices):
                            price = float(outcome_prices[idx])
                            percentage = price * 100 if price <= 1 else price
                            msg.append(f"  • {name}: {percentage:.1f}%")
                else:
                    msg.append("🟠 <b>Options:</b>")
                    for idx, outcome in enumerate(outcomes):
                        name = outcome.get('name', outcome) if isinstance(outcome, dict) else outcome
                        if outcome_prices and idx < len(outcome_prices):
                            price = float(outcome_prices[idx])
                            percentage = price * 100 if price <= 1 else price
                            msg.append(f"  {idx + 1}. {name}: {percentage:.1f}%")
            else:
                # Filter markets with valid data
                valid_markets = []
                for market in markets:
                    market_outcomes = market.get('outcomes', [])
                    if isinstance(market_outcomes, str):
                        try:
                            market_outcomes = json.loads(market_outcomes)
                        except:
                            market_outcomes = []

                    market_prices = market.get('outcomePrices')
                    if isinstance(market_prices, str):
                        try:
                            market_prices = json.loads(market_prices)
                        except:
                            market_prices = []

                    # Only include markets with valid outcomes and prices
                    if market_outcomes and market_prices:
                        valid_markets.append(market)

                msg.append(f"🟠 <b>Markets ({len(valid_markets)}):</b>")
                for idx, market in enumerate(valid_markets, 1):
                    question = market.get('question', f'Market {idx}')
                    msg.append(f"  {idx}. {question}")

                    market_outcomes = market.get('outcomes', [])
                    if isinstance(market_outcomes, str):
                        try:
                            market_outcomes = json.loads(market_outcomes)
                        except:
                            market_outcomes = []

                    market_prices = market.get('outcomePrices')
                    if isinstance(market_prices, str):
                        try:
                            market_prices = json.loads(market_prices)
                        except:
                            market_prices = []

                    if market_outcomes and market_prices:
                        for o_idx, outcome in enumerate(market_outcomes[:5]):
                            o_name = outcome.get('name', outcome) if isinstance(outcome, dict) else outcome
                            if o_idx < len(market_prices):
                                o_price = float(market_prices[o_idx])
                                o_percentage = o_price * 100 if o_price <= 1 else o_price
                                msg.append(f"     • {o_name}: {o_percentage:.1f}%")

            return "\n".join(msg)

        except Exception as e:
            logger.error(f"Error formatting event: {e}")
            return "Error formatting event data"

    @staticmethod
    async def format_event_with_ai(event_data: Dict) -> str:
        """Format event with Market Context from Polymarket"""
        # Get basic format
        basic_msg = PolymarketAPI.format_event(event_data)

        # Get the market question for more specific context
        slug = event_data.get('slug', '')
        markets = event_data.get('markets', [])
        market_question = None

        # Use the first market's question if available
        if markets and len(markets) > 0:
            market_question = markets[0].get('question')

        # Fetch Market Context from Polymarket
        market_context = await PolymarketAPI.fetch_market_context(slug, market_question)

        if market_context:
            context_msg = f"\n\n🧠 <b>Market Context:</b>\n{market_context}"
            # Don't truncate - show full context
            return basic_msg + context_msg

        return basic_msg


class PolydictionsBot:

    def __init__(self, token: str):
        self.bot = Bot(
            token=token,
            default=DefaultBotProperties(parse_mode=ParseMode.HTML)
        )
        self.dp = Dispatcher()

        # Initialize new features
        self.watchlist = Watchlist()
        self.categories = Categories()
        self.alerts = Alerts()
        self.news_tracker = NewsTracker()
        self.dome = DomeClient()

        self.setup_handlers()

        global subscribed_users, seen_events, user_keywords, paused_users
        subscribed_users = Storage.load_users()
        seen_events = Storage.load_seen_events()
        user_keywords = Storage.load_keywords()
        paused_users = Storage.load_paused_users()

        logger.info(f"Loaded {len(subscribed_users)} users, {len(seen_events)} events, "
                   f"{len(user_keywords)} keyword filters, {len(paused_users)} paused users")

    def setup_handlers(self):
        self.dp.message.register(self.cmd_start, Command("start"))
        self.dp.message.register(self.cmd_deal, Command("deal"))
        self.dp.message.register(self.cmd_help, Command("help"))
        self.dp.message.register(self.cmd_keywords, Command("keywords"))
        self.dp.message.register(self.cmd_pause, Command("pause"))
        self.dp.message.register(self.cmd_resume, Command("resume"))

        # New feature handlers
        self.dp.message.register(self.cmd_watch, Command("watch"))
        self.dp.message.register(self.cmd_unwatch, Command("unwatch"))
        self.dp.message.register(self.cmd_watchlist, Command("watchlist"))
        self.dp.message.register(self.cmd_category, Command("category"))
        self.dp.message.register(self.cmd_categories, Command("categories"))
        self.dp.message.register(self.cmd_alert, Command("alert"))
        self.dp.message.register(self.cmd_alerts, Command("alerts"))
        self.dp.message.register(self.cmd_rmalert, Command("rmalert"))
        self.dp.message.register(self.cmd_interval, Command("interval"))

        # Dome API handlers (whale tracker, order flow, arbitrage) - PAID
        self.dp.message.register(self.cmd_whales, Command("whales"))
        self.dp.message.register(self.cmd_flow, Command("flow"))
        self.dp.message.register(self.cmd_orders, Command("orders"))
        self.dp.message.register(self.cmd_arb, Command("arb"))
        self.dp.message.register(self.cmd_arboff, Command("arboff"))

        # Payment system handlers
        self.dp.message.register(self.cmd_balance, Command("balance"))
        self.dp.message.register(self.cmd_deposit, Command("deposit"))
        self.dp.message.register(self.cmd_prices, Command("prices"))
        self.dp.message.register(self.cmd_confirm, Command("confirm"))
        self.dp.message.register(self.cmd_sub, Command("sub"))

        # State handlers - waiting for user input
        self.dp.message.register(self.handle_deal_link, StateFilter(UserStates.waiting_for_deal_link))
        self.dp.message.register(self.handle_watch_link, StateFilter(UserStates.waiting_for_watch_link))

    async def setup_bot_commands(self):
        """Set up the bot menu button with commands"""
        commands = [
            BotCommand(command="start", description="🚀 Subscribe to notifications"),
            BotCommand(command="deal", description="📊 Analyze event with AI"),
            BotCommand(command="sub", description="⚡ Buy arbitrage subscription"),
            BotCommand(command="arb", description="⚡ Enable arbitrage alerts"),
            BotCommand(command="arboff", description="🔕 Disable arbitrage alerts"),
            BotCommand(command="balance", description="💎 Your $POLYD balance"),
            BotCommand(command="deposit", description="💳 Add $POLYD tokens"),
            BotCommand(command="prices", description="💰 View command prices"),
            BotCommand(command="whales", description="🐋 Find top whales [200 $POLYD]"),
            BotCommand(command="flow", description="📈 Order flow [200 $POLYD]"),
            BotCommand(command="orders", description="💰 Large orders [200 $POLYD]"),
            BotCommand(command="watchlist", description="📋 Show your watchlist"),
            BotCommand(command="watch", description="⭐ Add event to watchlist"),
            BotCommand(command="interval", description="⏱️ Set update interval"),
            BotCommand(command="alerts", description="🔔 Show price alerts"),
            BotCommand(command="alert", description="⏰ Set price alert"),
            BotCommand(command="keywords", description="🔍 Set keyword filters"),
            BotCommand(command="categories", description="📂 Show categories"),
            BotCommand(command="pause", description="⏸️ Pause notifications"),
            BotCommand(command="resume", description="▶️ Resume notifications"),
            BotCommand(command="help", description="❓ Show help"),
        ]

        await self.bot.set_my_commands(commands)
        await self.bot.set_chat_menu_button(menu_button=MenuButtonCommands())
        logger.info("Bot menu commands set up successfully")

    async def cmd_start(self, message: Message):
        user_id = message.from_user.id
        subscribed_users.add(user_id)
        Storage.save_users(subscribed_users)

        text = (
            "🎯 <b>Welcome to Polydictions Bot</b>\n\n"
            "Track and analyze Polymarket events.\n\n"
            "<b>📊 Main Commands:</b>\n"
            "/deal &lt;link&gt; - Analyze event\n"
            "/start - Subscribe to notifications\n"
            "/pause - Pause notifications\n"
            "/resume - Resume notifications\n\n"
            "<b>⚡ Arbitrage Alerts:</b>\n"
            "/sub - Buy subscription (7K/week, 30K/month)\n"
            "/arb - Enable arbitrage alerts\n"
            "/arboff - Disable arbitrage alerts\n\n"
            "<b>🐋 Smart Money (200 $POLYD each):</b>\n"
            "/whales - Find top whales by PnL\n"
            "/flow - Order flow sentiment\n"
            "/orders - Large orders alerts\n\n"
            "<b>💎 $POLYD Payments:</b>\n"
            "/balance - Your token balance\n"
            "/deposit - Add $POLYD tokens\n"
            "/prices - View all prices\n\n"
            "<b>🔍 Filters:</b>\n"
            "/keywords - Filter by keywords\n"
            "/category - Filter by category\n"
            "/categories - Show all categories\n\n"
            "<b>📋 Watchlist:</b>\n"
            "/watch &lt;slug&gt; - Add to watchlist\n"
            "/watchlist - Show watchlist\n"
            "/unwatch &lt;slug&gt; - Remove from watchlist\n\n"
            "<b>🔔 Price Alerts:</b>\n"
            "/alert &lt;slug&gt; &gt; &lt;%&gt; - Set alert\n"
            "/alerts - Show alerts\n"
            "/rmalert &lt;#&gt; - Remove alert\n\n"
            "You're now subscribed to new events! 🔔\n"
            "Use /help for more info"
        )

        await message.answer(text)
        logger.info(f"User {user_id} subscribed")

    async def cmd_deal(self, message: Message, state: FSMContext):
        text = message.text or ""
        parts = text.split(maxsplit=1)

        if len(parts) < 2:
            # No link provided - ask for it and wait
            await message.answer(
                "🔗 <b>Send me a Polymarket link</b>\n\n"
                "Example:\nhttps://polymarket.com/event/your-event-slug"
            )
            await state.set_state(UserStates.waiting_for_deal_link)
            return

        url = parts[1].strip()
        slug = PolymarketAPI.parse_polymarket_url(url)

        if not slug:
            await message.answer("❌ Invalid Polymarket URL")
            return

        processing = await message.answer("⏳ Fetching event data...")

        try:
            event_data = await PolymarketAPI.fetch_event_by_slug(slug)

            if not event_data:
                await processing.edit_text("❌ Event not found")
                return

            # Format basic event info
            basic_msg = PolymarketAPI.format_event(event_data)

            # Send basic info first
            await processing.edit_text(basic_msg)

            # Now fetch Market Context (this takes time)
            context_msg = await message.answer("🧠 Generating Market Context... (this may take 10-30 seconds)")

            # Get the market question for more specific context
            markets = event_data.get('markets', [])
            market_question = None
            if markets and len(markets) > 0:
                market_question = markets[0].get('question')

            # Fetch Market Context
            event_slug = event_data.get('slug', '')
            logger.info(f"Attempting to fetch Market Context for slug: {event_slug}")

            market_context = await PolymarketAPI.fetch_market_context(
                event_slug,
                market_question
            )

            if market_context:
                logger.info(f"Successfully fetched Market Context: {len(market_context)} chars")

                # Split into chunks (Telegram limit is 4096, use 3500 to be safe)
                max_chunk = 3500

                async def send_chunked(text: str, first_msg):
                    """Send text in chunks, editing first message and sending rest as new"""
                    if len(text) <= max_chunk:
                        full_text = f"🧠 <b>Market Context:</b>\n\n{text}"
                        if len(full_text) > 4000:
                            full_text = full_text[:3950] + "..."
                        await first_msg.edit_text(full_text)
                        return

                    # Split into chunks
                    chunks = []
                    current = 0
                    while current < len(text):
                        end = min(current + max_chunk, len(text))
                        if end < len(text):
                            # Find break point
                            for sep in ['\n\n', '\n', '. ', ' ']:
                                pos = text.rfind(sep, current + 500, end)
                                if pos > current + 500:
                                    end = pos + len(sep)
                                    break
                        chunk = text[current:end].strip()
                        if chunk:
                            chunks.append(chunk)
                        current = end

                    # Send chunks
                    for idx, chunk in enumerate(chunks):
                        header = f"🧠 <b>Context ({idx+1}/{len(chunks)}):</b>\n\n"
                        msg_text = header + chunk
                        if len(msg_text) > 4000:
                            msg_text = msg_text[:3950] + "..."
                        try:
                            if idx == 0:
                                await first_msg.edit_text(msg_text)
                            else:
                                await message.answer(msg_text)
                            await asyncio.sleep(0.3)
                        except Exception as e:
                            logger.error(f"Chunk {idx+1} error: {e}")

                await send_chunked(market_context, context_msg)
            else:
                logger.error(f"Market Context returned None for slug: {event_slug}")
                await context_msg.edit_text(
                    "⚠️ Market Context generation failed.\n\n"
                    "This can happen if:\n"
                    "• The event is too new\n"
                    "• The API is temporarily unavailable\n"
                    "• The event doesn't have enough data\n\n"
                    "Check bot logs for details."
                )

            logger.info(f"User {message.from_user.id} checked event: {slug}")

        except Exception as e:
            logger.error(f"Error in /deal: {e}")
            await processing.edit_text(f"❌ Error: {str(e)}")

    async def handle_deal_link(self, message: Message, state: FSMContext):
        """Handle link sent after /deal command"""
        # Clear state first
        await state.clear()

        url = message.text.strip() if message.text else ""
        slug = PolymarketAPI.parse_polymarket_url(url)

        # Also try treating input as slug directly
        if not slug:
            slug = url.replace("https://", "").replace("http://", "").strip("/")
            if "/" in slug or " " in slug or len(slug) < 3:
                slug = None

        if not slug:
            await message.answer(
                "❌ Invalid link. Please send a valid Polymarket URL.\n\n"
                "Example: https://polymarket.com/event/your-event-slug"
            )
            return

        # Process the event (same logic as cmd_deal)
        processing = await message.answer("⏳ Fetching event data...")

        try:
            event_data = await PolymarketAPI.fetch_event_by_slug(slug)

            if not event_data:
                await processing.edit_text("❌ Event not found")
                return

            basic_msg = PolymarketAPI.format_event(event_data)
            await processing.edit_text(basic_msg)

            context_msg = await message.answer("🧠 Generating Market Context... (this may take 10-30 seconds)")

            markets = event_data.get('markets', [])
            market_question = markets[0].get('question') if markets else None

            event_slug = event_data.get('slug', '')
            market_context = await PolymarketAPI.fetch_market_context(event_slug, market_question)

            if market_context:
                # Use same chunking logic
                max_chunk = 3500

                async def send_chunked(text: str, first_msg):
                    if len(text) <= max_chunk:
                        full_text = f"🧠 <b>Market Context:</b>\n\n{text}"
                        if len(full_text) > 4000:
                            full_text = full_text[:3950] + "..."
                        await first_msg.edit_text(full_text)
                        return

                    chunks = []
                    current = 0
                    while current < len(text):
                        end = min(current + max_chunk, len(text))
                        if end < len(text):
                            for sep in ['\n\n', '\n', '. ', ' ']:
                                pos = text.rfind(sep, current + 500, end)
                                if pos > current + 500:
                                    end = pos + len(sep)
                                    break
                        chunk = text[current:end].strip()
                        if chunk:
                            chunks.append(chunk)
                        current = end

                    for idx, chunk in enumerate(chunks):
                        header = f"🧠 <b>Context ({idx+1}/{len(chunks)}):</b>\n\n"
                        msg_text = header + chunk
                        if len(msg_text) > 4000:
                            msg_text = msg_text[:3950] + "..."
                        try:
                            if idx == 0:
                                await first_msg.edit_text(msg_text)
                            else:
                                await message.answer(msg_text)
                            await asyncio.sleep(0.3)
                        except Exception as e:
                            logger.error(f"Chunk {idx+1} error: {e}")

                await send_chunked(market_context, context_msg)
            else:
                await context_msg.edit_text(
                    "⚠️ Market Context generation failed.\n\n"
                    "This can happen if:\n"
                    "• The event is too new\n"
                    "• The API is temporarily unavailable"
                )

            logger.info(f"User {message.from_user.id} checked event: {slug}")

        except Exception as e:
            logger.error(f"Error in handle_deal_link: {e}")
            await processing.edit_text(f"❌ Error: {str(e)}")

    async def handle_watch_link(self, message: Message, state: FSMContext):
        """Handle link sent after /watch command"""
        await state.clear()

        url = message.text.strip() if message.text else ""
        slug = PolymarketAPI.parse_polymarket_url(url)

        if not slug:
            slug = url.replace("https://", "").replace("http://", "").strip("/")
            if "/" in slug or " " in slug or len(slug) < 3:
                slug = None

        if not slug:
            await message.answer(
                "❌ Invalid link. Please send a valid Polymarket URL.\n\n"
                "Example: https://polymarket.com/event/your-event-slug"
            )
            return

        user_id = message.from_user.id

        if self.watchlist.add(user_id, slug):
            await message.answer(f"✅ Added <b>{slug}</b> to your watchlist!")
            logger.info(f"User {user_id} added {slug} to watchlist")
        else:
            await message.answer(f"ℹ️ <b>{slug}</b> is already in your watchlist.")

    async def cmd_help(self, message: Message):
        text = (
            "<b>Polydictions Bot</b>\n\n"
            "<b>📊 Main Commands:</b>\n"
            "/deal &lt;link&gt; - Analyze event with Market Context\n"
            "/start - Subscribe to notifications\n"
            "/pause - Pause notifications\n"
            "/resume - Resume notifications\n\n"
            "<b>⚡ Arbitrage Alerts:</b>\n"
            "/sub - Buy subscription (7K/week, 30K/month)\n"
            "/arb - Enable arbitrage alerts\n"
            "/arboff - Disable arbitrage alerts\n\n"
            "<b>🐋 Smart Money (200 $POLYD each):</b>\n"
            "/whales - Find top whales by PnL\n"
            "/flow [slug] - Order flow sentiment\n"
            "/orders [min$] - Large orders alerts\n\n"
            "<b>💎 $POLYD Payments:</b>\n"
            "/balance - Your token balance\n"
            "/deposit - Add $POLYD tokens\n"
            "/prices - View all prices\n\n"
            "<b>🔍 Filters:</b>\n"
            "/keywords - Filter by keywords\n"
            "/category - Filter by category\n"
            "/categories - Show all categories\n\n"
            "<b>📋 Watchlist:</b>\n"
            "/watch &lt;slug&gt; - Add to watchlist\n"
            "/watchlist - Show watchlist\n"
            "/unwatch &lt;slug&gt; - Remove\n\n"
            "<b>🔔 Price Alerts:</b>\n"
            "/alert &lt;slug&gt; &gt; &lt;%&gt; - Set alert\n"
            "/alerts - Show alerts\n"
            "/rmalert &lt;#&gt; - Remove alert\n\n"
            "<b>🔥 Tokenomics:</b>\n"
            "50% burned + 50% to product development"
        )

        await message.answer(text)

    async def cmd_keywords(self, message: Message):
        user_id = message.from_user.id
        text = message.text or ""
        parts = text.split(maxsplit=1)

        # Show current keywords and help
        if len(parts) < 2:
            current = user_keywords.get(user_id, [])
            if current:
                keywords_text = ", ".join(current)
                help_text = (
                    f"<b>Your current keywords:</b>\n{keywords_text}\n\n"
                    "<b>How to use:</b>\n"
                    "/keywords btc, eth, election - Set keywords\n"
                    "/keywords clear - Remove all filters\n\n"
                    "<b>Filter options:</b>\n"
                    "• Simple words: btc, eth, sports\n"
                    "• Phrases: \"united states\", \"world cup\"\n"
                    "• OR logic: keywords separated by commas\n\n"
                    "<b>Examples:</b>\n"
                    "• <code>btc, eth</code> → any event with btc OR eth\n"
                    "• <code>\"united states\", election</code> → phrase + word\n"
                    "• <code>sports, football, basketball</code> → any sports event\n\n"
                    "Only events matching your keywords will be sent!"
                )
            else:
                help_text = (
                    "<b>Keyword Filters</b>\n\n"
                    "Filter events by keywords to see only what matters!\n\n"
                    "<b>How to use:</b>\n"
                    "/keywords btc, eth, election - Set keywords\n"
                    "/keywords clear - Remove all filters\n\n"
                    "<b>Filter options:</b>\n"
                    "• Simple words: btc, eth, sports\n"
                    "• Phrases: \"united states\", \"world cup\"\n"
                    "• OR logic: keywords separated by commas\n\n"
                    "<b>Examples:</b>\n"
                    "• <code>btc, eth</code> → any event with btc OR eth\n"
                    "• <code>\"united states\", election</code> → phrase + word\n"
                    "• <code>sports, football, basketball</code> → any sports event\n\n"
                    "Currently no filters set - you'll receive all events."
                )

            await message.answer(help_text)
            return

        # Parse keywords
        keyword_input = parts[1].strip()

        # Clear keywords
        if keyword_input.lower() == "clear":
            if user_id in user_keywords:
                del user_keywords[user_id]
                Storage.save_keywords(user_keywords)
                await message.answer("✅ All keyword filters removed. You'll receive all events.")
            else:
                await message.answer("You don't have any keyword filters set.")
            return

        # Parse comma-separated keywords
        keywords = [k.strip() for k in keyword_input.split(',')]
        keywords = [k for k in keywords if k]  # Remove empty strings

        if not keywords:
            await message.answer("❌ Please provide at least one keyword.")
            return

        # Save keywords
        user_keywords[user_id] = keywords
        Storage.save_keywords(user_keywords)

        keywords_display = "\n".join([f"  • {k}" for k in keywords])
        await message.answer(
            f"✅ <b>Keywords saved!</b>\n\n"
            f"You will only receive events matching:\n{keywords_display}\n\n"
            f"Use /keywords clear to remove filters."
        )
        logger.info(f"User {user_id} set keywords: {keywords}")

    async def cmd_pause(self, message: Message):
        user_id = message.from_user.id

        if user_id in paused_users:
            await message.answer("You're already paused. Use /resume to resume notifications.")
            return

        paused_users.add(user_id)
        Storage.save_paused_users(paused_users)

        await message.answer(
            "⏸️ <b>Notifications paused</b>\n\n"
            "You won't receive any new event notifications.\n\n"
            "Use /resume when you want to resume notifications."
        )
        logger.info(f"User {user_id} paused notifications")

    async def cmd_resume(self, message: Message):
        user_id = message.from_user.id

        if user_id not in paused_users:
            await message.answer("You're not paused. Notifications are already active!")
            return

        paused_users.remove(user_id)
        Storage.save_paused_users(paused_users)

        keywords_info = ""
        if user_id in user_keywords:
            keywords_info = f"\n\n🔍 Active filters: {', '.join(user_keywords[user_id])}"

        await message.answer(
            f"▶️ <b>Notifications resumed</b>\n\n"
            f"You'll receive new event notifications again!{keywords_info}"
        )
        logger.info(f"User {user_id} resumed notifications")

    # Watchlist commands
    async def cmd_watch(self, message: Message, state: FSMContext):
        """Add event to watchlist"""
        user_id = message.from_user.id
        text = message.text or ""
        parts = text.split(maxsplit=1)

        if len(parts) < 2:
            # No link provided - ask for it and wait
            await message.answer(
                "🔗 <b>Send me a Polymarket link to watch</b>\n\n"
                "Example:\nhttps://polymarket.com/event/btc-price-2025"
            )
            await state.set_state(UserStates.waiting_for_watch_link)
            return

        url_or_slug = parts[1].strip()
        slug = PolymarketAPI.parse_polymarket_url(url_or_slug)
        if not slug:
            slug = url_or_slug

        if self.watchlist.add(user_id, slug):
            await message.answer(f"✅ Added <b>{slug}</b> to your watchlist!\n\n⏳ Fetching Market Context...")
            logger.info(f"User {user_id} added {slug} to watchlist")

            # Fetch and send initial Market Context
            try:
                context = await PolymarketAPI.fetch_market_context(slug)
                if context:
                    # Cache it for future comparison
                    self.news_tracker.check_for_update(slug, context)

                    context_msg = f"🧠 <b>Market Context for {slug}:</b>\n\n{context[:2000]}"
                    if len(context) > 2000:
                        context_msg += "...\n\n<i>Use /deal for full context</i>"
                    await message.answer(context_msg)
                else:
                    await message.answer("⚠️ Could not fetch Market Context for this event.")
            except Exception as e:
                logger.error(f"Error fetching context for {slug}: {e}")
                await message.answer("⚠️ Error fetching Market Context.")
        else:
            # Already in watchlist - check for updates
            await message.answer(f"⏳ <b>{slug}</b> is already in your watchlist. Checking for updates...")

            try:
                context = await PolymarketAPI.fetch_market_context(slug)
                if context:
                    update = self.news_tracker.check_for_update(slug, context)
                    if update:
                        context_msg = f"📰 <b>New updates for {slug}:</b>\n\n{context[:2000]}"
                        if len(context) > 2000:
                            context_msg += "...\n\n<i>Use /deal for full context</i>"
                        await message.answer(context_msg)
                    else:
                        await message.answer(f"ℹ️ No new updates for <b>{slug}</b> in the last 5 minutes.")
                else:
                    await message.answer("⚠️ Could not fetch Market Context.")
            except Exception as e:
                logger.error(f"Error checking updates for {slug}: {e}")
                await message.answer("⚠️ Error checking for updates.")

    async def cmd_unwatch(self, message: Message):
        """Remove event from watchlist"""
        user_id = message.from_user.id
        text = message.text or ""
        parts = text.split(maxsplit=1)

        if len(parts) < 2:
            await message.answer("❌ Please provide an event slug.\n\nExample:\n/unwatch btc-price-2025")
            return

        slug = parts[1].strip()

        if self.watchlist.remove(user_id, slug):
            await message.answer(f"✅ Removed <b>{slug}</b> from your watchlist.")
            logger.info(f"User {user_id} removed {slug} from watchlist")
        else:
            await message.answer("⚠️ Event not found in your watchlist.")

    async def cmd_watchlist(self, message: Message):
        """Show user's watchlist"""
        user_id = message.from_user.id
        watchlist = self.watchlist.get(user_id)

        if not watchlist:
            await message.answer(
                "📋 <b>Your Watchlist is empty</b>\n\n"
                "Add events with:\n/watch &lt;event-slug&gt;"
            )
            return

        msg = ["📋 <b>Your Watchlist:</b>\n"]
        for idx, slug in enumerate(watchlist, 1):
            msg.append(f"{idx}. {slug}")
            msg.append(f"   https://polymarket.com/event/{slug}\n")

        msg.append(f"\n<b>Total:</b> {len(watchlist)} events")
        msg.append("\nUse /unwatch &lt;slug&gt; to remove")

        await message.answer("\n".join(msg))

    # Category filter commands
    async def cmd_category(self, message: Message):
        """Set category filters"""
        user_id = message.from_user.id
        text = message.text or ""
        parts = text.split(maxsplit=1)

        if len(parts) < 2:
            current_cats = self.categories.get_categories(user_id)
            all_cats = self.categories.get_all_categories()

            if current_cats:
                msg = f"<b>Your categories:</b> {', '.join(current_cats)}\n\n"
            else:
                msg = "<b>No category filters set</b>\n\n"

            msg += "<b>Available categories:</b>\n"
            msg += ", ".join(all_cats)
            msg += "\n\n<b>Usage:</b>\n"
            msg += "/category crypto politics\n"
            msg += "/category clear - Remove filters"

            await message.answer(msg)
            return

        categories_input = parts[1].strip()

        if categories_input.lower() == "clear":
            self.categories.clear_categories(user_id)
            await message.answer("✅ Category filters cleared. You'll receive all events.")
            logger.info(f"User {user_id} cleared category filters")
            return

        categories = [c.strip() for c in categories_input.split()]

        if self.categories.set_categories(user_id, categories):
            await message.answer(
                f"✅ <b>Category filters set!</b>\n\n"
                f"You'll only receive events in:\n{', '.join(self.categories.get_categories(user_id))}"
            )
            logger.info(f"User {user_id} set categories: {categories}")
        else:
            await message.answer(
                f"❌ Invalid categories.\n\n"
                f"Available: {', '.join(self.categories.get_all_categories())}"
            )

    async def cmd_categories(self, message: Message):
        """Show all available categories"""
        all_cats = self.categories.get_all_categories()
        msg = "<b>📂 Available Categories:</b>\n\n"
        msg += "\n".join([f"• {cat}" for cat in all_cats])
        msg += "\n\n<b>Usage:</b>\n/category crypto politics"

        await message.answer(msg)

    # Price alert commands
    async def cmd_alert(self, message: Message):
        """Set price alert"""
        user_id = message.from_user.id
        text = message.text or ""
        parts = text.split()

        if len(parts) < 4:
            await message.answer(
                "❌ Invalid format.\n\n"
                "<b>Usage:</b>\n"
                "/alert &lt;event-slug&gt; &gt; &lt;threshold&gt;\n"
                "/alert &lt;event-slug&gt; &lt; &lt;threshold&gt;\n\n"
                "<b>Examples:</b>\n"
                "/alert btc-price-2025 &gt; 70\n"
                "/alert election-winner &lt; 30"
            )
            return

        slug = parts[1]
        condition = parts[2]
        try:
            threshold = float(parts[3])
        except ValueError:
            await message.answer("❌ Threshold must be a number (0-100)")
            return

        if condition not in [">", "<"]:
            await message.answer("❌ Condition must be '>' or '<'")
            return

        if threshold < 0 or threshold > 100:
            await message.answer("❌ Threshold must be between 0 and 100")
            return

        if self.alerts.add_alert(user_id, slug, condition, threshold):
            await message.answer(
                f"✅ <b>Alert set!</b>\n\n"
                f"Event: {slug}\n"
                f"Condition: {condition} {threshold}%\n\n"
                f"You'll be notified when the price crosses this threshold."
            )
            logger.info(f"User {user_id} set alert: {slug} {condition} {threshold}%")
        else:
            await message.answer("⚠️ This alert already exists.")

    async def cmd_alerts(self, message: Message):
        """Show user's alerts"""
        user_id = message.from_user.id
        alerts = self.alerts.get_alerts(user_id)

        if not alerts:
            await message.answer(
                "🔔 <b>No alerts set</b>\n\n"
                "Set alerts with:\n/alert &lt;event-slug&gt; &gt; &lt;threshold&gt;"
            )
            return

        msg = ["🔔 <b>Your Price Alerts:</b>\n"]
        for idx, alert in enumerate(alerts):
            status = "✅ Triggered" if alert.triggered else "⏳ Active"
            msg.append(f"{idx + 1}. {alert.event_slug}")
            msg.append(f"   {alert.condition} {alert.threshold}% - {status}\n")

        msg.append(f"\n<b>Total:</b> {len(alerts)} alerts")
        msg.append("\nUse /rmalert &lt;number&gt; to remove")

        await message.answer("\n".join(msg))

    async def cmd_rmalert(self, message: Message):
        """Remove price alert"""
        user_id = message.from_user.id
        text = message.text or ""
        parts = text.split()

        if len(parts) < 2:
            await message.answer("❌ Please provide alert number.\n\nExample:\n/rmalert 1")
            return

        try:
            index = int(parts[1]) - 1
        except ValueError:
            await message.answer("❌ Invalid number")
            return

        if self.alerts.remove_alert(user_id, index):
            await message.answer("✅ Alert removed!")
            logger.info(f"User {user_id} removed alert {index}")
        else:
            await message.answer("❌ Alert not found.")

    async def cmd_interval(self, message: Message):
        """Set watchlist update interval"""
        user_id = message.from_user.id
        text = message.text or ""
        parts = text.split()

        current_interval = self.news_tracker.get_interval_minutes(user_id)

        if len(parts) < 2:
            await message.answer(
                f"⏱️ <b>Update Interval</b>\n\n"
                f"Current: <b>{current_interval} minutes</b>\n\n"
                f"<b>Usage:</b>\n"
                f"/interval &lt;minutes&gt;\n\n"
                f"<b>Examples:</b>\n"
                f"/interval 3 - every 3 minutes\n"
                f"/interval 10 - every 10 minutes\n"
                f"/interval 30 - every 30 minutes\n\n"
                f"<i>Minimum: 3 minutes</i>"
            )
            return

        try:
            minutes = int(parts[1])
        except ValueError:
            await message.answer("❌ Please provide a number of minutes.")
            return

        if self.news_tracker.set_interval(user_id, minutes):
            await message.answer(
                f"✅ <b>Interval set to {minutes} minutes!</b>\n\n"
                f"You'll receive watchlist updates every {minutes} minutes."
            )
        else:
            await message.answer(
                f"❌ Minimum interval is 3 minutes.\n\n"
                f"Example: /interval 3"
            )

    # ============ DOME API COMMANDS (PAID - 200 $POLYD each) ============

    async def cmd_whales(self, message: Message):
        """Find whales by PnL - PAID COMMAND"""
        user_id = message.from_user.id

        # Check balance
        has_enough, balance, required = payments.check_balance(user_id, 'whales')
        if not has_enough:
            await message.answer(
                f"❌ <b>Insufficient $POLYD balance</b>\n\n"
                f"Required: {required} $POLYD\n"
                f"Your balance: {balance} $POLYD\n\n"
                f"/deposit — Add tokens\n"
                f"/prices — View all prices"
            )
            return

        # Deduct and burn tokens
        if not payments.deduct_and_burn(user_id, 'whales'):
            await message.answer("❌ Payment failed. Try again.")
            return

        processing = await message.answer("🐋 Loading leaderboard... (100 burned 🔥 + 100 to product)")

        try:
            # Use official Polymarket leaderboard API
            whales = await self.dome.get_leaderboard(period='MONTH', limit=25)

            if not whales:
                await processing.edit_text("No data available.")
                return

            # Add delay before sending result
            await asyncio.sleep(4)

            msg = ["🐋 <b>Top Whales (by PnL) | December 2025</b>\n"]
            for w in whales[:20]:
                rank = w['rank']
                addr = w['address']
                username = w['username']
                pnl = w['pnl']
                vol = w['volume']

                # Display name (username or shortened address)
                if username and not username.startswith('0x'):
                    name = username[:15]
                else:
                    name = f"{addr[:6]}...{addr[-4:]}"

                emoji = "🟢" if pnl > 0 else "🔴"
                msg.append(f"{rank}. {emoji} <b>${pnl:,.0f}</b> | {name}")
                msg.append(f"   Vol: ${vol:,.0f}")
                msg.append(f"   <code>{addr}</code>\n")

            new_balance = payments.get_balance(user_id)
            msg.append(f"<i>Found {len(whales[:20])} whales | Balance: {new_balance:,} $POLYD</i>")
            await processing.edit_text("\n".join(msg))
            logger.info(f"User {user_id} checked whales (burned 200 $POLYD)")

        except Exception as e:
            logger.error(f"Error in /whales: {e}")
            await processing.edit_text(f"❌ Error: {str(e)}")

    async def cmd_flow(self, message: Message):
        """Get order flow sentiment - PAID COMMAND"""
        user_id = message.from_user.id

        # Check balance
        has_enough, balance, required = payments.check_balance(user_id, 'flow')
        if not has_enough:
            await message.answer(
                f"❌ <b>Insufficient $POLYD balance</b>\n\n"
                f"Required: {required} $POLYD\n"
                f"Your balance: {balance} $POLYD\n\n"
                f"/deposit — Add tokens\n"
                f"/prices — View all prices"
            )
            return

        # Deduct and burn tokens
        if not payments.deduct_and_burn(user_id, 'flow'):
            await message.answer("❌ Payment failed. Try again.")
            return

        text = message.text or ""
        parts = text.split(maxsplit=1)

        market_slug = None
        if len(parts) > 1:
            market_slug = parts[1].strip()

        processing = await message.answer("📊 Analyzing order flow... (100 burned 🔥 + 100 to product)")

        try:
            # Get more orders for better analysis
            sentiment = await self.dome.get_order_flow_sentiment(market_slug=market_slug, limit=500)

            # Add delay before result
            await asyncio.sleep(4)

            # Determine sentiment emoji and bar
            buy_pct = sentiment['buy_percentage']
            if buy_pct > 60:
                sentiment_emoji = "🟢"
                sentiment_text = "BULLISH"
            elif buy_pct < 40:
                sentiment_emoji = "🔴"
                sentiment_text = "BEARISH"
            else:
                sentiment_emoji = "⚪"
                sentiment_text = "NEUTRAL"

            # Create visual bar
            bar_length = 20
            buy_bars = int(buy_pct / 100 * bar_length)
            sell_bars = bar_length - buy_bars
            bar = "🟩" * buy_bars + "🟥" * sell_bars

            total_volume = sentiment['buy_volume'] + sentiment['sell_volume']
            total_orders = sentiment['buy_count'] + sentiment['sell_count']

            new_balance = payments.get_balance(user_id)

            msg = [
                f"📊 <b>Order Flow Analysis</b>",
                f"",
                f"<b>Market:</b> {market_slug or 'All Polymarket'}",
                f"<b>Sample:</b> {total_orders} orders | ${total_volume:,.0f} volume",
                f"",
                f"{bar}",
                f"",
                f"{sentiment_emoji} <b>Sentiment: {sentiment_text}</b>",
                f"",
                f"┌─────────────────────────────",
                f"│ 🟢 <b>BUY</b>",
                f"│ Volume: <code>${sentiment['buy_volume']:,.0f}</code>",
                f"│ Orders: {sentiment['buy_count']}",
                f"│ Share: <b>{buy_pct:.1f}%</b>",
                f"├─────────────────────────────",
                f"│ 🔴 <b>SELL</b>",
                f"│ Volume: <code>${sentiment['sell_volume']:,.0f}</code>",
                f"│ Orders: {sentiment['sell_count']}",
                f"│ Share: <b>{100 - buy_pct:.1f}%</b>",
                f"└─────────────────────────────",
                f"",
                f"<i>Found {total_orders} orders | Balance: {new_balance:,} $POLYD</i>"
            ]

            await processing.edit_text("\n".join(msg))
            logger.info(f"User {user_id} checked flow (burned 200 $POLYD)")

        except Exception as e:
            logger.error(f"Error in /flow: {e}")
            await processing.edit_text(f"❌ Error: {str(e)}")

    async def cmd_orders(self, message: Message):
        """Get large orders - PAID COMMAND"""
        user_id = message.from_user.id

        # Check balance
        has_enough, balance, required = payments.check_balance(user_id, 'orders')
        if not has_enough:
            await message.answer(
                f"❌ <b>Insufficient $POLYD balance</b>\n\n"
                f"Required: {required} $POLYD\n"
                f"Your balance: {balance} $POLYD\n\n"
                f"/deposit — Add tokens\n"
                f"/prices — View all prices"
            )
            return

        # Deduct and burn tokens
        if not payments.deduct_and_burn(user_id, 'orders'):
            await message.answer("❌ Payment failed. Try again.")
            return

        text = message.text or ""
        parts = text.split()

        min_value = 100  # Default $100 (lowered from $500)
        if len(parts) > 1:
            try:
                min_value = float(parts[1])
            except ValueError:
                pass

        processing = await message.answer(f"📈 Finding orders >${min_value:,.0f}... (100 burned 🔥 + 100 to product)")

        try:
            # Get more orders to find large ones
            orders = await self.dome.get_large_orders(min_value=min_value, limit=200)

            # Add delay before result
            await asyncio.sleep(4)

            if not orders:
                new_balance = payments.get_balance(user_id)
                await processing.edit_text(
                    f"📈 <b>Large Orders</b>\n\n"
                    f"No orders found above ${min_value:,.0f}\n\n"
                    f"<i>Try: /orders 50 for smaller threshold</i>\n\n"
                    f"<i>Balance: {new_balance:,} $POLYD</i>"
                )
                return

            # Calculate stats
            total_value = sum(o['value'] for o in orders)
            buy_orders = [o for o in orders if o['side'] == 'BUY']
            sell_orders = [o for o in orders if o['side'] == 'SELL']
            buy_value = sum(o['value'] for o in buy_orders)
            sell_value = sum(o['value'] for o in sell_orders)

            new_balance = payments.get_balance(user_id)

            msg = [
                f"📈 <b>Large Orders (>${min_value:,.0f})</b>",
                f"",
                f"<b>Total:</b> {len(orders)} orders | ${total_value:,.0f}",
                f"🟢 Buy: {len(buy_orders)} (${buy_value:,.0f}) | 🔴 Sell: {len(sell_orders)} (${sell_value:,.0f})",
                f"",
                f"┌─────────────────────────────"
            ]

            for idx, o in enumerate(orders[:12], 1):
                side_emoji = "🟢" if o['side'] == 'BUY' else "🔴"
                addr = f"{o['user'][:6]}..{o['user'][-4:]}"
                title = o['title'][:30] + ".." if len(o['title']) > 30 else o['title']
                outcome = o.get('token_label', '')[:10]

                msg.append(f"│ {idx}. {side_emoji} <b>${o['value']:,.0f}</b> | {o['shares']:.0f}@{o['price']:.2f}")
                msg.append(f"│    {title}")
                msg.append(f"│    {outcome} • <code>{addr}</code>")
                if idx < min(len(orders), 12):
                    msg.append(f"│")

            msg.append(f"└─────────────────────────────")
            msg.append(f"")
            msg.append(f"<i>Found {len(orders)} orders | Balance: {new_balance:,} $POLYD</i>")

            await processing.edit_text("\n".join(msg))
            logger.info(f"User {user_id} checked orders (burned 200 $POLYD)")

        except Exception as e:
            logger.error(f"Error in /orders: {e}")
            await processing.edit_text(f"❌ Error: {str(e)}")

    async def cmd_arb(self, message: Message):
        """Enable arbitrage alerts - requires subscription"""
        user_id = message.from_user.id

        # Check if has active subscription
        sub = payments.get_subscription(user_id, 'arb')
        if not sub['active']:
            await message.answer(
                "⚡ <b>Arbitrage Alerts</b>\n\n"
                "Get real-time alerts for Polymarket vs Kalshi arbitrage opportunities.\n\n"
                "<b>Subscription required:</b>\n"
                "/sub week — 7 days for 7,000 $POLYD\n"
                "/sub month — 30 days for 30,000 $POLYD\n\n"
                "50% burned + 50% to product development"
            )
            return

        # Check if already enabled
        if payments.is_arb_alerts_enabled(user_id):
            expires = sub['expires_at'][:10]  # YYYY-MM-DD
            await message.answer(
                f"✅ <b>Arb alerts already enabled</b>\n\n"
                f"Subscription active until: <b>{expires}</b>\n\n"
                f"Use /arboff to disable alerts."
            )
            return

        # Enable arb alerts
        payments.enable_arb_alerts(user_id)
        expires = sub['expires_at'][:10]

        await message.answer(
            f"✅ <b>Arb alerts enabled!</b>\n\n"
            f"Subscription active until: <b>{expires}</b>\n\n"
            f"You'll receive real-time alerts when arbitrage opportunities (>2% spread) "
            f"appear between Polymarket and Kalshi.\n\n"
            f"Use /arboff to disable alerts."
        )
        logger.info(f"User {user_id} enabled arb alerts")

    async def cmd_arboff(self, message: Message):
        """Disable arbitrage alerts - toggle OFF"""
        user_id = message.from_user.id

        # Check if enabled
        if not payments.is_arb_alerts_enabled(user_id):
            await message.answer(
                "❌ <b>Arb alerts not enabled</b>\n\n"
                "Use /arb to enable arbitrage alerts."
            )
            return

        # Disable arb alerts
        payments.disable_arb_alerts(user_id)

        await message.answer(
            "🔕 <b>Arb alerts disabled</b>\n\n"
            "You won't receive arbitrage alerts anymore.\n\n"
            "Use /arb to re-enable alerts."
        )
        logger.info(f"User {user_id} disabled arb alerts")

    # ============ PAYMENT COMMANDS ============

    async def cmd_balance(self, message: Message):
        """Show user's $POLYD balance"""
        user_id = message.from_user.id
        await message.answer(format_balance(user_id))

    async def cmd_deposit(self, message: Message):
        """Show deposit instructions"""
        user_id = message.from_user.id
        memo = payments.create_pending_deposit(user_id, 0)
        await message.answer(format_deposit_instructions(memo))

    async def cmd_prices(self, message: Message):
        """Show prices for paid commands"""
        await message.answer(format_prices())

    async def cmd_sub(self, message: Message):
        """Subscribe to arb alerts"""
        user_id = message.from_user.id
        text = message.text or ""
        parts = text.split()

        # Check current subscription
        sub = payments.get_subscription(user_id, 'arb')
        balance = payments.get_balance(user_id)

        if len(parts) < 2:
            # Show subscription options
            sub_status = ""
            if sub['active']:
                expires = sub['expires_at'][:10]
                sub_status = f"\n✅ <b>Active until:</b> {expires}\n"

            await message.answer(
                f"⚡ <b>Arbitrage Subscription</b>\n"
                f"{sub_status}\n"
                f"Get real-time Polymarket vs Kalshi arb alerts.\n\n"
                f"<b>Plans:</b>\n"
                f"/sub week — 7 days for <b>7,000</b> $POLYD\n"
                f"/sub month — 30 days for <b>30,000</b> $POLYD\n\n"
                f"Your balance: <b>{balance:,}</b> $POLYD\n\n"
                f"50% burned + 50% to product development"
            )
            return

        plan = parts[1].lower()

        if plan == 'week':
            sub_type = 'arb_week'
            price = 7000
            days = 7
        elif plan == 'month':
            sub_type = 'arb_month'
            price = 30000
            days = 30
        else:
            await message.answer(
                "❌ Invalid plan. Use:\n"
                "/sub week — 7 days\n"
                "/sub month — 30 days"
            )
            return

        # Check balance
        if balance < price:
            await message.answer(
                f"❌ <b>Insufficient balance</b>\n\n"
                f"Required: {price:,} $POLYD\n"
                f"Your balance: {balance:,} $POLYD\n\n"
                f"/deposit — Add tokens"
            )
            return

        # Process subscription
        if payments.subscribe(user_id, sub_type):
            new_balance = payments.get_balance(user_id)
            new_sub = payments.get_subscription(user_id, 'arb')
            expires = new_sub['expires_at'][:10]

            await message.answer(
                f"✅ <b>Subscription activated!</b>\n\n"
                f"Plan: {days} days\n"
                f"Paid: {price:,} $POLYD\n"
                f"Active until: <b>{expires}</b>\n\n"
                f"New balance: {new_balance:,} $POLYD\n\n"
                f"Use /arb to enable alerts!"
            )
            logger.info(f"User {user_id} subscribed to {sub_type} ({price} $POLYD)")
        else:
            await message.answer("❌ Subscription failed. Try again.")

    async def cmd_confirm(self, message: Message):
        """Confirm a deposit transaction"""
        user_id = message.from_user.id
        text = message.text or ""
        parts = text.split(maxsplit=1)

        if len(parts) < 2:
            await message.answer(
                "❌ Please provide transaction signature\n\n"
                "Example: /confirm 5abc123..."
            )
            return

        tx_signature = parts[1].strip()
        processing = await message.answer("🔍 Verifying transaction...")

        try:
            # Check if transaction already used
            if payments.is_tx_used(tx_signature):
                await processing.edit_text(
                    "❌ <b>Transaction already used</b>\n\n"
                    "This transaction has already been claimed."
                )
                return

            result = await payments.verify_deposit(tx_signature)

            if result['success']:
                amount = result['amount']
                new_balance = payments.add_balance(user_id, amount, tx_signature)

                if new_balance == -1:
                    await processing.edit_text(
                        "❌ <b>Transaction already used</b>\n\n"
                        "This transaction has already been claimed."
                    )
                    return

                await processing.edit_text(
                    f"✅ <b>Deposit confirmed!</b>\n\n"
                    f"Amount: +{amount:,} $POLYD\n"
                    f"New balance: <b>{new_balance:,} $POLYD</b>\n\n"
                    f"Use /whales, /flow, /orders to access Smart Money data!"
                )
                logger.info(f"User {user_id} deposited {amount} $POLYD")
            else:
                await processing.edit_text(
                    f"❌ Could not verify transaction\n\n"
                    f"Error: {result.get('error', 'Unknown')}\n\n"
                    f"Make sure you sent $POLYD tokens to the correct address."
                )

        except Exception as e:
            logger.error(f"Error confirming deposit: {e}")
            await processing.edit_text(f"❌ Error: {str(e)}")

    async def check_new_events(self):
        global seen_events
        logger.info(f"Starting event monitoring with {len(seen_events)} seen events already loaded")

        if not seen_events:
            logger.info("Seen events is empty, initializing with recent 100 events...")
            initial = await PolymarketAPI.fetch_recent_events(limit=100)
            for event in initial:
                event_id = event.get('id')
                if event_id:
                    seen_events.add(str(event_id))
            Storage.save_seen_events(seen_events)
            logger.info(f"Initialized with {len(seen_events)} events")
        else:
            logger.info(f"Using existing {len(seen_events)} seen events from storage")

        while True:
            try:
                await asyncio.sleep(CHECK_INTERVAL)

                recent = await PolymarketAPI.fetch_recent_events(limit=20)
                new_events = []
                filtered_count = 0
                filtered_high_volume = 0

                for event in recent:
                    event_id = str(event.get('id', ''))
                    if event_id:
                        if event_id not in seen_events:
                            # Check if event is actually new (created in last 48 hours)
                            created_at_str = event.get('createdAt') or event.get('startDate')
                            is_actually_new = False

                            if created_at_str:
                                try:
                                    created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                                    now = datetime.now(created_at.tzinfo) if created_at.tzinfo else datetime.now()
                                    age_hours = (now - created_at).total_seconds() / 3600
                                    is_actually_new = age_hours < 48  # Only events created in last 48h
                                except Exception as e:
                                    logger.error(f"Error parsing date {created_at_str}: {e}")
                                    is_actually_new = False

                            volume = float(event.get('volume', 0) or 0)
                            title = event.get('title', '').lower()

                            # Skip spam/repetitive markets (Up or Down, hourly markets, etc.)
                            is_spam_market = any(spam in title for spam in [
                                'up or down',
                                'higher or lower',
                                'above or below',
                                '8:30am',
                                '9:00am',
                                '9:30am',
                                '10:00am',
                                '10:30am',
                                '11:00am',
                                '11:30am',
                                '12:00pm',
                                '12:30pm',
                                '1:00pm',
                                '1:30pm',
                                '2:00pm',
                                '2:30pm',
                                '3:00pm',
                                '3:30pm',
                                '4:00pm',
                                'am-',
                                'pm-',
                                'am et',
                                'pm et',
                            ])

                            if not is_actually_new:
                                # Old event appearing for first time - mark as seen but don't notify
                                seen_events.add(event_id)
                                filtered_high_volume += 1
                                logger.info(f"Filtered old event: ID={event_id}, Created={created_at_str}, Title={event.get('title', 'N/A')[:50]}")
                            elif is_spam_market:
                                # Skip repetitive/spam markets
                                seen_events.add(event_id)
                                filtered_high_volume += 1
                                logger.info(f"Filtered spam market: ID={event_id}, Title={event.get('title', 'N/A')[:50]}")
                            elif volume > 50000:
                                # This is likely an old event with high volume, mark as seen but don't notify
                                seen_events.add(event_id)
                                filtered_high_volume += 1
                                logger.info(f"Filtered high-volume event: ID={event_id}, Volume=${volume:,.0f}, Title={event.get('title', 'N/A')[:50]}")
                            else:
                                # This is a genuinely new event
                                seen_events.add(event_id)
                                new_events.append(event)
                                logger.info(f"New event found: ID={event_id}, Volume=${volume:,.0f}, Title={event.get('title', 'N/A')[:50]}")
                        else:
                            filtered_count += 1

                logger.info(f"Checked {len(recent)} events: {len(new_events)} new, {filtered_count} already seen, {filtered_high_volume} filtered (high volume)")

                if new_events:
                    Storage.save_seen_events(seen_events)
                    logger.info(f"Found {len(new_events)} new events")

                    for event in new_events:
                        formatted = PolymarketAPI.format_event(event)
                        notification = f"🆕 <b>New Polymarket Event</b>\n\n{formatted}"

                        # Post to channel only (no DM to users)
                        if CHANNEL_ID:
                            try:
                                await self.bot.send_message(CHANNEL_ID, notification)
                                logger.info(f"Posted event to channel {CHANNEL_ID}: {event.get('title', 'N/A')[:50]}")

                                # Save to posted_events.json for extension sync
                                Storage.save_posted_event(event)

                                await asyncio.sleep(0.5)
                            except Exception as e:
                                logger.error(f"Failed to post to channel {CHANNEL_ID}: {e}")

            except Exception as e:
                logger.error(f"Error in monitoring: {e}")

    async def check_watchlist_news(self):
        """Monitor watchlist events for news/context updates with per-user intervals"""
        logger.info("Starting watchlist news monitoring...")

        # Track last check time per user: {user_id: timestamp}
        last_check: Dict[int, float] = {}

        # Initial delay - wait 60 seconds before first check
        logger.info("Waiting 60 seconds before first watchlist check...")
        await asyncio.sleep(60)

        while True:
            try:
                current_time = datetime.now().timestamp()

                for user_id, user_slugs in self.watchlist.user_watchlists.items():
                    if not user_slugs:
                        continue

                    # Get user's interval
                    user_interval = self.news_tracker.get_interval(user_id)

                    # Check if it's time to send update to this user
                    user_last = last_check.get(user_id, 0)
                    if current_time - user_last < user_interval:
                        continue  # Not time yet

                    # Update last check time
                    last_check[user_id] = current_time

                    logger.info(f"Checking news for user {user_id}: {len(user_slugs)} events (interval: {user_interval}s)")

                    # Collect status for all slugs
                    updates = []
                    no_updates = []

                    for slug in user_slugs:
                        try:
                            new_context = await PolymarketAPI.fetch_market_context(slug)

                            if not new_context:
                                no_updates.append(slug)
                                continue

                            update = self.news_tracker.check_for_update(slug, new_context)

                            if update:
                                updates.append((slug, new_context))
                                logger.info(f"News update detected for {slug}")
                            else:
                                no_updates.append(slug)

                            await asyncio.sleep(2)

                        except Exception as e:
                            logger.error(f"Error checking news for {slug}: {e}")
                            no_updates.append(slug)

                    # Build and send status message
                    try:
                        msg_parts = []
                        interval_min = self.news_tracker.get_interval_minutes(user_id)

                        for slug, context in updates:
                            msg_parts.append(
                                f"📰 <b>{slug}</b>\n"
                                f"🔗 https://polymarket.com/event/{slug}\n"
                                f"🧠 <b>New Update:</b>\n{context[:800]}"
                            )
                            if len(context) > 800:
                                msg_parts.append("...\n<i>Use /deal for full context</i>")

                        if no_updates:
                            if len(no_updates) == 1:
                                msg_parts.append(f"ℹ️ No new updates for <b>{no_updates[0]}</b>")
                            else:
                                no_update_lines = [f"<b>{slug}</b>\nℹ️ No new updates" for slug in no_updates]
                                msg_parts.append("\n\n".join(no_update_lines))

                        if msg_parts:
                            header = f"📋 <b>Watchlist Status</b> ({datetime.now().strftime('%H:%M')})\n⏱️ Next update in {interval_min} min\n\n"
                            full_msg = header + "\n\n".join(msg_parts)

                            if len(full_msg) > 4000:
                                full_msg = full_msg[:3950] + "\n\n<i>...truncated</i>"

                            await self.bot.send_message(user_id, full_msg)
                            logger.info(f"Sent watchlist status to user {user_id}")
                            await asyncio.sleep(0.5)

                    except Exception as e:
                        logger.error(f"Failed to send status to {user_id}: {e}")

            except Exception as e:
                logger.error(f"Error in news monitoring: {e}")

            # Check every 30 seconds for users who need updates
            await asyncio.sleep(30)

    async def check_arb_alerts(self):
        """Check for arbitrage opportunities and alert users (free)"""
        logger.info("Starting arbitrage alerts monitoring (80% API usage)...")

        # Keep track of recently sent opportunities to avoid spam
        sent_opps = {}  # key: "poly_slug:kalshi_ticker" -> timestamp

        await asyncio.sleep(60)  # Wait 1 minute before first check

        while True:
            try:
                # Get users with active subscription AND alerts enabled
                subscribers = payments.get_all_arb_subscribers()
                alert_users = payments.get_all_arb_alert_users()
                # Only send to users who have both subscription and alerts enabled
                users = [u for u in alert_users if u in subscribers]

                if users:
                    # Find opportunities (using 80% of API limits - more aggressive)
                    opportunities = await self.dome.find_arbitrage_opportunities(min_diff=2.0)

                    if opportunities:
                        # Filter out recently sent (last 20 minutes)
                        import time
                        now = time.time()
                        new_opps = []

                        for opp in opportunities[:5]:  # Top 5
                            key = f"{opp.get('polymarket_slug', '')}:{opp.get('kalshi_ticker', '')}"
                            if key not in sent_opps or (now - sent_opps[key]) > 1200:  # 20 min
                                new_opps.append(opp)
                                sent_opps[key] = now

                        # Clean old entries
                        sent_opps = {k: v for k, v in sent_opps.items() if now - v < 3600}

                        if new_opps:
                            # Send individual alert for each opportunity (like @prediction_xbt format)
                            for opp in new_opps:
                                alert_text = self.format_arb_alert(opp)

                                # Send to all users
                                for user_id in users:
                                    try:
                                        await self.bot.send_message(user_id, alert_text, disable_web_page_preview=True)
                                        await asyncio.sleep(0.05)
                                    except Exception as e:
                                        logger.error(f"Failed to send arb alert to {user_id}: {e}")

                            logger.info(f"Sent {len(new_opps)} arb alerts to {len(users)} users")

            except Exception as e:
                logger.error(f"Error in arb alerts: {e}")

            # Check every 1 minute (80% API usage for aggressive monitoring)
            await asyncio.sleep(60)

    def format_arb_alert(self, opp: dict) -> str:
        """Format arbitrage alert like @prediction_xbt style"""
        poly_pct = opp['polymarket_price'] * 100
        kalshi_pct = opp['kalshi_price'] * 100
        spread = opp['difference']
        confidence = opp.get('confidence', 0)

        # Get titles
        poly_title = opp['polymarket_title']
        kalshi_title = opp.get('kalshi_title', '')

        # Get slugs for links
        poly_slug = opp.get('polymarket_slug', '')
        kalshi_ticker = opp.get('kalshi_ticker', '')

        # Determine action
        if opp['direction'] == 'buy_poly':
            action = "BUY Polymarket / SELL Kalshi"
            arrow = "↗️"
        else:
            action = "BUY Kalshi / SELL Polymarket"
            arrow = "↘️"

        # Confidence indicator
        if confidence >= 0.8:
            conf_emoji = "🟢"
            conf_text = "High"
        elif confidence >= 0.6:
            conf_emoji = "🟡"
            conf_text = "Medium"
        else:
            conf_emoji = "🟠"
            conf_text = "Low"

        # Build message
        msg = [
            f"⚡ <b>Arbitrage Alert</b>",
            f"",
            f"<b>Polymarket:</b>",
            f"{poly_title}",
            f"Price: <b>{poly_pct:.1f}%</b>",
            f"",
            f"<b>Kalshi:</b>",
            f"{kalshi_title}",
            f"Price: <b>{kalshi_pct:.1f}%</b>",
            f"",
            f"{arrow} <b>Spread: {spread:.1f}%</b>",
            f"{conf_emoji} Match confidence: {conf_text}",
            f"<i>{action}</i>",
            f"",
        ]

        # Add links
        if poly_slug:
            msg.append(f"🔗 <a href='https://polymarket.com/event/{poly_slug}'>Polymarket</a>")
        if kalshi_ticker:
            msg.append(f"🔗 <a href='https://kalshi.com/markets/{kalshi_ticker}'>Kalshi</a>")

        msg.append("")
        msg.append("<i>⚠️ DYOR - verify markets match before trading</i>")

        return "\n".join(msg)

    async def start(self):
        # Set up menu commands
        await self.setup_bot_commands()

        # Start API server for Chrome extension sync
        from api_server import APIServer
        api_server = APIServer(port=8765)
        await api_server.start()

        # Start event monitoring
        asyncio.create_task(self.check_new_events())

        # Start watchlist news monitoring
        asyncio.create_task(self.check_watchlist_news())

        # Start arbitrage alerts for subscribers
        asyncio.create_task(self.check_arb_alerts())

        logger.info("Bot started with API server on port 8765")
        await self.dp.start_polling(self.bot, allowed_updates=["message"])


async def main():
    token = None

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from config import BOT_TOKEN
        token = BOT_TOKEN
        logger.info("Loaded token from config.py")
    except ImportError as e:
        logger.error(f"Failed to import config.py: {e}")
        token = os.getenv('BOT_TOKEN')
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        token = os.getenv('BOT_TOKEN')

    if not token:
        logger.error("BOT_TOKEN not found!")
        logger.error(f".env path: {Path(__file__).parent / '.env'}")
        logger.error(f".env exists: {(Path(__file__).parent / '.env').exists()}")
        logger.error("Create .env with: BOT_TOKEN=your_token")
        logger.error("Or config.py with: BOT_TOKEN = 'your_token'")
        return

    token = token.strip()

    bot = PolydictionsBot(token)
    await bot.start()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped")
