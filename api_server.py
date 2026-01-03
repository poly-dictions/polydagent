"""
Unified API server for Polydictions
- Chrome extension sync
- Polyweb frontend API
- Launchpad + Agent runner
"""
import os
import json
import logging
import asyncio
import time
import secrets
import hashlib
import jwt
import aiohttp
import base64
import re
import random
from aiohttp import web
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Set
from dotenv import load_dotenv
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION FROM ENVIRONMENT
# =============================================================================

# Dev mode - MUST be explicitly enabled
DEV_MODE = os.getenv('DEV_MODE', 'false').lower() == 'true'

# API Keys (REQUIRED in production)
POLYMARKET_BUILDERS_KEY = os.getenv('POLYMARKET_BUILDERS_KEY', '')
DOME_API_KEY = os.getenv('DOME_API_KEY', '')

# Session security
SESSION_SECRET = os.getenv('SESSION_SECRET', '')
if not SESSION_SECRET:
    if DEV_MODE:
        SESSION_SECRET = secrets.token_hex(32)
        logger.warning("⚠️ SESSION_SECRET not set, using random value (dev mode)")
    else:
        raise ValueError("SESSION_SECRET environment variable is required in production!")

SESSION_DURATION_HOURS = 24

# CORS - Allowed origins
ALLOWED_ORIGINS = [
    origin.strip() 
    for origin in os.getenv('ALLOWED_ORIGINS', 'https://polydictions.xyz').split(',')
    if origin.strip()
]
if DEV_MODE:
    ALLOWED_ORIGINS.extend(['http://localhost:8765', 'http://localhost:3000', 'http://127.0.0.1:8765'])

# Dynamic.xyz Configuration
DYNAMIC_ENV_ID = os.getenv('DYNAMIC_ENV_ID', 'ea528e4a-63e9-4d21-97e5-1ff2b8a24a85')
DYNAMIC_JWKS_URL = f"https://app.dynamic.xyz/api/v0/sdk/{DYNAMIC_ENV_ID}/.well-known/jwks"

# Solana Token Gating
REQUIRED_TOKEN_MINT = os.getenv('REQUIRED_TOKEN_MINT', 'iATcGSt9DhJF9ZiJ6dmR153N7bW2G4J9dSSDxWSpump')
MIN_TOKEN_BALANCE = int(os.getenv('MIN_TOKEN_BALANCE', '0'))  # TEMP: disabled for screenshot

# Solana RPC endpoints
SOLANA_RPC_ENDPOINTS = [
    "https://api.mainnet-beta.solana.com",
    "https://rpc.ankr.com/solana",
    "https://solana-api.projectserum.com"
]

# Cache for JWKS keys
_jwks_cache = None
_jwks_cache_time = 0
JWKS_CACHE_TTL = 3600  # 1 hour

# =============================================================================
# LAUNCHPAD CONFIGURATION
# =============================================================================

LAUNCHPAD_WALLET_PRIVATE_KEY = os.getenv("LAUNCHPAD_WALLET_PRIVATE_KEY")
LAUNCHPAD_WALLET_PUBLIC_KEY = "8iWGVEYYvrqArN6ChbbLEgsY3eEHeEbwssswYANq2mgS"
POLYD_MINT = "iATcGSt9DhJF9ZiJ6dmR153N7bW2G4J9dSSDxWSpump"
REQUIRED_POLYD_BALANCE = 1_000  # 1K $POLYD required
LAUNCH_FEE_SOL = 0.05
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")
HELIUS_RPC = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"

# TwitterAPI.io config
TWITTERAPI_KEY = os.getenv("TWITTERAPI_KEY", "")
TWITTERAPI_BASE = "https://api.twitterapi.io"
TWITTER_PROXY = os.getenv("TWITTER_PROXY", "")

# X OAuth 2.0 config
X_CLIENT_ID = os.getenv("X_CLIENT_ID", "")
X_CLIENT_SECRET = os.getenv("X_CLIENT_SECRET", "")
X_REDIRECT_URI = os.getenv("X_REDIRECT_URI", "https://polydictions-production.up.railway.app/api/x/callback")
X_OAUTH_STATES: Dict[str, Dict] = {}  # state -> {code_verifier, created_at, launch_id}

# Agent runner config
POLYMARKET_API = "https://gamma-api.polymarket.com"
POLYFACTUAL_API_URL = "https://deep-research-api.thekid-solana.workers.dev/answer"
POLYFACTUAL_API_KEY = os.getenv("POLYFACTUAL_API_KEY", "")
FACTSAI_API_URL = "https://factsai.org/answer"
FACTSAI_API_KEY = os.getenv("FACTSAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Agent intervals
POST_INTERVAL_HOURS = int(os.getenv("AGENT_POST_INTERVAL", "4"))
MENTION_CHECK_INTERVAL = int(os.getenv("MENTION_CHECK_INTERVAL", "60"))

# Niche keywords for filtering Polymarket events
NICHE_KEYWORDS = {
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol", "blockchain", "defi", "nft", "token", "coin", "memecoin", "binance", "coinbase"],
    "politics": ["trump", "biden", "election", "president", "congress", "senate", "democrat", "republican", "vote", "poll", "governor", "mayor", "political"],
    "sports": ["nba", "nfl", "mlb", "nhl", "soccer", "football", "basketball", "baseball", "championship", "playoffs", "super bowl",
               "world cup", "olympics", "tennis", "golf", "ufc", "mma", "boxing", "f1", "formula 1", "nascar", "pga", "wimbledon",
               "champions league", "premier league", "la liga", "bundesliga", "serie a", "ncaa", "march madness", "world series",
               "stanley cup", "mvp", "all-star", "lakers", "celtics", "warriors", "chiefs", "eagles", "cowboys", "yankees", "dodgers"],
    "entertainment": ["movie", "film", "oscar", "emmy", "grammy", "netflix", "disney", "marvel", "actor", "actress", "box office", "album", "song", "concert", "tour"],
    "tech": ["ai", "artificial intelligence", "openai", "google", "apple", "microsoft", "amazon", "meta", "tesla", "elon musk", "chatgpt", "gpt", "robot", "startup"],
    "finance": ["stock", "market", "fed", "interest rate", "inflation", "gdp", "recession", "economy", "s&p", "nasdaq", "dow jones", "treasury", "bonds", "earnings"]
}

# pump.fun API endpoints
PUMPFUN_IPFS = "https://pump.fun/api/ipfs"
PUMPPORTAL_API = "https://pumpportal.fun/api/trade-local"

# Encryption key derivation
def get_encryption_key() -> bytes:
    """Derive encryption key from SESSION_SECRET"""
    secret = SESSION_SECRET
    salt = b"polydictions-launchpad-salt"
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(secret.encode()))
    return key

# Simple TTL Cache for user trade counts
class TTLCache:
    def __init__(self, ttl_seconds: int = 300, max_size: int = 10000):
        self._cache = {}
        self._ttl = ttl_seconds
        self._max_size = max_size

    def get(self, key):
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return value
            del self._cache[key]
        return None

    def set(self, key, value):
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        self._cache[key] = (value, time.time())

    def __contains__(self, key):
        return self.get(key) is not None

WATCHLIST_FILE = "watchlist.json"
SESSIONS_FILE = "wallet_sessions.json"
LAUNCHES_FILE = Path("launches_history.json")
AGENTS_FILE = Path("agents.json")
ANSWERED_MENTIONS_FILE = Path("agent_answered_mentions.json")

# Global stores for launchpad
pending_launches: Dict[str, Dict[str, Any]] = {}
running_agents: Dict[str, Dict[str, Any]] = {}
agent_runner_state = {
    "last_post_time": 0,
    "answered_mentions": {},
    "posted_events": {},
    "mention_start_times": {}
}

# Global server reference for OAuth token refresh
_server_instance: 'APIServer' = None

def set_server_instance(server: 'APIServer'):
    global _server_instance
    _server_instance = server

def get_server_instance() -> 'APIServer':
    return _server_instance

# Lazy init encryption (after SESSION_SECRET is validated)
_fernet = None
def get_fernet():
    global _fernet
    if _fernet is None:
        _fernet = Fernet(get_encryption_key())
    return _fernet

def encrypt_credential(data: str) -> str:
    if not data:
        return ""
    return get_fernet().encrypt(data.encode()).decode()

def decrypt_credential(encrypted_data: str) -> str:
    if not encrypted_data:
        return ""
    try:
        return get_fernet().decrypt(encrypted_data.encode()).decode()
    except Exception as e:
        logger.error(f"Decryption error: {e}")
        return ""

def load_launches():
    global pending_launches
    if LAUNCHES_FILE.exists():
        try:
            with open(LAUNCHES_FILE, "r") as f:
                data = json.load(f)
                for lid, launch in data.items():
                    launch.pop("image_data", None)
                pending_launches = data
                logger.info(f"Loaded {len(pending_launches)} launches from file")
        except Exception as e:
            logger.error(f"Error loading launches: {e}")

def save_launches():
    try:
        save_data = {}
        for lid, launch in pending_launches.items():
            save_data[lid] = {k: v for k, v in launch.items()
                           if k not in ["image_data", "twitter_cookie", "twitter_password",
                                       "twitter_credentials"]}
        with open(LAUNCHES_FILE, "w") as f:
            json.dump(save_data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving launches: {e}")

def load_agents():
    global running_agents
    if AGENTS_FILE.exists():
        try:
            with open(AGENTS_FILE, "r") as f:
                data = json.load(f)
                for aid, agent in data.items():
                    for key in list(agent.keys()):
                        if key.endswith("_encrypted"):
                            original_key = key.replace("_encrypted", "")
                            agent[original_key] = decrypt_credential(agent[key])
                            del agent[key]
                running_agents = data
                logger.info(f"Loaded {len(running_agents)} agents from file")
        except Exception as e:
            logger.error(f"Error loading agents: {e}")

def save_agents():
    try:
        save_data = {}
        for aid, agent in running_agents.items():
            agent_data = {}
            # Fields that need encryption
            sensitive_fields = ["twitter_cookie", "twitter_password", "twitter_email",
                               "twitter_username_cred", "twitter_totp_secret",
                               "x_access_token", "x_refresh_token"]
            for k, v in agent.items():
                if k in sensitive_fields:
                    agent_data[f"{k}_encrypted"] = encrypt_credential(v) if v else ""
                else:
                    agent_data[k] = v
            save_data[aid] = agent_data
        with open(AGENTS_FILE, "w") as f:
            json.dump(save_data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving agents: {e}")

def load_answered_mentions() -> Dict[str, Set[str]]:
    if ANSWERED_MENTIONS_FILE.exists():
        try:
            with open(ANSWERED_MENTIONS_FILE, 'r') as f:
                data = json.load(f)
                return {k: set(v) for k, v in data.items()}
        except:
            pass
    return {}

def save_answered_mentions(answered: Dict[str, Set[str]]):
    try:
        with open(ANSWERED_MENTIONS_FILE, 'w') as f:
            json.dump({k: list(v) for k, v in answered.items()}, f)
    except Exception as e:
        logger.error(f"Error saving answered mentions: {e}")

# =============================================================================
# X OAUTH API CLASS (Official API)
# =============================================================================

class XOAuthAPI:
    """Post tweets using X OAuth 2.0 access tokens (official API)"""

    @staticmethod
    async def post_tweet(access_token: str, text: str, reply_to: str = None) -> Optional[str]:
        """Post a tweet using OAuth access token"""
        payload = {"text": text}
        if reply_to:
            payload["reply"] = {"in_reply_to_tweet_id": reply_to}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.twitter.com/2/tweets",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                ) as resp:
                    result = await resp.json()
                    if resp.status == 201:
                        tweet_id = result.get("data", {}).get("id")
                        logger.info(f"X OAuth tweet posted: {tweet_id}")
                        return tweet_id
                    else:
                        logger.error(f"X OAuth tweet error: {result}")
                        return None
        except Exception as e:
            logger.error(f"X OAuth post_tweet error: {e}")
            return None

    @staticmethod
    async def post_reply(access_token: str, tweet_id: str, text: str) -> Optional[str]:
        """Post a reply using OAuth access token"""
        return await XOAuthAPI.post_tweet(access_token, text, reply_to=tweet_id)

    @staticmethod
    async def post_thread(access_token: str, tweets: List[str]) -> Optional[str]:
        """Post a thread of tweets using OAuth access token"""
        if not tweets:
            return None

        # Post first tweet
        first_tweet_id = await XOAuthAPI.post_tweet(access_token, tweets[0])
        if not first_tweet_id:
            return None

        logger.info(f"X OAuth thread started: {first_tweet_id}")

        # Post replies
        current_reply_to = first_tweet_id
        for i, tweet_text in enumerate(tweets[1:], start=2):
            await asyncio.sleep(2)
            reply_id = await XOAuthAPI.post_reply(access_token, current_reply_to, tweet_text)
            if reply_id:
                logger.info(f"X OAuth thread tweet {i}: {reply_id}")
                current_reply_to = reply_id
            else:
                logger.warning(f"X OAuth thread tweet {i} failed")
                break

        return first_tweet_id


# =============================================================================
# TWITTER API CLASS (TwitterAPI.io - legacy)
# =============================================================================

class TwitterAPI:
    @staticmethod
    async def post_tweet(login_cookies: str, tweet_text: str, force_note_tweet: bool = False) -> Optional[str]:
        url = f"{TWITTERAPI_BASE}/twitter/create_tweet_v2"
        headers = {"Content-Type": "application/json", "X-API-Key": TWITTERAPI_KEY}
        data = {"tweet_text": tweet_text, "login_cookies": login_cookies, "proxy": TWITTER_PROXY}
        # Use note tweet for longer posts (up to 4000 chars) or if forced
        if len(tweet_text) > 280 or force_note_tweet:
            data["is_note_tweet"] = True
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data, timeout=60) as resp:
                    result = await resp.json()
                    if result.get("status") == "success":
                        return result.get("tweet_id") or "posted"
                    logger.warning(f"Tweet failed: {result}")
                    return None
        except Exception as e:
            logger.error(f"Tweet error: {e}")
            return None

    @staticmethod
    async def post_reply(login_cookies: str, tweet_id: str, reply_text: str) -> Optional[str]:
        url = f"{TWITTERAPI_BASE}/twitter/create_tweet_v2"
        headers = {"Content-Type": "application/json", "X-API-Key": TWITTERAPI_KEY}
        data = {"tweet_text": reply_text, "login_cookies": login_cookies, "proxy": TWITTER_PROXY, "reply_to_tweet_id": tweet_id}
        if len(reply_text) > 280:
            data["is_note_tweet"] = True
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data, timeout=60) as resp:
                    result = await resp.json()
                    if result.get("status") == "success":
                        return result.get("tweet_id") or "replied"
                    logger.warning(f"Reply failed: {result}")
                    return None
        except Exception as e:
            logger.error(f"Reply error: {e}")
            return None

    @staticmethod
    async def post_thread(login_cookies: str, tweets: List[str]) -> Optional[str]:
        """Post a thread of tweets, returns ID of first tweet"""
        if not tweets:
            return None

        # Post first tweet
        first_tweet_id = await TwitterAPI.post_tweet(login_cookies, tweets[0])
        if not first_tweet_id:
            return None

        logger.info(f"Thread started with tweet ID: {first_tweet_id}")

        # Post replies
        current_reply_to = first_tweet_id
        for i, tweet_text in enumerate(tweets[1:], start=2):
            await asyncio.sleep(2)  # Small delay between tweets
            reply_id = await TwitterAPI.post_reply(login_cookies, current_reply_to, tweet_text)
            if reply_id:
                logger.info(f"Thread tweet {i} posted: {reply_id}")
                current_reply_to = reply_id
            else:
                logger.warning(f"Thread tweet {i} failed")
                break

        return first_tweet_id

    @staticmethod
    async def get_mentions(username: str, since_time: int = None) -> List[Dict]:
        url = f"{TWITTERAPI_BASE}/twitter/user/mentions"
        params = {"userName": username}
        if since_time:
            params["sinceTime"] = since_time
        headers = {"X-API-Key": TWITTERAPI_KEY}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("status") == "success":
                            return data.get("tweets", [])
                    return []
        except Exception as e:
            logger.error(f"Get mentions error: {e}")
            return []

    @staticmethod
    async def login(username: str, email: str, password: str, totp_secret: str) -> Optional[str]:
        url = f"{TWITTERAPI_BASE}/twitter/user_login_v2"
        headers = {"Content-Type": "application/json", "X-API-Key": TWITTERAPI_KEY}
        data = {"user_name": username, "email": email, "password": password, "proxy": TWITTER_PROXY, "totp_secret": totp_secret}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data, timeout=120) as resp:
                    result = await resp.json()
                    if result.get("status") == "success":
                        return result.get("login_cookies") or result.get("login_cookie")
                    logger.warning(f"Twitter login failed: {result}")
                    return None
        except Exception as e:
            logger.error(f"Twitter login error: {e}")
            return None

    @staticmethod
    async def check_blue_verified(username: str) -> bool:
        """Check if Twitter account has blue verification (Twitter Blue/Premium)"""
        url = f"{TWITTERAPI_BASE}/twitter/user/info"
        headers = {"X-API-Key": TWITTERAPI_KEY}
        params = {"userName": username}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        user_data = data.get("data", {})
                        # Check for blue verified badge
                        is_blue = user_data.get("is_blue_verified", False)
                        verified_type = user_data.get("verified_type", "")
                        logger.info(f"Twitter @{username}: blue_verified={is_blue}, verified_type={verified_type}")
                        return is_blue or verified_type == "Blue"
                    return False
        except Exception as e:
            logger.error(f"Check blue verified error: {e}")
            return False

# =============================================================================
# POLYMARKET SCANNER CLASS
# =============================================================================

class PolymarketScanner:
    SPAM_WORDS = ['up or down', 'higher or lower', 'above or below', 'am-', 'pm-', 'am et', 'pm et', ':00', ':15', ':30', ':45']

    async def fetch_events(self, limit: int = 100) -> List[Dict]:
        url = f"{POLYMARKET_API}/events"
        params = {'limit': limit, 'active': 'true', 'closed': 'false', 'order': 'volume', 'ascending': 'false'}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception as e:
            logger.error(f"Fetch error: {e}")
        return []

    def filter_by_niche(self, events: List[Dict], niche: str) -> List[Dict]:
        if niche == "general":
            return events
        keywords = NICHE_KEYWORDS.get(niche, [])
        if not keywords:
            logger.warning(f"No keywords for niche '{niche}', returning all events")
            return events
        filtered = []
        for event in events:
            title = event.get("title", "").lower()
            description = event.get("description", "").lower()
            combined = f"{title} {description}"
            if any(kw in combined for kw in keywords):
                filtered.append(event)
        logger.info(f"Filtered {len(events)} events by niche '{niche}': {len(filtered)} matches")
        # If niche is specified but no matches found, return empty list (don't fallback to all)
        # This ensures agents only post about their niche
        return filtered

    def is_valid_market(self, event: Dict, posted_ids: set) -> bool:
        event_id = event.get('id', '')
        title = event.get('title', '').lower()
        volume = float(event.get('volume', 0) or 0)
        markets = event.get('markets', [])
        if event_id in posted_ids:
            return False
        if any(spam in title for spam in self.SPAM_WORDS):
            return False
        if volume < 50000:
            return False
        if not markets or len(markets) > 3:
            return False
        market = markets[0]
        outcomes = market.get('outcomes', [])
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if len(outcomes) != 2:
            return False
        outcome_prices = market.get('outcomePrices')
        if isinstance(outcome_prices, str):
            outcome_prices = json.loads(outcome_prices)
        if outcome_prices:
            yes_pct = float(outcome_prices[0]) * 100
            if yes_pct >= 92 or yes_pct <= 8:
                return False
        return True

    def parse_market_data(self, event: Dict) -> Dict:
        market = event.get('markets', [{}])[0]
        outcome_prices = market.get('outcomePrices')
        if isinstance(outcome_prices, str):
            outcome_prices = json.loads(outcome_prices)
        yes_pct = float(outcome_prices[0]) * 100 if outcome_prices else 50
        return {
            'event_id': event.get('id', ''),
            'title': event.get('title', ''),
            'slug': event.get('slug', ''),
            'yes_odds': yes_pct,
            'no_odds': 100 - yes_pct,
            'volume': float(event.get('volume', 0) or 0),
            'liquidity': float(event.get('liquidity', 0) or market.get('liquidity', 0) or 0),
            'end_date': event.get('endDate') or market.get('endDate')
        }

# =============================================================================
# AI ANALYZER CLASS
# =============================================================================

class AIAnalyzer:
    @staticmethod
    async def analyze_market(title: str, yes_odds: float, no_odds: float, volume: float, custom_prompt: str = "", niche: str = "general") -> Optional[Dict]:
        if not ANTHROPIC_API_KEY:
            return {
                "signal": "uncertain",
                "confidence": "low",
                "reasons": ["insufficient data"],
                "main_risk": "market uncertainty",
                "notes": ""
            }

        # First get facts from FactsAI
        facts = ""
        if FACTSAI_API_KEY:
            try:
                headers = {"Authorization": f"Bearer {FACTSAI_API_KEY}", "Content-Type": "application/json"}
                data = {"query": f"Latest news and context about: {title}", "text": True}
                async with aiohttp.ClientSession() as session:
                    async with session.post(FACTSAI_API_URL, headers=headers, json=data, timeout=60) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            if result.get("success") and result.get("data"):
                                facts = result["data"].get("answer", "") or result["data"].get("text", "")
                                if isinstance(facts, dict):
                                    facts = facts.get("text", str(facts))
                                logger.info(f"Got facts: {str(facts)[:100]}...")
            except Exception as e:
                logger.warning(f"FactsAI error: {e}")

        # Use Anthropic Claude for structured analysis
        prompt = f"""You are a prediction market analyst. Analyze this market and return a JSON response.

Market: {title}
Current odds: YES {yes_odds:.0f}% / NO {no_odds:.0f}%
Volume: ${volume:,.0f}

{f'CONTEXT/NEWS: {str(facts)[:1500]}' if facts else ''}

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{{
  "signal": "yes" or "no",
  "confidence": "high" or "mid" or "low",
  "reasons": ["reason 1 (max 50 chars)", "reason 2", "reason 3"],
  "main_risk": "primary risk or uncertainty (max 80 chars)",
  "notes": "optional macro/timing context (max 60 chars, or empty string)"
}}

{f'Style note: {custom_prompt}' if custom_prompt else ''}"""

        headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
        data = {"model": "claude-3-haiku-20240307", "max_tokens": 400, "messages": [{"role": "user", "content": prompt}]}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(ANTHROPIC_API_URL, headers=headers, json=data, timeout=60) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        text = result.get("content", [{}])[0].get("text", "")
                        # Parse JSON response
                        try:
                            # Clean up potential markdown
                            text = text.strip()
                            if text.startswith("```"):
                                text = text.split("```")[1]
                                if text.startswith("json"):
                                    text = text[4:]
                            analysis = json.loads(text.strip())
                            logger.info(f"AI analysis: signal={analysis.get('signal')}, confidence={analysis.get('confidence')}")
                            return analysis
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON parse error: {e}, raw: {text[:200]}")
                            # Fallback: extract what we can
                            return {
                                "signal": "yes" if yes_odds > 50 else "no",
                                "confidence": "low",
                                "reasons": ["AI analysis parsing failed"],
                                "main_risk": "analysis uncertainty",
                                "notes": ""
                            }
                    else:
                        error = await resp.text()
                        logger.error(f"Anthropic API error: {resp.status} - {error[:200]}")
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
        return None

    @staticmethod
    async def get_facts(question: str) -> Optional[str]:
        if not FACTSAI_API_KEY:
            return None
        headers = {"Authorization": f"Bearer {FACTSAI_API_KEY}", "Content-Type": "application/json"}
        data = {"query": f"Research this thoroughly: {question}", "text": True}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(FACTSAI_API_URL, headers=headers, json=data, timeout=120) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        if result.get("success") and result.get("data"):
                            answer = result["data"].get("answer", "") or result["data"].get("text", "")
                            if isinstance(answer, dict):
                                answer = answer.get("text", str(answer))
                            return str(answer).strip()
        except Exception as e:
            logger.error(f"FactsAI error: {e}")
        return None

    @staticmethod
    async def generate_reply(question: str, facts: str, custom_prompt: str = "", ticker: str = "") -> Optional[str]:
        if not ANTHROPIC_API_KEY:
            response = facts[:400] if facts else "interesting question! let me look into that."
            if ticker:
                response += f"\n\n${ticker}"
            return response
        prompt = f"""You are a Polymarket prediction expert agent for ${ticker if ticker else 'a token'}.
USER QUESTION: {question}
RESEARCH DATA:
{facts[:2000] if facts else 'No additional data available.'}
Write a helpful response (300-400 chars):
- Answer the question directly
- Use key facts if relevant
- If it's about a prediction market, give your call (YES/NO)
- Be conversational, lowercase
- End with your token ticker if provided
{f'PERSONALITY: {custom_prompt}' if custom_prompt else ''}
Output ONLY the response, no quotes or prefixes."""
        headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
        data = {"model": "claude-3-haiku-20240307", "max_tokens": 300, "messages": [{"role": "user", "content": prompt}]}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(ANTHROPIC_API_URL, headers=headers, json=data, timeout=60) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        text = result.get("content", [{}])[0].get("text", "")
                        response = text.strip().lower()
                        if ticker and f"${ticker.lower()}" not in response:
                            response += f"\n\n${ticker}"
                        return response[:500]
        except Exception as e:
            logger.error(f"Claude error: {e}")
        return None

# =============================================================================
# AGENT RUNNER HELPERS
# =============================================================================

def format_volume(volume: float) -> str:
    if volume >= 1_000_000:
        return f"${volume/1_000_000:.1f}M"
    elif volume >= 1_000:
        return f"${volume/1_000:.0f}K"
    return f"${volume:.0f}"

def create_tweet(agent: Dict, market: Dict, analysis: Dict) -> str:
    """Create tweet in new format (requires Twitter Blue for long posts)"""
    signal = analysis.get("signal", "uncertain")
    confidence = analysis.get("confidence", "low")
    reasons = analysis.get("reasons", ["market dynamics unclear"])
    main_risk = analysis.get("main_risk", "outcome uncertainty")
    notes = analysis.get("notes", "")

    # Format volume and liquidity
    volume_str = format_volume(market['volume'])
    liquidity_str = format_volume(market.get('liquidity', 0))

    # Calculate time to resolution
    end_date = market.get('end_date')
    if end_date:
        try:
            from datetime import datetime
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            days_left = (end_dt - now).days
            if days_left > 30:
                time_str = f"{days_left // 30} months"
            else:
                time_str = f"{days_left} days"
        except:
            time_str = "tbd"
    else:
        time_str = "tbd"

    # Build reasoning bullets
    reasoning_lines = ""
    for i, reason in enumerate(reasons[:3]):
        reasoning_lines += f"• {reason}\n"

    tweet = f"""worth a look

{market['title']}

yes {market['yes_odds']:.0f}% / no {market['no_odds']:.0f}%
volume: {volume_str}
liquidity: {liquidity_str}
time to resolution: {time_str}

market link:
polymarket.com/event/{market['slug']}

signal: {signal} with {confidence} confidence

reasoning:
{reasoning_lines.strip()}

main risk: {main_risk}

{"notes: " + notes + chr(10) + chr(10) if notes else ""}by @polydictions"""

    return tweet.strip()

def create_tweet_thread(agent: Dict, market: Dict, analysis: Dict) -> List[str]:
    """Create a thread of tweets instead of one long tweet"""
    signal = analysis.get("signal", "uncertain")
    confidence = analysis.get("confidence", "low")
    reasons = analysis.get("reasons", ["market dynamics unclear"])
    main_risk = analysis.get("main_risk", "outcome uncertainty")
    notes = analysis.get("notes", "")

    volume_str = format_volume(market['volume'])
    liquidity_str = format_volume(market.get('liquidity', 0))

    # Calculate time to resolution
    end_date = market.get('end_date')
    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            days_left = (end_dt - now).days
            if days_left > 30:
                time_str = f"{days_left // 30} months"
            elif days_left > 0:
                time_str = f"{days_left} days"
            else:
                time_str = "ending soon"
        except:
            time_str = "tbd"
    else:
        time_str = "tbd"

    tweets = []

    # Tweet 1: Hook + Title + key stats
    tweet1 = f"""market worth watching

{market['title']}

current odds:
• YES: {market['yes_odds']:.0f}%
• NO: {market['no_odds']:.0f}%

volume: {volume_str}
liquidity: {liquidity_str}
resolves in: {time_str}

polymarket.com/event/{market['slug']}"""
    tweets.append(tweet1)

    # Tweet 2: Signal + all reasoning points
    reasoning_lines = "\n".join([f"• {r}" for r in reasons[:4]])
    signal_emoji = "🟢" if signal == "yes" else "🔴" if signal == "no" else "⚪"
    tweet2 = f"""my take: {signal.upper()} {signal_emoji}
confidence: {confidence}

why i think this:

{reasoning_lines}"""
    tweets.append(tweet2)

    # Tweet 3: Risk analysis
    tweet3 = f"""key risk to consider:

{main_risk}

always do your own research. prediction markets can be volatile and odds can shift quickly based on new information."""
    tweets.append(tweet3)

    # Tweet 4: Notes + call to action + signature
    if notes:
        tweet4 = f"""{notes}

what do you think? drop your take below

powered by @polydictions
track more markets: polydictions.xyz"""
    else:
        tweet4 = f"""what do you think? drop your take below

powered by @polydictions
track more markets: polydictions.xyz"""
    tweets.append(tweet4)

    return tweets

def create_tweet_short(agent: Dict, market: Dict, analysis: Dict) -> str:
    """Create short tweet (280 chars) for accounts without Twitter Blue"""
    signal = analysis.get("signal", "uncertain")
    confidence = analysis.get("confidence", "low")
    reasons = analysis.get("reasons", ["market dynamics unclear"])
    main_risk = analysis.get("main_risk", "outcome uncertainty")

    reason_short = reasons[0][:45] if reasons else "market dynamics"
    risk_short = main_risk[:40] if main_risk else "uncertainty"

    tweet = f"""{market['title']}

{market['yes_odds']:.0f}% yes / {market['no_odds']:.0f}% no • {format_volume(market['volume'])}

signal: {signal.upper()} ({confidence})
• {reason_short}
risk: {risk_short}

polymarket.com/event/{market['slug']}

@polydictions"""

    if len(tweet) > 280:
        tweet = tweet[:277] + "..."
    return tweet

async def try_agent_relogin(agent: Dict) -> Optional[str]:
    username = agent.get("twitter_username_cred")
    email = agent.get("twitter_email")
    password = agent.get("twitter_password")
    totp = agent.get("twitter_totp_secret")
    if not all([username, email, password, totp]):
        return None
    logger.info(f"Attempting re-login for @{username}...")
    new_cookie = await TwitterAPI.login(username, email, password, totp)
    if new_cookie:
        logger.info("Re-login successful!")
    return new_cookie

async def process_agent_posting(agent_id: str, agent: Dict, scanner: PolymarketScanner, server: 'PolydictionsServer' = None) -> bool:
    niche = agent.get("agent_niche", "general")
    custom_prompt = agent.get("custom_prompt", "")

    # Check for OAuth tokens first, fallback to cookie
    x_access_token = agent.get("x_access_token")
    x_refresh_token = agent.get("x_refresh_token")
    twitter_cookie = agent.get("twitter_cookie")

    use_oauth = bool(x_access_token and x_refresh_token)

    if not use_oauth and not twitter_cookie:
        logger.warning(f"[{agent_id}] No OAuth tokens or cookie available")
        return False

    # Cooldown: don't post more than once per POST_INTERVAL_HOURS
    last_post = agent_runner_state.get("last_agent_post", {}).get(agent_id, 0)
    if time.time() - last_post < POST_INTERVAL_HOURS * 3600:
        logger.info(f"[{agent_id}] Cooldown active ({POST_INTERVAL_HOURS}h), skipping post")
        return False
    logger.info(f"[{agent_id}] Scanning {niche} markets...")
    posted_events = agent_runner_state["posted_events"].get(agent_id, set())
    events = await scanner.fetch_events(limit=100)
    events = scanner.filter_by_niche(events, niche)
    valid_markets = []
    for event in events:
        if scanner.is_valid_market(event, posted_events):
            market_data = scanner.parse_market_data(event)
            valid_markets.append(market_data)
    if not valid_markets:
        logger.info(f"[{agent_id}] No valid markets found for niche '{niche}'")
        return False
    top_markets = sorted(valid_markets, key=lambda x: x['volume'], reverse=True)[:10]
    logger.info(f"[{agent_id}] Top 10 markets: {[m['title'][:40] for m in top_markets]}")
    market = random.choice(top_markets)
    logger.info(f"[{agent_id}] Selected: {market['title']}")
    analysis = await AIAnalyzer.analyze_market(
        market['title'], market['yes_odds'], market['no_odds'], market['volume'],
        custom_prompt=custom_prompt, niche=niche
    )
    if not analysis:
        analysis = {
            "signal": "yes" if market['yes_odds'] > 50 else "no",
            "confidence": "low",
            "reasons": ["market dynamics favor this outcome"],
            "main_risk": "unexpected developments",
            "notes": ""
        }
    # Create thread of tweets
    thread_tweets = create_tweet_thread(agent, market, analysis)
    logger.info(f"[{agent_id}] Posting thread with {len(thread_tweets)} tweets...")

    tweet_id = None

    if use_oauth and server:
        # Use X OAuth API
        valid_token = await server.get_valid_x_token(agent)
        if valid_token:
            tweet_id = await XOAuthAPI.post_thread(valid_token, thread_tweets)
        else:
            logger.error(f"[{agent_id}] Failed to get valid OAuth token")
    elif twitter_cookie:
        # Fallback to legacy TwitterAPI.io
        tweet_id = await TwitterAPI.post_thread(twitter_cookie, thread_tweets)

    if tweet_id:
        logger.info(f"[{agent_id}] Thread posted! First ID: {tweet_id}")
        if agent_id not in agent_runner_state["posted_events"]:
            agent_runner_state["posted_events"][agent_id] = set()
        agent_runner_state["posted_events"][agent_id].add(market['event_id'])
        # Record last post time for cooldown
        if "last_agent_post" not in agent_runner_state:
            agent_runner_state["last_agent_post"] = {}
        agent_runner_state["last_agent_post"][agent_id] = time.time()
        return True
    else:
        # If OAuth failed, we can't recover (no re-login for OAuth)
        # For legacy cookie, try re-login
        if not use_oauth and twitter_cookie:
            new_cookie = await try_agent_relogin(agent)
            if new_cookie:
                agent["twitter_cookie"] = new_cookie
                running_agents[agent_id]["twitter_cookie"] = new_cookie
                save_agents()
                # Retry posting with new cookie
                tweet_id = await TwitterAPI.post_thread(new_cookie, thread_tweets)
                if tweet_id:
                    logger.info(f"[{agent_id}] Thread posted after re-login! First ID: {tweet_id}")
                    if agent_id not in agent_runner_state["posted_events"]:
                        agent_runner_state["posted_events"][agent_id] = set()
                    agent_runner_state["posted_events"][agent_id].add(market['event_id'])
                    # Record last post time for cooldown
                    if "last_agent_post" not in agent_runner_state:
                        agent_runner_state["last_agent_post"] = {}
                    agent_runner_state["last_agent_post"][agent_id] = time.time()
                    return True
        return False

async def process_agent_mentions(agent_id: str, agent: Dict, server: 'PolydictionsServer' = None) -> int:
    username = agent.get("twitter_username")
    twitter_cookie = agent.get("twitter_cookie")
    custom_prompt = agent.get("custom_prompt", "")
    ticker = agent.get("token_ticker", "")

    # Check for OAuth tokens
    x_access_token = agent.get("x_access_token")
    x_refresh_token = agent.get("x_refresh_token")
    use_oauth = bool(x_access_token and x_refresh_token)

    if not username or (not use_oauth and not twitter_cookie):
        return 0
    mention_start = agent_runner_state["mention_start_times"].get(agent_id, int(datetime.now().timestamp()))
    mentions = await TwitterAPI.get_mentions(username, since_time=mention_start)
    answered = agent_runner_state["answered_mentions"].get(agent_id, set())
    real_mentions = [
        m for m in mentions
        if f"@{username.lower()}" in m.get("text", "").lower()
        and m.get("id") not in answered
        and m.get("author", {}).get("userName", "").lower() != username.lower()
    ]
    if not real_mentions:
        return 0
    answered_count = 0
    for mention in real_mentions[:5]:
        tweet_id = mention.get("id")
        text = mention.get("text", "")
        question = re.sub(r'@\w+\s*', '', text).strip()
        if len(question) < 5:
            continue
        facts = await AIAnalyzer.get_facts(question)
        reply = await AIAnalyzer.generate_reply(question=question, facts=facts or "", custom_prompt=custom_prompt, ticker=ticker)
        if not reply:
            continue

        result = None
        if use_oauth and server:
            # Use X OAuth API for reply
            valid_token = await server.get_valid_x_token(agent)
            if valid_token:
                result = await XOAuthAPI.post_reply(valid_token, tweet_id, reply)
        elif twitter_cookie:
            # Fallback to legacy TwitterAPI.io
            result = await TwitterAPI.post_reply(twitter_cookie, tweet_id, reply)

        if result:
            if agent_id not in agent_runner_state["answered_mentions"]:
                agent_runner_state["answered_mentions"][agent_id] = set()
            agent_runner_state["answered_mentions"][agent_id].add(tweet_id)
            answered_count += 1
        else:
            # Only try re-login for legacy cookie method
            if not use_oauth and twitter_cookie:
                new_cookie = await try_agent_relogin(agent)
                if new_cookie:
                    agent["twitter_cookie"] = new_cookie
                    running_agents[agent_id]["twitter_cookie"] = new_cookie
                    save_agents()
        await asyncio.sleep(5)
    agent_runner_state["mention_start_times"][agent_id] = int(datetime.now().timestamp())
    return answered_count

async def agent_runner_loop():
    logger.info("=" * 50)
    logger.info("AGENT RUNNER STARTED (integrated)")
    logger.info(f"Post interval: {POST_INTERVAL_HOURS} hours")
    logger.info(f"Mention check interval: {MENTION_CHECK_INTERVAL} seconds")
    logger.info("=" * 50)
    scanner = PolymarketScanner()
    agent_runner_state["answered_mentions"] = load_answered_mentions()
    while True:
        try:
            if not running_agents:
                await asyncio.sleep(60)
                continue
            current_time = datetime.now().timestamp()
            should_post = (current_time - agent_runner_state["last_post_time"]) >= (POST_INTERVAL_HOURS * 3600)
            server = get_server_instance()
            for agent_id, agent in list(running_agents.items()):
                if agent.get("status") != "running":
                    continue
                try:
                    mentions_answered = await process_agent_mentions(agent_id, agent, server)
                    if mentions_answered > 0:
                        save_answered_mentions(agent_runner_state["answered_mentions"])
                    if should_post:
                        await process_agent_posting(agent_id, agent, scanner, server)
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error(f"[{agent_id}] Error: {e}")
            if should_post:
                agent_runner_state["last_post_time"] = current_time
        except Exception as e:
            logger.error(f"[AgentRunner] Main loop error: {e}")
        await asyncio.sleep(MENTION_CHECK_INTERVAL)

async def start_agent(launch_id: str, launch: Dict[str, Any], credentials: Dict[str, str] = None, oauth_tokens: Dict[str, Any] = None):
    agent_id = launch_id
    running_agents[agent_id] = {
        "agent_id": agent_id,  # Store agent_id for token refresh
        "launch_id": launch_id,
        "token_ticker": launch["token_ticker"],
        "token_mint": launch.get("token_mint"),
        "twitter_username": launch.get("twitter_username"),
        "profile_link": launch.get("profile_link"),
        "twitter_cookie": launch.get("twitter_cookie"),
        "agent_niche": launch.get("agent_niche", "general"),
        "custom_prompt": launch.get("custom_prompt", ""),
        "description": launch.get("description", ""),
        "status": "running",
        "started_at": time.time()
    }
    if credentials:
        running_agents[agent_id].update({
            "twitter_username_cred": credentials.get("username"),
            "twitter_email": credentials.get("email"),
            "twitter_password": credentials.get("password"),
            "twitter_totp_secret": credentials.get("totp_secret")
        })
    # Add OAuth tokens if provided
    if oauth_tokens:
        running_agents[agent_id].update({
            "x_access_token": oauth_tokens.get("access_token"),
            "x_refresh_token": oauth_tokens.get("refresh_token"),
            "x_token_expires_at": time.time() + oauth_tokens.get("expires_in", 7200),
            "x_user_id": oauth_tokens.get("user_id")
        })
        logger.info(f"Agent {agent_id} using X OAuth tokens")
    save_agents()
    logger.info(f"Agent {agent_id} started for @{launch.get('twitter_username')}")

    # First post on agent launch - check for OAuth or cookie
    if running_agents[agent_id].get("x_access_token") or running_agents[agent_id].get("twitter_cookie"):
        asyncio.create_task(first_agent_post(agent_id))

    return agent_id

async def first_agent_post(agent_id: str):
    """Make first post when agent is launched"""
    await asyncio.sleep(5)  # Small delay for initialization
    if agent_id not in running_agents:
        return
    agent = running_agents[agent_id]
    scanner = PolymarketScanner()
    server = get_server_instance()
    logger.info(f"[{agent_id}] Making first post on launch...")
    # Set last_post to 0 to bypass cooldown for first post
    if "last_agent_post" not in agent_runner_state:
        agent_runner_state["last_agent_post"] = {}
    agent_runner_state["last_agent_post"][agent_id] = 0
    await process_agent_posting(agent_id, agent, scanner, server)

# =============================================================================
# LAUNCHPAD HELPERS
# =============================================================================

async def check_polyd_balance(wallet_address: str) -> float:
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenAccountsByOwner",
                "params": [
                    wallet_address,
                    {"mint": POLYD_MINT},
                    {"encoding": "jsonParsed"}
                ]
            }
            async with session.post(HELIUS_RPC, json=payload) as resp:
                data = await resp.json()
                if "result" in data and data["result"]["value"]:
                    token_account = data["result"]["value"][0]
                    return float(token_account["account"]["data"]["parsed"]["info"]["tokenAmount"]["uiAmount"])
                return 0
    except Exception as e:
        logger.error(f"Error checking POLYD balance: {e}")
        return 0

async def upload_metadata_to_ipfs(name: str, symbol: str, description: str, image_data: bytes, image_filename: str,
                                   website: str = None, twitter: str = None, telegram: str = None) -> Optional[str]:
    try:
        form_data = aiohttp.FormData()
        form_data.add_field('file', image_data, filename=image_filename, content_type='image/png')
        form_data.add_field('name', name)
        form_data.add_field('symbol', symbol)
        form_data.add_field('description', description)
        form_data.add_field('showName', 'true')
        if website and website.strip():
            form_data.add_field('website', website.strip())
        if twitter and twitter.strip():
            tw = twitter.strip()
            if not tw.startswith('http'):
                tw = f"https://x.com/{tw.replace('@', '')}"
            form_data.add_field('twitter', tw)
        if telegram and telegram.strip():
            tg = telegram.strip()
            if not tg.startswith('http'):
                tg = f"https://t.me/{tg.replace('@', '')}"
            form_data.add_field('telegram', tg)
        async with aiohttp.ClientSession() as session:
            async with session.post(PUMPFUN_IPFS, data=form_data) as resp:
                if resp.status != 200:
                    return None
                result = await resp.json()
                return result.get("metadataUri")
    except Exception as e:
        logger.error(f"Error uploading to IPFS: {e}")
        return None

async def create_token_on_pumpfun(name: str, symbol: str, description: str, image_data: bytes,
                                   website: str = None, twitter: str = None, telegram: str = None,
                                   dev_buy_sol: float = 0) -> Optional[Dict[str, str]]:
    try:
        from solders.keypair import Keypair
        from solders.transaction import VersionedTransaction
        from solders.commitment_config import CommitmentLevel
        from solders.rpc.requests import SendVersionedTransaction
        from solders.rpc.config import RpcSendTransactionConfig

        if not LAUNCHPAD_WALLET_PRIVATE_KEY:
            logger.error("LAUNCHPAD_WALLET_PRIVATE_KEY not set")
            return None

        signer_keypair = Keypair.from_base58_string(LAUNCHPAD_WALLET_PRIVATE_KEY)
        mint_keypair = Keypair()

        metadata_uri = await upload_metadata_to_ipfs(
            name=name, symbol=symbol, description=description, image_data=image_data,
            image_filename=f"{symbol.lower()}.png", website=website, twitter=twitter, telegram=telegram
        )
        if not metadata_uri:
            return None

        create_payload = {
            "publicKey": str(signer_keypair.pubkey()),
            "action": "create",
            "tokenMetadata": {"name": name, "symbol": symbol, "uri": metadata_uri},
            "mint": str(mint_keypair.pubkey()),
            "denominatedInSol": "true",
            "amount": dev_buy_sol,
            "slippage": 10,
            "priorityFee": 0.0005,
            "pool": "pump"
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(PUMPPORTAL_API, headers={"Content-Type": "application/json"}, json=create_payload) as resp:
                if resp.status != 200:
                    return None
                tx_bytes = await resp.read()

        tx = VersionedTransaction.from_bytes(tx_bytes)
        tx = VersionedTransaction(tx.message, [mint_keypair, signer_keypair])
        commitment = CommitmentLevel.Confirmed
        config = RpcSendTransactionConfig(preflight_commitment=commitment)
        tx_payload = SendVersionedTransaction(tx, config)

        async with aiohttp.ClientSession() as session:
            async with session.post(HELIUS_RPC, headers={"Content-Type": "application/json"}, data=tx_payload.to_json()) as resp:
                result = await resp.json()
                if "error" in result:
                    return None
                tx_signature = result.get("result")

        return {"token_mint": str(mint_keypair.pubkey()), "tx_signature": tx_signature, "signer_keypair": signer_keypair}
    except Exception as e:
        logger.error(f"Error creating token: {e}")
        return None


async def transfer_spl_tokens(token_mint: str, recipient_wallet: str, signer_keypair) -> Optional[str]:
    """Transfer all tokens from launchpad wallet to user wallet using spl-token-cli style transfer"""
    try:
        from solders.pubkey import Pubkey
        from solders.system_program import ID as SYSTEM_PROGRAM_ID
        from solders.transaction import Transaction
        from solders.instruction import Instruction, AccountMeta
        from solders.message import Message
        from solders.hash import Hash

        # Both token programs - standard and Token-2022
        TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
        TOKEN_2022_PROGRAM_ID = Pubkey.from_string("TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb")
        ASSOCIATED_TOKEN_PROGRAM_ID = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")

        mint_pubkey = Pubkey.from_string(token_mint)
        recipient_pubkey = Pubkey.from_string(recipient_wallet)
        signer_pubkey = signer_keypair.pubkey()

        # First, detect which token program this mint uses
        token_program_id = TOKEN_PROGRAM_ID  # default
        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getAccountInfo",
                "params": [str(mint_pubkey), {"encoding": "jsonParsed"}]
            }
            async with session.post(HELIUS_RPC, json=payload) as resp:
                result = await resp.json()
                mint_info = result.get("result", {}).get("value")
                if mint_info:
                    owner = mint_info.get("owner", "")
                    if owner == str(TOKEN_2022_PROGRAM_ID):
                        token_program_id = TOKEN_2022_PROGRAM_ID
                        logger.info(f"Detected Token-2022 program for mint {token_mint}")
                    else:
                        logger.info(f"Using standard Token program for mint {token_mint}")

        # Find token account for signer with this mint
        source_ata = None
        amount = 0
        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenAccountsByOwner",
                "params": [str(signer_pubkey), {"mint": str(mint_pubkey)}, {"encoding": "jsonParsed"}]
            }
            async with session.post(HELIUS_RPC, json=payload) as resp:
                result = await resp.json()
                accounts = result.get("result", {}).get("value", [])

                if not accounts:
                    logger.error(f"No token accounts found for {signer_pubkey} with mint {mint_pubkey}")
                    return None

                for acc in accounts:
                    acc_amount = int(acc["account"]["data"]["parsed"]["info"]["tokenAmount"]["amount"])
                    if acc_amount > 0:
                        source_ata = Pubkey.from_string(acc["pubkey"])
                        amount = acc_amount
                        logger.info(f"Found source token account {source_ata} with {amount} tokens")
                        break

                if not source_ata or amount <= 0:
                    logger.error(f"No tokens found in any account for {signer_pubkey}")
                    return None

        # Calculate destination ATA (uses detected token program)
        def get_ata(owner: Pubkey, mint: Pubkey, token_prog: Pubkey) -> Pubkey:
            seeds = [bytes(owner), bytes(token_prog), bytes(mint)]
            ata, _ = Pubkey.find_program_address(seeds, ASSOCIATED_TOKEN_PROGRAM_ID)
            return ata

        dest_ata = get_ata(recipient_pubkey, mint_pubkey, token_program_id)
        logger.info(f"Destination ATA: {dest_ata}")

        # Check if destination ATA exists
        dest_ata_exists = False
        async with aiohttp.ClientSession() as session:
            payload = {"jsonrpc": "2.0", "id": 1, "method": "getAccountInfo", "params": [str(dest_ata), {"encoding": "base64"}]}
            async with session.post(HELIUS_RPC, json=payload) as resp:
                result = await resp.json()
                if result.get("result", {}).get("value"):
                    dest_ata_exists = True
                    logger.info("Destination ATA exists")

        # Get recent blockhash
        async with aiohttp.ClientSession() as session:
            payload = {"jsonrpc": "2.0", "id": 1, "method": "getLatestBlockhash"}
            async with session.post(HELIUS_RPC, json=payload) as resp:
                result = await resp.json()
                blockhash = Hash.from_string(result["result"]["value"]["blockhash"])

        instructions = []

        # Create ATA if needed (use idempotent create)
        if not dest_ata_exists:
            # CreateIdempotent instruction (instruction discriminator = 1)
            create_ata_ix = Instruction(
                program_id=ASSOCIATED_TOKEN_PROGRAM_ID,
                accounts=[
                    AccountMeta(signer_pubkey, is_signer=True, is_writable=True),  # payer
                    AccountMeta(dest_ata, is_signer=False, is_writable=True),  # ata
                    AccountMeta(recipient_pubkey, is_signer=False, is_writable=False),  # owner
                    AccountMeta(mint_pubkey, is_signer=False, is_writable=False),  # mint
                    AccountMeta(SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),
                    AccountMeta(token_program_id, is_signer=False, is_writable=False),  # Use detected program
                ],
                data=bytes([1])  # CreateIdempotent instruction
            )
            instructions.append(create_ata_ix)
            logger.info(f"Added CreateIdempotent ATA instruction with token program {token_program_id}")

        # Transfer instruction
        import struct
        transfer_data = bytes([3]) + struct.pack('<Q', amount)
        transfer_ix = Instruction(
            program_id=token_program_id,  # Use detected program
            accounts=[
                AccountMeta(source_ata, is_signer=False, is_writable=True),
                AccountMeta(dest_ata, is_signer=False, is_writable=True),
                AccountMeta(signer_pubkey, is_signer=True, is_writable=False),
            ],
            data=transfer_data
        )
        instructions.append(transfer_ix)

        logger.info(f"Transferring {amount} tokens from {source_ata} to {dest_ata}")

        # Build and sign transaction
        message = Message.new_with_blockhash(instructions, signer_pubkey, blockhash)
        tx = Transaction.new_unsigned(message)
        tx.sign([signer_keypair], blockhash)

        # Send transaction
        import base64
        tx_bytes = bytes(tx)
        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "sendTransaction",
                "params": [base64.b64encode(tx_bytes).decode(), {"encoding": "base64"}]
            }
            async with session.post(HELIUS_RPC, json=payload) as resp:
                result = await resp.json()
                if "error" in result:
                    logger.error(f"Transfer failed: {result['error']}")
                    return None
                tx_sig = result.get("result")
                logger.info(f"Token transfer successful: {tx_sig}")
                return tx_sig

    except Exception as e:
        logger.error(f"Token transfer error: {e}")
        import traceback
        traceback.print_exc()
        return None


class APIServer:
    def __init__(self, host: str = "0.0.0.0", port: int = None):
        import os
        self.host = host
        self.port = port or int(os.environ.get("PORT", 8765))
        self.app = web.Application()
        # Cache for user trade counts (5 min TTL)
        self.user_trade_cache = TTLCache(ttl_seconds=300, max_size=10000)
        # Cache for token balance checks (5 min TTL)
        self.token_balance_cache = TTLCache(ttl_seconds=300, max_size=1000)
        # Active sessions (wallet -> session data)
        self.sessions = {}
        self._load_sessions()
        # Semaphore to limit concurrent API requests
        self.trade_count_semaphore = asyncio.Semaphore(20)
        self.setup_routes()

    def _load_sessions(self):
        """Load sessions from file"""
        try:
            if Path(SESSIONS_FILE).exists():
                with open(SESSIONS_FILE, 'r') as f:
                    data = json.load(f)
                    # Filter expired sessions
                    now = datetime.now().timestamp()
                    self.sessions = {
                        k: v for k, v in data.items()
                        if v.get('expires', 0) > now
                    }
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
            self.sessions = {}

    def _save_sessions(self):
        """Save sessions to file"""
        try:
            with open(SESSIONS_FILE, 'w') as f:
                json.dump(self.sessions, f)
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")

    def setup_routes(self):
        # API routes
        self.app.router.add_get("/api/watchlist/{user_id}", self.get_watchlist)
        self.app.router.add_post("/api/watchlist/{user_id}", self.update_watchlist)
        self.app.router.add_get("/api/events", self.get_events)
        self.app.router.add_get("/api/new-markets", self.get_new_markets)
        self.app.router.add_get("/api/arbitrage", self.get_arbitrage)
        self.app.router.add_get("/api/alerts", self.get_alerts)
        self.app.router.add_get("/api/context", self.get_market_context)
        self.app.router.add_get("/api/whales", self.get_whale_trades)
        self.app.router.add_options("/{path:.*}", self.handle_options)

        # Wallet authentication routes
        self.app.router.add_post("/api/wallet/verify", self.verify_wallet)
        self.app.router.add_get("/api/wallet/check", self.check_wallet_session)
        self.app.router.add_post("/api/wallet/disconnect", self.disconnect_wallet)

        # Launchpad routes
        self.app.router.add_get("/api/launchpad/config", self.launchpad_config)
        self.app.router.add_get("/api/launchpad/wallet", self.launchpad_wallet)
        self.app.router.add_get("/api/launchpad/check-balance", self.launchpad_check_balance)
        self.app.router.add_post("/api/launchpad/twitter-login", self.launchpad_twitter_login)
        self.app.router.add_post("/api/launchpad/check-eligibility", self.launchpad_check_eligibility)
        self.app.router.add_post("/api/launchpad/submit", self.launchpad_submit)
        self.app.router.add_post("/api/launchpad/confirm/{launch_id}", self.launchpad_confirm)
        self.app.router.add_get("/api/launchpad/status/{launch_id}", self.launchpad_status)
        self.app.router.add_get("/api/launchpad/launches", self.launchpad_list)
        self.app.router.add_get("/api/launchpad/stats", self.launchpad_stats)
        self.app.router.add_get("/api/launchpad/agents", self.launchpad_agents)
        self.app.router.add_post("/api/launchpad/agents/{agent_id}/refresh", self.launchpad_agent_refresh)
        self.app.router.add_post("/api/launchpad/agents/{agent_id}/post", self.launchpad_agent_trigger_post)

        # X OAuth 2.0
        self.app.router.add_get("/api/x/auth", self.x_oauth_start)
        self.app.router.add_get("/api/x/callback", self.x_oauth_callback)
        self.app.router.add_post("/api/x/tweet", self.x_post_tweet)

        # Project launchpad (token only, no agent)
        self.app.router.add_post("/api/projects/launch", self.projects_launch)
        self.app.router.add_post("/api/projects/confirm/{launch_id}", self.projects_confirm)
        self.app.router.add_get("/api/projects/list", self.projects_list)

        # Static file serving (only if polyweb folder exists - not needed on Railway)
        static_dir = Path(__file__).parent / 'polyweb'
        if static_dir.exists():
            # Page routes - serve index.html for each directory
            self.app.router.add_get('/', lambda r: web.FileResponse(static_dir / 'index.html'))
            self.app.router.add_get('/markets', lambda r: web.FileResponse(static_dir / 'markets' / 'index.html'))
            self.app.router.add_get('/markets/', lambda r: web.FileResponse(static_dir / 'markets' / 'index.html'))
            self.app.router.add_get('/whales', lambda r: web.FileResponse(static_dir / 'whales' / 'index.html'))
            self.app.router.add_get('/whales/', lambda r: web.FileResponse(static_dir / 'whales' / 'index.html'))
            self.app.router.add_get('/arbitrage', lambda r: web.FileResponse(static_dir / 'arbitrage' / 'index.html'))
            self.app.router.add_get('/arbitrage/', lambda r: web.FileResponse(static_dir / 'arbitrage' / 'index.html'))
            self.app.router.add_get('/roadmap', lambda r: web.FileResponse(static_dir / 'roadmap' / 'index.html'))
            self.app.router.add_get('/roadmap/', lambda r: web.FileResponse(static_dir / 'roadmap' / 'index.html'))
            self.app.router.add_get('/alerts', lambda r: web.FileResponse(static_dir / 'alerts' / 'index.html'))
            self.app.router.add_get('/alerts/', lambda r: web.FileResponse(static_dir / 'alerts' / 'index.html'))
            self.app.router.add_get('/launchpad', lambda r: web.FileResponse(static_dir / 'launchpad' / 'index.html'))
            self.app.router.add_get('/launchpad/', lambda r: web.FileResponse(static_dir / 'launchpad' / 'index.html'))
            self.app.router.add_get('/launchpad/projects', lambda r: web.FileResponse(static_dir / 'launchpad' / 'projects' / 'index.html'))
            self.app.router.add_get('/launchpad/projects/', lambda r: web.FileResponse(static_dir / 'launchpad' / 'projects' / 'index.html'))

            # Static assets
            self.app.router.add_static('/css', static_dir / 'css')
            self.app.router.add_static('/js', static_dir / 'js')
            self.app.router.add_static('/images', static_dir / 'images')
            self.app.router.add_static('/fonts', static_dir / 'fonts')

            # Root level static files (stylesheet, manifest, etc)
            self.app.router.add_get('/stylesheet_0.css', lambda r: web.FileResponse(static_dir / 'stylesheet_0.css'))
            self.app.router.add_get('/manifest.json', lambda r: web.FileResponse(static_dir / 'manifest.json'))
        else:
            # API-only mode - add a simple root endpoint
            self.app.router.add_get('/', lambda r: web.json_response({"status": "ok", "service": "polydictions-api"}))

        # Add CORS middleware
        self.app.middlewares.append(self.cors_middleware)

    @web.middleware
    async def cors_middleware(self, request, handler):
        # Get origin from request
        origin = request.headers.get('Origin', '')
        
        # Handle preflight OPTIONS requests
        if request.method == 'OPTIONS':
            response = web.Response(status=200)
        else:
            response = await handler(request)

        # Check if origin is allowed
        if origin in ALLOWED_ORIGINS or DEV_MODE:
            response.headers['Access-Control-Allow-Origin'] = origin if origin else '*'
        elif not origin:
            # No origin header (same-origin request or server-to-server)
            response.headers['Access-Control-Allow-Origin'] = ALLOWED_ORIGINS[0] if ALLOWED_ORIGINS else '*'
        # If origin not allowed, don't add CORS headers (browser will block)
        
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Telegram-User-Id, X-Wallet-Address, X-Session-Token'
        response.headers['Access-Control-Max-Age'] = '86400'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response

    async def handle_options(self, request):
        """Handle CORS preflight requests"""
        origin = request.headers.get('Origin', '')
        response = web.Response(status=200)
        
        if origin in ALLOWED_ORIGINS or DEV_MODE:
            response.headers['Access-Control-Allow-Origin'] = origin if origin else '*'
        
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Telegram-User-Id, X-Wallet-Address, X-Session-Token'
        response.headers['Access-Control-Max-Age'] = '86400'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response

    async def get_watchlist(self, request):
        """Get user's watchlist"""
        user_id = request.match_info.get('user_id')

        try:
            if Path(WATCHLIST_FILE).exists():
                with open(WATCHLIST_FILE, 'r') as f:
                    data = json.load(f)
                    user_watchlist = data.get(str(user_id), [])
                    return web.json_response({
                        "success": True,
                        "watchlist": user_watchlist
                    })
            return web.json_response({"success": True, "watchlist": []})
        except Exception as e:
            logger.error(f"Error getting watchlist: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def update_watchlist(self, request):
        """Update user's watchlist from extension"""
        user_id = request.match_info.get('user_id')

        try:
            body = await request.json()
            slugs = body.get('slugs', [])

            # Load existing data
            data = {}
            if Path(WATCHLIST_FILE).exists():
                with open(WATCHLIST_FILE, 'r') as f:
                    data = json.load(f)

            # Update user's watchlist
            data[str(user_id)] = slugs

            # Save
            with open(WATCHLIST_FILE, 'w') as f:
                json.dump(data, f, indent=2)

            return web.json_response({"success": True})
        except Exception as e:
            logger.error(f"Error updating watchlist: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    def is_updown_market(self, event):
        """Filter out crypto up/down spam markets"""
        title = (event.get('title') or '').lower()
        slug = (event.get('slug') or '').lower()

        # Crypto up/down markets (5-min, hourly, daily)
        if 'up or down' in title or 'up-or-down' in slug:
            return True
        if any(x in slug for x in ['-updown-', '-5m-', '-1h-']):
            return True

        return False

    async def get_events(self, request):
        """Proxy to Polymarket API with Builders key - fetches events sequentially"""
        import aiohttp

        limit = int(request.query.get('limit', '5000'))
        sort = request.query.get('sort', 'createdAt')  # createdAt, volume, startDate

        try:
            headers = {"Authorization": f"Bearer {POLYMARKET_BUILDERS_KEY}"}
            seen_ids = set()
            unique_events = []
            batch_size = 500
            offset = 0

            async with aiohttp.ClientSession() as session:
                while len(unique_events) < limit and offset < 10000:
                    url = f"https://gamma-api.polymarket.com/events?limit={batch_size}&offset={offset}&order={sort}&ascending=false"
                    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status != 200:
                            break
                        batch = await resp.json()
                        if not batch:
                            break

                        new_count = 0
                        for event in batch:
                            event_id = event.get('id')
                            if event_id and event_id not in seen_ids:
                                if not self.is_updown_market(event):
                                    seen_ids.add(event_id)
                                    unique_events.append(event)
                                    new_count += 1

                        if new_count == 0:
                            break
                        offset += batch_size

            logger.info(f"Fetched {len(unique_events)} unique events")
            return web.json_response(unique_events[:limit])
        except Exception as e:
            logger.error(f"Error fetching events: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def get_new_markets(self, request):
        """Get recently posted new markets (same as Telegram channel)"""
        posted_events_file = Path("posted_events.json")

        try:
            if posted_events_file.exists():
                with open(posted_events_file, 'r') as f:
                    data = json.load(f)
                    events = data.get('events', [])
                    return web.json_response({
                        "success": True,
                        "events": events
                    })
            return web.json_response({"success": True, "events": []})
        except Exception as e:
            logger.error(f"Error getting new markets: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def get_alerts(self, request):
        """Get recent market alerts for the alerts page - only last 3 days"""
        import aiohttp
        from datetime import datetime, timedelta, timezone

        limit = int(request.query.get('limit', '100'))
        days = int(request.query.get('days', '3'))

        try:
            headers = {"Authorization": f"Bearer {POLYMARKET_BUILDERS_KEY}"}
            alerts = []
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

            # Fetch recent events from API
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://gamma-api.polymarket.com/events?limit=500&active=true&closed=false&order=createdAt&ascending=false",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        api_events = await resp.json()

                        # Filter by creation date (last N days) and exclude up/down markets
                        for event in api_events:
                            # Skip up/down markets
                            if self.is_updown_market(event):
                                continue

                            created_at = event.get('createdAt') or event.get('startDate')
                            if created_at:
                                try:
                                    # Parse ISO date
                                    event_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                                    if event_date >= cutoff_date:
                                        event['posted_at'] = created_at
                                        alerts.append(event)
                                except:
                                    pass

            # Sort by creation date (newest first)
            alerts.sort(key=lambda x: x.get('createdAt', ''), reverse=True)

            return web.json_response({
                "success": True,
                "alerts": alerts[:limit]
            })
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def get_arbitrage(self, request):
        """Get arbitrage opportunities from Dome API - REQUIRES VALID SESSION"""

        # Check for valid session (server-side paywall)
        session_check = self._check_session_from_request(request)

        if not session_check['valid']:
            # Return 401 with limited preview data (just count, no actual opportunities)
            return web.json_response({
                "success": False,
                "error": "authentication_required",
                "message": "Connect wallet with sufficient token balance to access arbitrage data",
                "preview": {
                    "total_available": "15+",  # Teaser
                    "best_spread": "8%+",
                    "sports": ["NFL", "NBA", "NHL", "MLB"]
                },
                "required_tokens": MIN_TOKEN_BALANCE
            }, status=401)

        try:
            from dome_tracker import DomeClient
            client = DomeClient()

            sport = request.query.get('sport', 'all')
            min_diff = float(request.query.get('min_diff', '5.0'))

            # Handle general (non-sports) markets
            if sport == 'general':
                opportunities = await client.find_general_arbitrage(
                    min_diff=min_diff,
                    min_similarity=0.5
                )
                return web.json_response({
                    "success": True,
                    "opportunities": opportunities,
                    "total": len(opportunities),
                    "authenticated": True,
                    "wallet": session_check['wallet'][:8] + '...'
                })

            # Convert sport parameter to list format expected by find_sports_arbitrage
            if sport == 'all':
                sports_list = ['nfl', 'nba', 'nhl', 'mlb', 'cfb', 'cbb']  # All sports
            else:
                sports_list = [sport]  # Single sport as list

            # Call with sports as list parameter
            opportunities = await client.find_sports_arbitrage(
                sports=sports_list,
                days_ahead=7,
                min_diff=min_diff
            )

            return web.json_response({
                "success": True,
                "opportunities": opportunities,
                "total": len(opportunities),
                "authenticated": True,
                "wallet": session_check['wallet'][:8] + '...'
            })
        except Exception as e:
            logger.error(f"Error getting arbitrage: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def get_market_context(self, request):
        """Get AI-generated Market Context from Polymarket API"""
        import aiohttp
        import ssl

        slug = request.query.get('slug', '')
        if not slug:
            return web.json_response({"success": False, "error": "No slug provided"}, status=400)

        # Use the Grok event-summary API (same as bot.py)
        url = f"https://polymarket.com/api/grok/event-summary?prompt={slug}"

        # SSL context that doesn't verify certificates (fixes Windows SSL issues)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        try:
            timeout = aiohttp.ClientTimeout(total=120)
            connector = aiohttp.TCPConnector(ssl=ssl_context)

            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                logger.info(f"Fetching Market Context for: {slug}")
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
                        # Remove sources block if present
                        if '__SOURCES__' in text:
                            text = text.split('__SOURCES__')[0].strip()
                        if text and len(text) > 50:
                            return web.json_response({
                                "success": True,
                                "context": text
                            })
                        else:
                            return web.json_response({
                                "success": False,
                                "error": "Response too short or empty"
                            }, status=500)
                    else:
                        error_text = await response.text()
                        logger.error(f"Market Context API error: {response.status} - {error_text}")
                        return web.json_response({
                            "success": False,
                            "error": f"API returned status {response.status}"
                        }, status=response.status)

        except asyncio.TimeoutError:
            logger.error(f"Market Context request timed out for {slug}")
            return web.json_response({
                "success": False,
                "error": "Request timed out (may take up to 2 minutes)"
            }, status=504)
        except Exception as e:
            logger.error(f"Error fetching Market Context: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def _fetch_user_trade_count(self, user_address, session):
        """Internal: fetch trade count from API"""
        try:
            url = f"https://data-api.polymarket.com/activity?user={user_address}&limit=100"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    activities = await resp.json()
                    trade_count = sum(1 for a in activities if a.get('type') == 'TRADE')
                    return trade_count
                logger.warning(f"Activity API returned {resp.status} for {user_address}")
                return -1
        except Exception as e:
            logger.warning(f"Error checking user trades for {user_address}: {e}")
            return -1

    async def get_user_trade_count_cached(self, user_address, session):
        """Get user trade count with caching and rate limiting"""
        # Check cache first
        cached = self.user_trade_cache.get(user_address)
        if cached is not None:
            return cached

        # Fetch with semaphore to limit concurrent requests
        async with self.trade_count_semaphore:
            # Double-check cache (another request might have populated it)
            cached = self.user_trade_cache.get(user_address)
            if cached is not None:
                return cached

            count = await self._fetch_user_trade_count(user_address, session)
            self.user_trade_cache.set(user_address, count)
            return count

    async def get_whale_trades(self, request):
        """Get whale trades from Polymarket Data API - real trades over $5000"""
        import aiohttp

        min_amount = float(request.query.get('min', '5000'))
        limit = int(request.query.get('limit', '100'))
        first_trade_only = request.query.get('first_trade_only', 'false').lower() == 'true'

        try:
            async with aiohttp.ClientSession() as session:
                # Use official Polymarket Data API with CASH filter for whale trades
                url = f"https://data-api.polymarket.com/trades?limit={min(limit * 2, 500)}&filterType=CASH&filterAmount={int(min_amount)}&takerOnly=true"

                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"Polymarket Data API error: {resp.status} - {error_text}")
                        return web.json_response({
                            "success": False,
                            "error": f"Polymarket API returned {resp.status}"
                        }, status=resp.status)

                    trades_data = await resp.json()

                    # Filter spam markets first
                    filtered_trades = []
                    for trade in trades_data:
                        title = (trade.get('title') or '').lower()
                        slug = (trade.get('slug') or '').lower()
                        if 'up or down' in title or 'updown' in slug:
                            continue
                        filtered_trades.append(trade)

                    # LAZY LOAD: Only fetch trade counts when insider mode is enabled
                    user_trade_counts = {}
                    if first_trade_only:
                        unique_users = list(set(t.get('proxyWallet', '') for t in filtered_trades if t.get('proxyWallet')))
                        if unique_users:
                            # Parallel fetch with caching and rate limiting
                            counts = await asyncio.gather(*[
                                self.get_user_trade_count_cached(addr, session)
                                for addr in unique_users
                            ])
                            user_trade_counts = dict(zip(unique_users, counts))

                    # Transform to our format
                    whale_trades = []
                    for trade in filtered_trades:
                        size = float(trade.get('size', 0))
                        price = float(trade.get('price', 0))
                        trade_value = size * price

                        user_address = trade.get('proxyWallet', '')
                        timestamp = int(trade.get('timestamp', 0))
                        timestamp_ms = timestamp * 1000 if timestamp < 10000000000 else timestamp

                        # Get trade count from cache/results (only available in insider mode)
                        trade_count = user_trade_counts.get(user_address) if user_address else None
                        is_new_account = (trade_count is not None and 0 < trade_count <= 5)

                        # If filter is enabled, only show confirmed new accounts (1-5 trades)
                        if first_trade_only:
                            if trade_count is None or trade_count < 0 or trade_count > 5:
                                continue

                        whale_trades.append({
                            'id': trade.get('transactionHash', ''),
                            'market': trade.get('title', 'Unknown Market'),
                            'slug': trade.get('slug', ''),
                            'event_slug': trade.get('eventSlug', ''),
                            'outcome': trade.get('outcome', 'Yes'),
                            'side': trade.get('side', 'BUY').lower(),
                            'amount': trade_value,
                            'size': size,
                            'price': round(price * 100, 1),  # Convert to cents (0-100)
                            'timestamp': timestamp_ms,
                            'user': user_address,
                            'user_short': user_address[:6] + '...' + user_address[-4:] if len(user_address) > 10 else user_address,
                            'name': trade.get('name') or trade.get('pseudonym') or '',
                            'profile_image': trade.get('profileImageOptimized') or trade.get('profileImage') or '',
                            'is_new_account': is_new_account,
                            'trade_count': trade_count
                        })

                    return web.json_response({
                        "success": True,
                        "trades": whale_trades[:limit],
                        "total": len(whale_trades),
                        "first_trade_filter": first_trade_only
                    })

        except Exception as e:
            logger.error(f"Error fetching whale trades: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    # ============ WALLET AUTHENTICATION ============

    def _check_session_from_request(self, request) -> dict:
        """
        Check if request has valid session.
        Returns dict with 'valid', 'wallet', 'balance' keys.
        Session can be passed via:
        - Query params: ?wallet=...&token=...
        - Headers: X-Wallet-Address, X-Session-Token
        - Dev bypass: X-Wallet-Address: dev-bypass
        """
        # Try headers first
        wallet = request.headers.get('X-Wallet-Address', '')
        token = request.headers.get('X-Session-Token', '')

        # Dev bypass for testing - ONLY in dev mode!
        if DEV_MODE and wallet == 'dev-bypass':
            logger.warning("⚠️ Dev bypass used - disable DEV_MODE in production!")
            return {'valid': True, 'wallet': 'dev-bypass', 'balance': 999999}

        # Fall back to query params
        if not wallet:
            wallet = request.query.get('wallet', '')
        if not token:
            token = request.query.get('token', '')

        if not wallet or not token:
            return {'valid': False, 'reason': 'Missing credentials'}

        session = self.sessions.get(wallet)
        if not session:
            return {'valid': False, 'reason': 'No session'}

        # Check expiry
        if datetime.now().timestamp() > session.get('expires', 0):
            del self.sessions[wallet]
            self._save_sessions()
            return {'valid': False, 'reason': 'Session expired'}

        # Check token
        if session.get('token') != token:
            return {'valid': False, 'reason': 'Invalid token'}

        return {
            'valid': True,
            'wallet': wallet,
            'balance': session.get('balance', 0)
        }

    async def _get_token_balance(self, wallet_address: str) -> float:
        """Get SPL token balance for a wallet from Solana RPC"""
        import aiohttp

        logger.info(f"🔍 Checking token balance for wallet: {wallet_address}")
        logger.info(f"🔍 Token mint: {REQUIRED_TOKEN_MINT}")

        # Check cache first
        cached = self.token_balance_cache.get(wallet_address)
        if cached is not None:
            logger.info(f"📦 Cached balance: {cached}")
            return cached

        for rpc_url in SOLANA_RPC_ENDPOINTS:
            try:
                logger.info(f"🌐 Trying RPC: {rpc_url}")
                # Use getTokenAccountsByOwner RPC method
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getTokenAccountsByOwner",
                    "params": [
                        wallet_address,
                        {"mint": REQUIRED_TOKEN_MINT},
                        {"encoding": "jsonParsed"}
                    ]
                }

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        rpc_url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as resp:
                        logger.info(f"📡 RPC response status: {resp.status}")
                        if resp.status == 200:
                            data = await resp.json()
                            logger.info(f"📦 RPC response: {data}")

                            if "error" in data:
                                logger.warning(f"RPC error from {rpc_url}: {data['error']}")
                                continue

                            result = data.get("result", {})
                            accounts = result.get("value", [])

                            if not accounts:
                                # No token account = 0 balance
                                logger.info(f"⚠️ No token accounts found for {wallet_address[:8]}...")
                                self.token_balance_cache.set(wallet_address, 0)
                                return 0

                            # Sum up balances from all token accounts
                            total_balance = 0
                            for account in accounts:
                                parsed = account.get("account", {}).get("data", {}).get("parsed", {})
                                info = parsed.get("info", {})
                                token_amount = info.get("tokenAmount", {})
                                ui_amount = token_amount.get("uiAmount", 0)
                                if ui_amount:
                                    total_balance += float(ui_amount)

                            logger.info(f"✅ Wallet {wallet_address[:8]}... has {total_balance} tokens")
                            self.token_balance_cache.set(wallet_address, total_balance)
                            return total_balance

            except Exception as e:
                logger.warning(f"RPC {rpc_url} failed: {e}")
                continue

        # All RPCs failed
        logger.error(f"❌ All Solana RPCs failed for {wallet_address}")
        return -1  # Error indicator

    def _generate_session_token(self, wallet_address: str) -> str:
        """Generate a secure session token"""
        data = f"{wallet_address}:{SESSION_SECRET}:{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()

    def _verify_signature(self, message: str, signature: str, wallet_address: str) -> bool:
        """
        Verify Solana ed25519 signature.
        """
        try:
            import base58
            import nacl.signing
            import nacl.exceptions
            
            # Decode wallet address (public key) from base58
            public_key_bytes = base58.b58decode(wallet_address)
            
            # Decode signature from base58 or hex
            try:
                signature_bytes = base58.b58decode(signature)
            except:
                signature_bytes = bytes.fromhex(signature)
            
            # Message should be UTF-8 encoded
            message_bytes = message.encode('utf-8')
            
            # Create verify key from public key
            verify_key = nacl.signing.VerifyKey(public_key_bytes)
            
            # Verify signature
            verify_key.verify(message_bytes, signature_bytes)
            return True
            
        except nacl.exceptions.BadSignature:
            logger.warning(f"Invalid signature for wallet {wallet_address[:8]}...")
            return False
        except ImportError:
            logger.error("nacl/base58 not installed. Run: pip install pynacl base58")
            # In dev mode, allow without verification
            if DEV_MODE:
                logger.warning("⚠️ Skipping signature verification in dev mode")
                return True
            return False
        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return False

    async def _get_dynamic_jwks(self):
        """Fetch and cache JWKS from Dynamic.xyz"""
        global _jwks_cache, _jwks_cache_time
        import aiohttp

        # Check cache
        if _jwks_cache and (time.time() - _jwks_cache_time) < JWKS_CACHE_TTL:
            return _jwks_cache

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(DYNAMIC_JWKS_URL, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        _jwks_cache = await resp.json()
                        _jwks_cache_time = time.time()
                        logger.info("Fetched JWKS from Dynamic.xyz")
                        return _jwks_cache
        except Exception as e:
            logger.error(f"Error fetching JWKS: {e}")

        return _jwks_cache  # Return cached even if stale

    async def _verify_dynamic_jwt(self, token: str) -> dict:
        """
        Verify JWT token from Dynamic.xyz with full signature verification.
        Returns decoded payload if valid, None otherwise.
        """
        try:
            from jwt import PyJWKClient
            from jwt.algorithms import RSAAlgorithm
            import json as json_module

            # Get JWKS
            jwks = await self._get_dynamic_jwks()
            if not jwks:
                logger.warning("No JWKS available for JWT verification")
                return None

            # Get the token header to find the key ID (kid)
            try:
                header = jwt.get_unverified_header(token)
                kid = header.get('kid')
                alg = header.get('alg', 'RS256')
            except Exception as e:
                logger.warning(f"Failed to get JWT header: {e}")
                return None

            # Find the matching key in JWKS
            signing_key = None
            for key in jwks.get('keys', []):
                if key.get('kid') == kid:
                    # Convert JWK to PEM format for verification
                    signing_key = RSAAlgorithm.from_jwk(json_module.dumps(key))
                    break

            if not signing_key:
                logger.warning(f"No matching key found for kid: {kid}")
                return None

            # Verify and decode the token with full signature verification
            decoded = jwt.decode(
                token,
                signing_key,
                algorithms=[alg],
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "require": ["exp", "iat"]
                }
            )

            # Check environment ID matches
            if decoded.get('environment_id') != DYNAMIC_ENV_ID:
                logger.warning(f"JWT environment_id mismatch: {decoded.get('environment_id')}")
                return None

            wallet_address = decoded.get('verified_credentials', [{}])[0].get('address', 'unknown')
            logger.info(f"Dynamic JWT verified (signature valid) for wallet: {wallet_address}")
            return decoded

        except jwt.exceptions.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.exceptions.InvalidSignatureError:
            logger.warning("JWT signature verification failed - possible forgery attempt")
            return None
        except jwt.exceptions.DecodeError as e:
            logger.warning(f"JWT decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"JWT verification error: {e}")
            return None

    async def verify_wallet(self, request):
        """
        Verify wallet ownership and token balance.
        Called after user connects wallet on frontend.
        Supports both Dynamic.xyz JWT auth and direct wallet verification.
        """
        try:
            body = await request.json()
            wallet_address = body.get('wallet')

            # Check for Dynamic.xyz JWT token in Authorization header
            auth_header = request.headers.get('Authorization', '')
            dynamic_jwt = None
            if auth_header.startswith('Bearer '):
                jwt_token = auth_header[7:]
                if jwt_token:
                    dynamic_jwt = await self._verify_dynamic_jwt(jwt_token)
                    if dynamic_jwt:
                        # Extract wallet from JWT verified credentials
                        verified_creds = dynamic_jwt.get('verified_credentials', [])
                        for cred in verified_creds:
                            if cred.get('chain') == 'solana' and cred.get('address'):
                                jwt_wallet = cred.get('address')
                                # Verify JWT wallet matches provided wallet
                                if wallet_address and jwt_wallet != wallet_address:
                                    logger.warning(f"JWT wallet mismatch: {jwt_wallet} vs {wallet_address}")
                                else:
                                    wallet_address = jwt_wallet
                                    logger.info(f"Using wallet from Dynamic JWT: {wallet_address}")
                                break

            if not wallet_address:
                return web.json_response({
                    "success": False,
                    "error": "No wallet address provided"
                }, status=400)

            # Validate wallet address format (Solana base58, 32-44 chars)
            if not (32 <= len(wallet_address) <= 44):
                return web.json_response({
                    "success": False,
                    "error": "Invalid wallet address format"
                }, status=400)

            # Get token balance from Solana
            balance = await self._get_token_balance(wallet_address)

            if balance < 0:
                return web.json_response({
                    "success": False,
                    "error": "Failed to check token balance. Please try again."
                }, status=503)

            # Check if balance meets minimum
            has_access = balance >= MIN_TOKEN_BALANCE

            if has_access:
                # Create session
                session_token = self._generate_session_token(wallet_address)
                expires = datetime.now().timestamp() + (SESSION_DURATION_HOURS * 3600)

                self.sessions[wallet_address] = {
                    "token": session_token,
                    "balance": balance,
                    "expires": expires,
                    "created": datetime.now().isoformat()
                }
                self._save_sessions()

                return web.json_response({
                    "success": True,
                    "access": True,
                    "balance": balance,
                    "required": MIN_TOKEN_BALANCE,
                    "session_token": session_token,
                    "expires_in": SESSION_DURATION_HOURS * 3600
                })
            else:
                return web.json_response({
                    "success": True,
                    "access": False,
                    "balance": balance,
                    "required": MIN_TOKEN_BALANCE,
                    "need_more": MIN_TOKEN_BALANCE - balance
                })

        except Exception as e:
            logger.error(f"Wallet verification error: {e}")
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)

    async def check_wallet_session(self, request):
        """Check if a wallet session is still valid"""
        try:
            wallet = request.query.get('wallet', '')
            token = request.query.get('token', '')

            if not wallet or not token:
                return web.json_response({
                    "valid": False,
                    "reason": "Missing wallet or token"
                })

            session = self.sessions.get(wallet)
            if not session:
                return web.json_response({
                    "valid": False,
                    "reason": "No session found"
                })

            # Check if session expired
            if datetime.now().timestamp() > session.get('expires', 0):
                del self.sessions[wallet]
                self._save_sessions()
                return web.json_response({
                    "valid": False,
                    "reason": "Session expired"
                })

            # Check token matches
            if session.get('token') != token:
                return web.json_response({
                    "valid": False,
                    "reason": "Invalid token"
                })

            return web.json_response({
                "valid": True,
                "balance": session.get('balance', 0),
                "expires": session.get('expires', 0)
            })

        except Exception as e:
            logger.error(f"Session check error: {e}")
            return web.json_response({
                "valid": False,
                "reason": str(e)
            })

    async def disconnect_wallet(self, request):
        """Disconnect wallet and invalidate session"""
        try:
            body = await request.json()
            wallet = body.get('wallet', '')

            if wallet and wallet in self.sessions:
                del self.sessions[wallet]
                self._save_sessions()

            return web.json_response({"success": True})

        except Exception as e:
            logger.error(f"Disconnect error: {e}")
            return web.json_response({"success": False, "error": str(e)})

    # =========================================================================
    # LAUNCHPAD ENDPOINTS
    # =========================================================================

    async def launchpad_config(self, request):
        """Get frontend configuration (public values only)"""
        return web.json_response({
            "helius_rpc": HELIUS_RPC,
            "polyd_mint": POLYD_MINT,
            "required_balance": REQUIRED_POLYD_BALANCE,
            "launch_fee": LAUNCH_FEE_SOL,
            "payment_address": LAUNCHPAD_WALLET_PUBLIC_KEY,
            "reown_project_id": os.getenv("REOWN_PROJECT_ID", "")
        })

    async def launchpad_wallet(self, request):
        """Get the payment wallet address"""
        return web.json_response({
            "address": LAUNCHPAD_WALLET_PUBLIC_KEY,
            "fee_sol": LAUNCH_FEE_SOL,
            "required_polyd": REQUIRED_POLYD_BALANCE
        })

    async def launchpad_check_balance(self, request):
        """Check $POLYD balance for a wallet"""
        wallet = request.query.get('wallet', '')
        if not wallet:
            return web.json_response({"error": "wallet parameter required"}, status=400)

        balance = await check_polyd_balance(wallet)
        eligible = balance >= REQUIRED_POLYD_BALANCE

        return web.json_response({
            "balance": balance,
            "required": REQUIRED_POLYD_BALANCE,
            "eligible": eligible,
            "missing": max(0, REQUIRED_POLYD_BALANCE - balance)
        })

    async def launchpad_twitter_login(self, request):
        """Login to Twitter and get login cookie"""
        try:
            body = await request.json()
            username = body.get('username', '')
            email = body.get('email', '')
            password = body.get('password', '')
            totp_secret = body.get('totp_secret', '')

            if not all([username, email, password]):
                return web.json_response({"success": False, "message": "Missing required fields"}, status=400)

            if not TWITTERAPI_KEY:
                return web.json_response({"success": False, "message": "TwitterAPI.io not configured"}, status=500)

            login_cookie = await TwitterAPI.login(username, email, password, totp_secret)

            if login_cookie:
                return web.json_response({
                    "success": True,
                    "message": "Twitter login successful",
                    "login_cookie": login_cookie,
                    "is_blue_verified": True
                })
            else:
                return web.json_response({"success": False, "message": "Twitter login failed"}, status=400)
        except Exception as e:
            logger.error(f"Twitter login error: {e}")
            return web.json_response({"success": False, "message": str(e)}, status=500)

    async def launchpad_check_eligibility(self, request):
        """Check if user is eligible to launch"""
        try:
            body = await request.json()
            user_wallet = body.get('user_wallet', '')
            if not user_wallet:
                return web.json_response({"eligible": False, "error": "No wallet provided"}, status=400)

            balance = await check_polyd_balance(user_wallet)
            eligible = balance >= REQUIRED_POLYD_BALANCE

            return web.json_response({
                "eligible": eligible,
                "balance": balance,
                "required": REQUIRED_POLYD_BALANCE,
                "missing": max(0, REQUIRED_POLYD_BALANCE - balance)
            })
        except Exception as e:
            logger.error(f"Eligibility check error: {e}")
            return web.json_response({"eligible": False, "error": str(e)}, status=500)

    async def launchpad_submit(self, request):
        """Submit a new agent + token launch request"""
        try:
            reader = await request.multipart()
            fields = {}
            image_data = None

            async for field in reader:
                if field.name == 'token_image':
                    image_data = await field.read()
                else:
                    fields[field.name] = await field.text()

            # Check if using OAuth or legacy credentials
            has_oauth = bool(fields.get('x_access_token') and fields.get('x_refresh_token'))
            has_credentials = bool(fields.get('twitter_cookie'))

            # Validate required fields - OAuth or credentials required
            required = ['token_ticker', 'agent_niche', 'description', 'profile_link',
                       'wallet_address', 'user_wallet', 'twitter_username']

            if not has_oauth and not has_credentials:
                return web.json_response({
                    "success": False,
                    "message": "Either X OAuth tokens or Twitter credentials required"
                }, status=400)

            for f in required:
                if f not in fields:
                    return web.json_response({"success": False, "message": f"Missing field: {f}"}, status=400)

            if not image_data:
                return web.json_response({"success": False, "message": "Missing token image"}, status=400)

            if len(image_data) > 5 * 1024 * 1024:
                return web.json_response({"success": False, "message": "Image too large (max 5MB)"}, status=400)

            # Check eligibility
            balance = await check_polyd_balance(fields['user_wallet'])
            if balance < REQUIRED_POLYD_BALANCE:
                return web.json_response({
                    "success": False,
                    "message": f"Insufficient $POLYD balance. Need {REQUIRED_POLYD_BALANCE:,}, have {balance:,.0f}"
                }, status=400)

            # Generate launch ID
            launch_id = hashlib.sha256(f"{fields['user_wallet']}{time.time()}".encode()).hexdigest()[:16]
            token_ticker = fields['token_ticker'].replace('$', '').upper()
            token_name = f"{token_ticker} Agent"

            # Parse dev buy amount (max 85 SOL)
            dev_buy_sol = min(float(fields.get('dev_buy_sol', 0) or 0), 85.0)

            launch_data = {
                "token_ticker": token_ticker,
                "token_name": token_name,
                "agent_niche": fields['agent_niche'],
                "custom_prompt": fields.get('custom_prompt', ''),
                "profile_link": fields['profile_link'],
                "wallet_address": fields['wallet_address'],
                "description": fields['description'],
                "website": fields.get('website'),
                "telegram": fields.get('telegram'),
                "twitter_username": fields['twitter_username'].replace('@', ''),
                "user_wallet": fields['user_wallet'],
                "image_data": image_data,
                "dev_buy_sol": dev_buy_sol,
                "status": "pending_payment",
                "created_at": time.time()
            }

            # Add OAuth tokens if provided
            if has_oauth:
                launch_data["x_oauth"] = {
                    "access_token": fields['x_access_token'],
                    "refresh_token": fields['x_refresh_token'],
                    "user_id": fields.get('x_user_id', ''),
                    "expires_in": int(fields.get('x_expires_in', 7200))
                }
                logger.info(f"Launch {launch_id} using X OAuth for @{fields['twitter_username']}")
            else:
                # Legacy credentials
                launch_data["twitter_cookie"] = fields.get('twitter_cookie', '')
                launch_data["twitter_credentials"] = {
                    "username": fields['twitter_username'].replace('@', ''),
                    "email": fields.get('twitter_email', ''),
                    "password": fields.get('twitter_password', ''),
                    "totp_secret": fields.get('twitter_totp_secret', '')
                }

            pending_launches[launch_id] = launch_data
            save_launches()

            return web.json_response({
                "success": True,
                "message": f"Launch submitted. Send {LAUNCH_FEE_SOL} SOL to complete.",
                "launch_id": launch_id,
                "payment_address": LAUNCHPAD_WALLET_PUBLIC_KEY,
                "payment_amount": LAUNCH_FEE_SOL
            })
        except Exception as e:
            logger.error(f"Launch submit error: {e}")
            return web.json_response({"success": False, "message": str(e)}, status=500)

    async def launchpad_confirm(self, request):
        """Confirm payment, create token, and start agent"""
        launch_id = request.match_info.get('launch_id')

        if launch_id not in pending_launches:
            return web.json_response({"success": False, "message": "Launch not found"}, status=404)

        launch = pending_launches[launch_id]

        if launch["status"] == "completed":
            return web.json_response({
                "success": True,
                "message": "Token already created",
                "launch_id": launch_id,
                "token_mint": launch.get("token_mint"),
                "tx_signature": launch.get("tx_signature")
            })

        if launch["status"] not in ["pending_payment", "failed"]:
            return web.json_response({"success": False, "message": f"Invalid status: {launch['status']}"}, status=400)

        launch["status"] = "creating"
        logger.info(f"Creating token: {launch['token_name']} (${launch['token_ticker']})")

        result = await create_token_on_pumpfun(
            name=launch["token_name"],
            symbol=launch["token_ticker"],
            description=launch["description"],
            image_data=launch["image_data"],
            website=launch.get("website"),
            twitter=launch.get("twitter_username"),
            telegram=launch.get("telegram"),
            dev_buy_sol=launch.get("dev_buy_sol", 0)
        )

        if result:
            launch["status"] = "completed"
            launch["token_mint"] = result["token_mint"]
            launch["tx_signature"] = result["tx_signature"]

            # Transfer tokens to user if dev_buy was used
            transfer_tx = None
            if launch.get("dev_buy_sol", 0) > 0 and result.get("signer_keypair"):
                logger.info(f"Transferring tokens to user wallet: {launch['user_wallet']}")
                # Wait for tx confirmation with retries
                for attempt in range(3):
                    await asyncio.sleep(5 + attempt * 5)  # 5s, 10s, 15s
                    transfer_tx = await transfer_spl_tokens(
                        token_mint=result["token_mint"],
                        recipient_wallet=launch["user_wallet"],
                        signer_keypair=result["signer_keypair"]
                    )
                    if transfer_tx:
                        launch["transfer_tx"] = transfer_tx
                        logger.info(f"Tokens transferred to user: {transfer_tx}")
                        break
                    logger.warning(f"Transfer attempt {attempt + 1} failed, retrying...")

            save_launches()

            # Start agent with OAuth or legacy credentials
            credentials = launch.get("twitter_credentials")
            oauth_tokens = launch.get("x_oauth")
            await start_agent(launch_id, launch, credentials=credentials, oauth_tokens=oauth_tokens)

            return web.json_response({
                "success": True,
                "message": "Token created and agent started!" + (" Tokens sent to your wallet." if transfer_tx else ""),
                "launch_id": launch_id,
                "token_mint": result["token_mint"],
                "tx_signature": result["tx_signature"],
                "transfer_tx": transfer_tx
            })
        else:
            launch["status"] = "failed"
            return web.json_response({"success": False, "message": "Failed to create token"}, status=500)

    async def launchpad_status(self, request):
        """Get status of a launch"""
        launch_id = request.match_info.get('launch_id')

        if launch_id not in pending_launches:
            return web.json_response({"success": False, "message": "Launch not found"}, status=404)

        launch = pending_launches[launch_id]
        return web.json_response({
            "launch_id": launch_id,
            "status": launch["status"],
            "token_name": launch["token_name"],
            "token_ticker": launch["token_ticker"],
            "twitter_username": launch.get("twitter_username"),
            "token_mint": launch.get("token_mint"),
            "tx_signature": launch.get("tx_signature")
        })

    async def launchpad_list(self, request):
        """List all launches"""
        return web.json_response({
            "total": len(pending_launches),
            "launches": [
                {
                    "launch_id": lid,
                    "status": l["status"],
                    "token_name": l["token_name"],
                    "token_ticker": l["token_ticker"],
                    "twitter_username": l.get("twitter_username"),
                    "created_at": l["created_at"],
                    "token_mint": l.get("token_mint")
                }
                for lid, l in pending_launches.items()
            ]
        })

    async def launchpad_stats(self, request):
        """Get launchpad statistics"""
        completed = sum(1 for l in pending_launches.values() if l["status"] == "completed")
        return web.json_response({
            "total_launches": completed,
            "pending": sum(1 for l in pending_launches.values() if l["status"] == "pending_payment"),
            "failed": sum(1 for l in pending_launches.values() if l["status"] == "failed"),
            "running_agents": len(running_agents)
        })

    async def launchpad_agents(self, request):
        """List all running agents (without sensitive data)"""
        safe_agents = []
        twitterapi_key = os.getenv("TWITTERAPI_KEY", "")

        for agent in running_agents.values():
            safe_agent = {k: v for k, v in agent.items()
                          if k not in ["twitter_cookie", "twitter_password", "twitter_email",
                                       "twitter_username_cred", "twitter_totp_secret",
                                       "x_access_token", "x_refresh_token"]}
            safe_agent["has_credentials"] = bool(agent.get("twitter_password"))
            safe_agent["has_oauth"] = bool(agent.get("x_access_token"))

            # Fetch Twitter profile image if we have twitter_username
            if agent.get("twitter_username") and not safe_agent.get("image_url") and twitterapi_key:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"https://api.twitterapi.io/twitter/user/info?userName={agent['twitter_username']}",
                            headers={"x-api-key": twitterapi_key},
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if data.get("data", {}).get("profilePicture"):
                                    # Get higher res version by replacing _normal with _400x400
                                    img_url = data["data"]["profilePicture"].replace("_normal", "_400x400")
                                    safe_agent["image_url"] = img_url
                except Exception as e:
                    logger.error(f"Failed to fetch Twitter profile image: {e}")

            safe_agents.append(safe_agent)
        return web.json_response({
            "total": len(running_agents),
            "agents": safe_agents
        })

    async def launchpad_agent_refresh(self, request):
        """Re-login to Twitter using stored credentials"""
        agent_id = request.match_info.get('agent_id')

        if agent_id not in running_agents:
            return web.json_response({"success": False, "message": "Agent not found"}, status=404)

        agent = running_agents[agent_id]

        if not agent.get("twitter_password"):
            return web.json_response({"success": False, "message": "No stored credentials"}, status=400)

        new_cookie = await try_agent_relogin(agent)

        if new_cookie:
            agent["twitter_cookie"] = new_cookie
            agent["last_relogin"] = time.time()
            agent["status"] = "running"
            save_agents()
            return web.json_response({"success": True, "message": "Agent re-login successful", "agent_id": agent_id})
        else:
            agent["status"] = "cookie_expired"
            save_agents()
            return web.json_response({"success": False, "message": "Re-login failed"}, status=400)

    async def launchpad_agent_trigger_post(self, request):
        """Manually trigger a post for an agent"""
        agent_id = request.match_info.get('agent_id')

        if agent_id not in running_agents:
            return web.json_response({"success": False, "message": "Agent not found"}, status=404)

        agent = running_agents[agent_id]

        if not agent.get("twitter_cookie"):
            return web.json_response({"success": False, "message": "No Twitter cookie, try /refresh first"}, status=400)

        # Bypass cooldown for manual trigger
        if "last_agent_post" not in agent_runner_state:
            agent_runner_state["last_agent_post"] = {}
        agent_runner_state["last_agent_post"][agent_id] = 0

        scanner = PolymarketScanner()
        result = await process_agent_posting(agent_id, agent, scanner, self)

        if result:
            return web.json_response({"success": True, "message": "Post published successfully"})
        else:
            return web.json_response({"success": False, "message": "Failed to post, check logs"}, status=500)

    # ==================== X OAuth 2.0 ====================

    async def x_oauth_start(self, request):
        """Start X OAuth 2.0 flow with PKCE"""
        launch_id = request.query.get('launch_id', '')

        if not X_CLIENT_ID:
            return web.json_response({"success": False, "message": "X OAuth not configured"}, status=500)

        # Generate PKCE code verifier and challenge
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).decode().rstrip('=')

        # Generate state
        state = secrets.token_urlsafe(32)

        # Store for callback
        X_OAUTH_STATES[state] = {
            "code_verifier": code_verifier,
            "created_at": time.time(),
            "launch_id": launch_id
        }

        # Clean old states (older than 10 minutes)
        now = time.time()
        for s in list(X_OAUTH_STATES.keys()):
            if now - X_OAUTH_STATES[s]["created_at"] > 600:
                del X_OAUTH_STATES[s]

        # Build authorization URL
        scopes = "tweet.read tweet.write users.read offline.access"
        auth_url = (
            f"https://x.com/i/oauth2/authorize"
            f"?response_type=code"
            f"&client_id={X_CLIENT_ID}"
            f"&redirect_uri={X_REDIRECT_URI}"
            f"&scope={scopes.replace(' ', '%20')}"
            f"&state={state}"
            f"&code_challenge={code_challenge}"
            f"&code_challenge_method=S256"
        )

        return web.json_response({
            "success": True,
            "auth_url": auth_url
        })

    async def x_oauth_callback(self, request):
        """Handle X OAuth 2.0 callback"""
        code = request.query.get('code')
        state = request.query.get('state')
        error = request.query.get('error')

        if error:
            return web.Response(
                text=f"<html><body><h1>Authorization failed</h1><p>{error}</p></body></html>",
                content_type="text/html"
            )

        if not code or not state:
            return web.Response(
                text="<html><body><h1>Missing code or state</h1></body></html>",
                content_type="text/html"
            )

        if state not in X_OAUTH_STATES:
            return web.Response(
                text="<html><body><h1>Invalid or expired state</h1></body></html>",
                content_type="text/html"
            )

        state_data = X_OAUTH_STATES.pop(state)
        code_verifier = state_data["code_verifier"]
        launch_id = state_data.get("launch_id", "")

        # Exchange code for tokens
        try:
            async with aiohttp.ClientSession() as session:
                auth_header = base64.b64encode(f"{X_CLIENT_ID}:{X_CLIENT_SECRET}".encode()).decode()
                async with session.post(
                    "https://api.twitter.com/2/oauth2/token",
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Authorization": f"Basic {auth_header}"
                    },
                    data={
                        "code": code,
                        "grant_type": "authorization_code",
                        "redirect_uri": X_REDIRECT_URI,
                        "code_verifier": code_verifier
                    }
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"X OAuth token error: {error_text}")
                        return web.Response(
                            text=f"<html><body><h1>Token exchange failed</h1><pre>{error_text}</pre></body></html>",
                            content_type="text/html"
                        )

                    token_data = await resp.json()
                    access_token = token_data.get("access_token")
                    refresh_token = token_data.get("refresh_token")
                    expires_in = token_data.get("expires_in", 7200)

                # Get user info
                async with session.get(
                    "https://api.twitter.com/2/users/me",
                    headers={"Authorization": f"Bearer {access_token}"}
                ) as user_resp:
                    if user_resp.status == 200:
                        user_data = await user_resp.json()
                        username = user_data.get("data", {}).get("username", "unknown")
                        user_id = user_data.get("data", {}).get("id", "")
                        name = user_data.get("data", {}).get("name", "")
                    else:
                        username = "unknown"
                        user_id = ""
                        name = ""

            # Store tokens (you can save to database/file)
            logger.info(f"X OAuth success: @{username} (ID: {user_id})")

            # Return success page with tokens in localStorage
            html = f"""
            <html>
            <head><title>Connected to X</title></head>
            <body style="font-family: sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: #fb651e;">
                <div style="background: white; padding: 40px; border-radius: 20px; text-align: center; max-width: 400px;">
                    <h1 style="color: #1da1f2;">Connected!</h1>
                    <p>Logged in as <strong>@{username}</strong></p>
                    <p style="color: #666; font-size: 14px;">You can close this window</p>
                </div>
                <script>
                    // Store tokens in opener window
                    if (window.opener) {{
                        window.opener.postMessage({{
                            type: 'x_oauth_success',
                            username: '{username}',
                            user_id: '{user_id}',
                            name: '{name}',
                            access_token: '{access_token}',
                            refresh_token: '{refresh_token}',
                            expires_in: {expires_in},
                            launch_id: '{launch_id}'
                        }}, '*');
                        setTimeout(() => window.close(), 2000);
                    }}
                </script>
            </body>
            </html>
            """
            return web.Response(text=html, content_type="text/html")

        except Exception as e:
            logger.error(f"X OAuth callback error: {e}")
            return web.Response(
                text=f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>",
                content_type="text/html"
            )

    async def x_post_tweet(self, request):
        """Post a tweet using X OAuth 2.0 access token"""
        try:
            data = await request.json()
            access_token = data.get("access_token")
            text = data.get("text")
            reply_to = data.get("reply_to")

            if not access_token or not text:
                return web.json_response({"success": False, "message": "Missing access_token or text"}, status=400)

            payload = {"text": text}
            if reply_to:
                payload["reply"] = {"in_reply_to_tweet_id": reply_to}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.twitter.com/2/tweets",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                ) as resp:
                    result = await resp.json()

                    if resp.status == 201:
                        tweet_id = result.get("data", {}).get("id")
                        return web.json_response({
                            "success": True,
                            "tweet_id": tweet_id,
                            "message": "Tweet posted successfully"
                        })
                    else:
                        logger.error(f"X API tweet error: {result}")
                        return web.json_response({
                            "success": False,
                            "message": result.get("detail", "Failed to post tweet"),
                            "error": result
                        }, status=resp.status)

        except Exception as e:
            logger.error(f"x_post_tweet error: {e}")
            return web.json_response({"success": False, "message": str(e)}, status=500)

    # ==================== X OAuth Token Management ====================

    async def refresh_x_token(self, refresh_token: str) -> Optional[Dict]:
        """Refresh X OAuth access token using refresh_token"""
        if not X_CLIENT_ID or not X_CLIENT_SECRET:
            logger.error("X OAuth not configured")
            return None

        try:
            async with aiohttp.ClientSession() as session:
                auth_header = base64.b64encode(f"{X_CLIENT_ID}:{X_CLIENT_SECRET}".encode()).decode()
                async with session.post(
                    "https://api.twitter.com/2/oauth2/token",
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Authorization": f"Basic {auth_header}"
                    },
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token
                    }
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"X token refresh error: {error_text}")
                        return None

                    token_data = await resp.json()
                    return {
                        "access_token": token_data.get("access_token"),
                        "refresh_token": token_data.get("refresh_token"),  # X gives new refresh token
                        "expires_in": token_data.get("expires_in", 7200)
                    }
        except Exception as e:
            logger.error(f"X token refresh exception: {e}")
            return None

    async def get_valid_x_token(self, agent: Dict) -> Optional[str]:
        """Get valid X access token for agent, refreshing if needed"""
        access_token = agent.get("x_access_token")
        refresh_token = agent.get("x_refresh_token")
        expires_at = agent.get("x_token_expires_at", 0)

        if not access_token or not refresh_token:
            return None

        # Check if token is expired (with 5 min buffer)
        if time.time() > expires_at - 300:
            logger.info(f"X token expired for @{agent.get('twitter_username')}, refreshing...")
            new_tokens = await self.refresh_x_token(refresh_token)
            if new_tokens:
                agent["x_access_token"] = new_tokens["access_token"]
                agent["x_refresh_token"] = new_tokens["refresh_token"]
                agent["x_token_expires_at"] = time.time() + new_tokens["expires_in"]
                # Save updated tokens
                agent_id = agent.get("agent_id")
                if agent_id and agent_id in running_agents:
                    running_agents[agent_id]["x_access_token"] = new_tokens["access_token"]
                    running_agents[agent_id]["x_refresh_token"] = new_tokens["refresh_token"]
                    running_agents[agent_id]["x_token_expires_at"] = agent["x_token_expires_at"]
                    save_agents()
                logger.info(f"X token refreshed for @{agent.get('twitter_username')}")
                return new_tokens["access_token"]
            else:
                logger.error(f"Failed to refresh X token for @{agent.get('twitter_username')}")
                return None

        return access_token

    # ==================== PROJECT LAUNCHPAD (Token Only) ====================

    async def projects_launch(self, request):
        """Submit a new project token launch (no agent)"""
        try:
            reader = await request.multipart()
            fields = {}
            image_data = None

            async for field in reader:
                if field.name == 'token_image':
                    image_data = await field.read()
                else:
                    fields[field.name] = await field.text()

            required = ['project_name', 'token_ticker', 'description', 'user_wallet']
            for f in required:
                if f not in fields:
                    return web.json_response({"success": False, "message": f"Missing field: {f}"}, status=400)

            if not image_data:
                return web.json_response({"success": False, "message": "Missing token image"}, status=400)

            # Check eligibility
            balance = await check_polyd_balance(fields['user_wallet'])
            if balance < REQUIRED_POLYD_BALANCE:
                return web.json_response({
                    "success": False,
                    "message": f"Insufficient $POLYD balance. Need {REQUIRED_POLYD_BALANCE:,}, have {balance:,.0f}"
                }, status=400)

            launch_id = hashlib.sha256(f"{fields['user_wallet']}{time.time()}".encode()).hexdigest()[:16]
            token_ticker = fields['token_ticker'].replace('$', '').upper()
            dev_buy_sol = min(float(fields.get('dev_buy_sol', 0) or 0), 85.0)

            pending_launches[launch_id] = {
                "type": "project",
                "project_name": fields['project_name'],
                "token_ticker": token_ticker,
                "token_name": fields['project_name'],
                "description": fields['description'],
                "website": fields.get('website'),
                "twitter": fields.get('twitter', '').replace('@', ''),
                "telegram": fields.get('telegram'),
                "user_wallet": fields['user_wallet'],
                "image_data": image_data,
                "dev_buy_sol": dev_buy_sol,
                "status": "pending_payment",
                "created_at": time.time()
            }
            save_launches()

            return web.json_response({
                "success": True,
                "message": f"Launch submitted. Send {LAUNCH_FEE_SOL} SOL to complete.",
                "launch_id": launch_id,
                "payment_address": LAUNCHPAD_WALLET_PUBLIC_KEY,
                "payment_amount": LAUNCH_FEE_SOL
            })

        except Exception as e:
            logger.error(f"Project launch error: {e}")
            return web.json_response({"success": False, "message": str(e)}, status=500)

    async def projects_confirm(self, request):
        """Confirm payment and create project token on pump.fun"""
        launch_id = request.match_info.get('launch_id')

        if launch_id not in pending_launches:
            return web.json_response({"success": False, "message": "Launch not found"}, status=404)

        launch = pending_launches[launch_id]

        if launch["status"] == "completed":
            return web.json_response({
                "success": True,
                "message": "Token already created",
                "launch_id": launch_id,
                "token_mint": launch.get("token_mint"),
                "tx_signature": launch.get("tx_signature")
            })

        if launch["status"] not in ["pending_payment", "failed"]:
            return web.json_response({"success": False, "message": f"Invalid status: {launch['status']}"}, status=400)

        launch["status"] = "creating"
        logger.info(f"Creating project token: {launch['token_name']} (${launch['token_ticker']})")

        dev_buy_amount = launch.get("dev_buy_sol", 0)
        logger.info(f"Creating token with dev_buy_sol={dev_buy_amount}")

        result = await create_token_on_pumpfun(
            name=launch["token_name"],
            symbol=launch["token_ticker"],
            description=launch["description"],
            image_data=launch["image_data"],
            website=launch.get("website"),
            twitter=launch.get("twitter"),
            telegram=launch.get("telegram"),
            dev_buy_sol=dev_buy_amount
        )

        if result:
            launch["status"] = "completed"
            launch["token_mint"] = result["token_mint"]
            launch["tx_signature"] = result["tx_signature"]

            # Transfer tokens to user if dev_buy was used
            transfer_tx = None
            if dev_buy_amount > 0 and result.get("signer_keypair"):
                logger.info(f"Transferring tokens to user wallet: {launch['user_wallet']} (dev_buy={dev_buy_amount})")
                # Wait for tx confirmation with retries
                for attempt in range(3):
                    await asyncio.sleep(5 + attempt * 5)  # 5s, 10s, 15s
                    transfer_tx = await transfer_spl_tokens(
                        token_mint=result["token_mint"],
                        recipient_wallet=launch["user_wallet"],
                        signer_keypair=result["signer_keypair"]
                    )
                    if transfer_tx:
                        launch["transfer_tx"] = transfer_tx
                        logger.info(f"Tokens transferred to user: {transfer_tx}")
                        break
                    logger.warning(f"Transfer attempt {attempt + 1} failed, retrying...")

            save_launches()

            return web.json_response({
                "success": True,
                "message": "Token created!" + (" Tokens sent to your wallet." if transfer_tx else ""),
                "launch_id": launch_id,
                "token_mint": result["token_mint"],
                "tx_signature": result["tx_signature"],
                "transfer_tx": transfer_tx
            })
        else:
            launch["status"] = "failed"
            return web.json_response({"success": False, "message": "Failed to create token"}, status=500)

    async def projects_list(self, request):
        """List all project applications"""
        try:
            projects_file = Path(__file__).parent / 'data' / 'projects.json'

            if not projects_file.exists():
                return web.json_response({"projects": []})

            with open(projects_file, 'r') as f:
                projects = json.load(f)

            return web.json_response({"projects": projects})

        except Exception as e:
            logger.error(f"Projects list error: {e}")
            return web.json_response({"projects": []})

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"API server started on http://{self.host}:{self.port}")
        return runner


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load launchpad data on startup
    load_launches()
    load_agents()

    server = APIServer()
    set_server_instance(server)

    async def main():
        runner = await server.start()

        # Start agent runner as background task
        agent_task = asyncio.create_task(agent_runner_loop())
        logger.info("Agent runner background task started")

        try:
            while True:
                await asyncio.sleep(3600)
        except KeyboardInterrupt:
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                logger.info("Agent runner stopped")
            await runner.cleanup()

    asyncio.run(main())
