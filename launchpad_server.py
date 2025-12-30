"""
Polydictions Launchpad Server
Creates agent tokens on pump.fun + Twitter bot agent
"""

import os
import json
import time
import hashlib
import asyncio
import aiohttp
import base64
import re
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
from dotenv import load_dotenv
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Solana imports
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction
from solders.commitment_config import CommitmentLevel
from solders.rpc.requests import SendVersionedTransaction
from solders.rpc.config import RpcSendTransactionConfig

load_dotenv()

# App will be initialized after lifespan is defined
app = None

# Config
LAUNCHPAD_WALLET_PRIVATE_KEY = os.getenv("LAUNCHPAD_WALLET_PRIVATE_KEY")
LAUNCHPAD_WALLET_PUBLIC_KEY = "8iWGVEYYvrqArN6ChbbLEgsY3eEHeEbwssswYANq2mgS"
POLYD_MINT = "iATcGSt9DhJF9ZiJ6dmR153N7bW2G4J9dSSDxWSpump"
REQUIRED_POLYD_BALANCE = 1_000  # 1000 $POLYD required
LAUNCH_FEE_SOL = 0.05
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")
HELIUS_RPC = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"

# TwitterAPI.io config
TWITTERAPI_KEY = os.getenv("TWITTERAPI_KEY", "")
TWITTERAPI_BASE = "https://api.twitterapi.io"
TWITTER_PROXY = os.getenv("TWITTER_PROXY", "")

# Agent runner config
POLYMARKET_API = "https://gamma-api.polymarket.com"
POLYFACTUAL_API_URL = "https://deep-research-api.thekid-solana.workers.dev/answer"
POLYFACTUAL_API_KEY = os.getenv("POLYFACTUAL_API_KEY", "")
FACTSAI_API_URL = "https://factsai.org/answer"
FACTSAI_API_KEY = os.getenv("FACTSAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# Agent intervals
POST_INTERVAL_HOURS = int(os.getenv("AGENT_POST_INTERVAL", "4"))
MENTION_CHECK_INTERVAL = int(os.getenv("MENTION_CHECK_INTERVAL", "60"))

# Niche keywords for filtering Polymarket events
NICHE_KEYWORDS = {
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol", "blockchain", "defi", "nft", "token", "coin", "memecoin"],
    "politics": ["trump", "biden", "election", "president", "congress", "senate", "democrat", "republican", "vote", "poll"],
    "sports": ["nba", "nfl", "mlb", "nhl", "soccer", "football", "basketball", "baseball", "championship", "playoffs", "super bowl"],
    "entertainment": ["movie", "film", "oscar", "emmy", "grammy", "netflix", "disney", "marvel", "actor", "actress", "box office"],
    "tech": ["ai", "artificial intelligence", "openai", "google", "apple", "microsoft", "amazon", "meta", "tesla", "elon musk"],
    "finance": ["stock", "market", "fed", "interest rate", "inflation", "gdp", "recession", "economy", "s&p", "nasdaq"]
}

# Encryption key for storing credentials
# Derived from SESSION_SECRET using PBKDF2
def get_encryption_key() -> bytes:
    """Derive encryption key from SESSION_SECRET"""
    secret = os.getenv("SESSION_SECRET", "default-secret-change-me")
    salt = b"polydictions-launchpad-salt"  # Fixed salt for consistent key
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(secret.encode()))
    return key

ENCRYPTION_KEY = get_encryption_key()
fernet = Fernet(ENCRYPTION_KEY)


def encrypt_credential(data: str) -> str:
    """Encrypt a credential string"""
    if not data:
        return ""
    return fernet.encrypt(data.encode()).decode()


def decrypt_credential(encrypted_data: str) -> str:
    """Decrypt a credential string"""
    if not encrypted_data:
        return ""
    try:
        return fernet.decrypt(encrypted_data.encode()).decode()
    except Exception as e:
        print(f"Decryption error: {e}")
        return ""

# pump.fun API endpoints
PUMPFUN_IPFS = "https://pump.fun/api/ipfs"
PUMPPORTAL_API = "https://pumpportal.fun/api/trade-local"

# Store pending launches (in memory + file persistence)
pending_launches: Dict[str, Dict[str, Any]] = {}
LAUNCHES_FILE = Path("launches_history.json")

# Store running agents
running_agents: Dict[str, Dict[str, Any]] = {}
AGENTS_FILE = Path("agents.json")


def load_launches():
    """Load launches from file"""
    global pending_launches
    if LAUNCHES_FILE.exists():
        try:
            with open(LAUNCHES_FILE, "r") as f:
                data = json.load(f)
                # Don't load image_data from file (too large)
                for lid, launch in data.items():
                    launch.pop("image_data", None)
                pending_launches = data
                print(f"Loaded {len(pending_launches)} launches from file")
        except Exception as e:
            print(f"Error loading launches: {e}")


def save_launches():
    """Save launches to file (without image data and sensitive twitter data)"""
    try:
        save_data = {}
        for lid, launch in pending_launches.items():
            save_data[lid] = {k: v for k, v in launch.items()
                           if k not in ["image_data", "twitter_cookie", "twitter_password",
                                       "twitter_credentials"]}
        with open(LAUNCHES_FILE, "w") as f:
            json.dump(save_data, f, indent=2)
    except Exception as e:
        print(f"Error saving launches: {e}")


def load_agents():
    """Load agents from file and decrypt credentials"""
    global running_agents
    if AGENTS_FILE.exists():
        try:
            with open(AGENTS_FILE, "r") as f:
                data = json.load(f)
                # Decrypt encrypted fields
                for aid, agent in data.items():
                    for key in list(agent.keys()):
                        if key.endswith("_encrypted"):
                            original_key = key.replace("_encrypted", "")
                            agent[original_key] = decrypt_credential(agent[key])
                            del agent[key]
                running_agents = data
                print(f"Loaded {len(running_agents)} agents from file")
        except Exception as e:
            print(f"Error loading agents: {e}")


def save_agents():
    """Save agents to file with encrypted credentials"""
    try:
        save_data = {}
        for aid, agent in running_agents.items():
            agent_data = {}
            for k, v in agent.items():
                # Encrypt sensitive fields
                if k in ["twitter_cookie", "twitter_password", "twitter_email",
                         "twitter_username_cred", "twitter_totp_secret"]:
                    agent_data[f"{k}_encrypted"] = encrypt_credential(v) if v else ""
                else:
                    agent_data[k] = v
            save_data[aid] = agent_data
        with open(AGENTS_FILE, "w") as f:
            json.dump(save_data, f, indent=2)
    except Exception as e:
        print(f"Error saving agents: {e}")


# Load on startup
load_launches()
load_agents()

# ============ AGENT RUNNER CLASSES ============

# Store for agent runner state
agent_runner_state = {
    "last_post_time": 0,
    "answered_mentions": {},  # agent_id -> set of tweet_ids
    "posted_events": {},  # agent_id -> set of event_ids
    "mention_start_times": {}  # agent_id -> timestamp
}
ANSWERED_MENTIONS_FILE = Path("agent_answered_mentions.json")


def load_answered_mentions() -> Dict[str, Set[str]]:
    """Load answered mentions per agent"""
    if ANSWERED_MENTIONS_FILE.exists():
        try:
            with open(ANSWERED_MENTIONS_FILE, 'r') as f:
                data = json.load(f)
                return {k: set(v) for k, v in data.items()}
        except:
            pass
    return {}


def save_answered_mentions(answered: Dict[str, Set[str]]):
    """Save answered mentions"""
    try:
        with open(ANSWERED_MENTIONS_FILE, 'w') as f:
            json.dump({k: list(v) for k, v in answered.items()}, f)
    except Exception as e:
        print(f"Error saving answered mentions: {e}")


class TwitterAPI:
    """Post tweets via TwitterAPI.io using login cookies"""

    @staticmethod
    async def post_tweet(login_cookies: str, tweet_text: str) -> Optional[str]:
        """Post a tweet using login cookies"""
        url = f"{TWITTERAPI_BASE}/twitter/tweet/create_tweet_v2"
        headers = {"Content-Type": "application/json", "X-API-Key": TWITTERAPI_KEY}
        data = {"tweet_text": tweet_text, "login_cookies": login_cookies, "proxy": TWITTER_PROXY}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data, timeout=60) as resp:
                    result = await resp.json()
                    if result.get("status") == "success":
                        return result.get("tweet_id") or "posted"
                    else:
                        print(f"Tweet failed: {result}")
                        return None
        except Exception as e:
            print(f"Tweet error: {e}")
            return None

    @staticmethod
    async def post_reply(login_cookies: str, tweet_id: str, reply_text: str) -> Optional[str]:
        """Post a reply to a tweet"""
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
                    print(f"Reply failed: {result}")
                    return None
        except Exception as e:
            print(f"Reply error: {e}")
            return None

    @staticmethod
    async def get_mentions(username: str, since_time: int = None) -> List[Dict]:
        """Get mentions for a username"""
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
            print(f"Get mentions error: {e}")
            return []

    @staticmethod
    async def relogin(username: str, email: str, password: str, totp_secret: str) -> Optional[str]:
        """Login to Twitter and get new cookies"""
        return await twitter_login(username, email, password, totp_secret)


class PolymarketScanner:
    """Scan Polymarket for opportunities filtered by niche"""
    SPAM_WORDS = ['up or down', 'higher or lower', 'above or below', 'am-', 'pm-', 'am et', 'pm et', ':00', ':15', ':30', ':45']

    async def fetch_events(self, limit: int = 100) -> List[Dict]:
        """Fetch events sorted by volume"""
        url = f"{POLYMARKET_API}/events"
        params = {'limit': limit, 'active': 'true', 'closed': 'false', 'order': 'volume', 'ascending': 'false'}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception as e:
            print(f"Fetch error: {e}")
        return []

    def filter_by_niche(self, events: List[Dict], niche: str) -> List[Dict]:
        """Filter events by niche keywords"""
        if niche == "general":
            return events
        keywords = NICHE_KEYWORDS.get(niche, [])
        if not keywords:
            return events
        filtered = []
        for event in events:
            title = event.get("title", "").lower()
            description = event.get("description", "").lower()
            combined = f"{title} {description}"
            if any(kw in combined for kw in keywords):
                filtered.append(event)
        return filtered

    def is_valid_market(self, event: Dict, posted_ids: set) -> bool:
        """Check if market is valid for posting"""
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
        """Extract market data"""
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
            'volume': float(event.get('volume', 0) or 0)
        }


class AIAnalyzer:
    """Generate AI analysis with custom prompts"""

    @staticmethod
    async def analyze_market(title: str, yes_odds: float, no_odds: float, volume: float, custom_prompt: str = "", niche: str = "general") -> Optional[Dict]:
        """Get AI analysis with custom personality"""
        if not POLYFACTUAL_API_KEY:
            return {"reasoning": "Interesting market setup."}

        query = f"""Analyze this Polymarket prediction market: [{title}]
Current odds: YES {yes_odds:.0f}% / NO {no_odds:.0f}%
Volume: ${volume:,.0f}

Provide a brief analysis with:
1) Key factors supporting YES
2) Key factors supporting NO
3) Your recommendation (YES or NO) with confidence level

Keep it concise - this is for a tweet."""

        if custom_prompt:
            query += f"\n\nADDITIONAL INSTRUCTIONS: {custom_prompt}"

        headers = {'Content-Type': 'application/json', 'X-API-Key': POLYFACTUAL_API_KEY}
        data = {'query': query, 'text': True}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(POLYFACTUAL_API_URL, headers=headers, json=data, timeout=120) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        if result.get('success') and result.get('data'):
                            answer = result['data'].get('answer', '') or result['data'].get('text', '')
                            if isinstance(answer, dict):
                                answer = answer.get('text', str(answer))
                            reasoning = str(answer).strip()
                            reasoning = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', reasoning)
                            reasoning = re.sub(r'https?://\S+', '', reasoning)
                            reasoning = reasoning.replace('**', '')
                            return {'reasoning': reasoning}
        except Exception as e:
            print(f"AI analysis error: {e}")
        return None

    @staticmethod
    async def get_facts(question: str) -> Optional[str]:
        """Get facts from FactsAI API"""
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
            print(f"FactsAI error: {e}")
        return None

    @staticmethod
    async def generate_reply(question: str, facts: str, custom_prompt: str = "", ticker: str = "") -> Optional[str]:
        """Generate a reply using Claude API"""
        if not ANTHROPIC_API_KEY:
            response = facts[:400] if facts else "interesting question! let me look into that."
            if ticker:
                response += f"\n\n${ticker} 🧡"
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
                            response += f"\n\n${ticker} 🧡"
                        return response[:500]
        except Exception as e:
            print(f"Claude error: {e}")
        return None


def format_volume(volume: float) -> str:
    if volume >= 1_000_000:
        return f"${volume/1_000_000:.1f}M"
    elif volume >= 1_000:
        return f"${volume/1_000:.0f}K"
    return f"${volume:.0f}"


def create_tweet(agent: Dict, market: Dict, reasoning: str) -> str:
    """Create tweet with market info and analysis"""
    niche = agent.get("agent_niche", "general")
    openers = {
        "crypto": ["🚀 alpha alert", "💎 found some edge", "🔮 prediction time"],
        "politics": ["🗳️ political alpha", "🏛️ worth watching", "📊 odds looking off"],
        "sports": ["🏆 game day alpha", "⚽ spotted value", "🏀 market inefficiency"],
        "entertainment": ["🎬 hot take", "🎭 entertainment alpha", "🌟 worth a look"],
        "tech": ["🤖 tech alpha", "💻 spotted something", "🔬 market analysis"],
        "finance": ["📈 market alpha", "💰 found some edge", "📊 worth watching"],
        "general": ["🧡 alpha alert", "🔮 prediction", "📊 market analysis"]
    }
    opener = random.choice(openers.get(niche, openers["general"]))
    short_reasoning = reasoning[:200] + "..." if len(reasoning) > 200 else reasoning

    tweet = f"""{opener}

{market['title']}

YES {market['yes_odds']:.1f}% / NO {market['no_odds']:.1f}%
vol: {format_volume(market['volume'])}

{short_reasoning}

polymarket.com/event/{market['slug']}"""

    ticker = agent.get("token_ticker", "")
    if ticker:
        tweet += f"\n\n${ticker} 🧡"
    return tweet[:280]


async def try_agent_relogin(agent: Dict) -> Optional[str]:
    """Try to re-login agent if tweet failed"""
    username = agent.get("twitter_username_cred")
    email = agent.get("twitter_email")
    password = agent.get("twitter_password")
    totp = agent.get("twitter_totp_secret")

    if not all([username, email, password, totp]):
        print(f"Agent missing credentials for re-login")
        return None

    print(f"Attempting re-login for @{username}...")
    new_cookie = await TwitterAPI.relogin(username, email, password, totp)

    if new_cookie:
        print("Re-login successful!")
        return new_cookie
    else:
        print("Re-login failed")
        return None


async def process_agent_posting(agent_id: str, agent: Dict, scanner: PolymarketScanner) -> bool:
    """Find a market and post about it for an agent"""
    niche = agent.get("agent_niche", "general")
    custom_prompt = agent.get("custom_prompt", "")
    twitter_cookie = agent.get("twitter_cookie")

    if not twitter_cookie:
        print(f"[{agent_id}] No twitter cookie")
        return False

    print(f"\n[{agent_id}] Scanning {niche} markets...")

    posted_events = agent_runner_state["posted_events"].get(agent_id, set())
    events = await scanner.fetch_events(limit=100)
    events = scanner.filter_by_niche(events, niche)

    valid_markets = []
    for event in events:
        if scanner.is_valid_market(event, posted_events):
            market_data = scanner.parse_market_data(event)
            valid_markets.append(market_data)

    print(f"[{agent_id}] Found {len(valid_markets)} valid {niche} markets")

    if not valid_markets:
        return False

    top_markets = sorted(valid_markets, key=lambda x: x['volume'], reverse=True)[:10]
    market = random.choice(top_markets)
    print(f"[{agent_id}] Selected: {market['title']}")

    print(f"[{agent_id}] Getting AI analysis...")
    ai_result = await AIAnalyzer.analyze_market(
        market['title'], market['yes_odds'], market['no_odds'], market['volume'],
        custom_prompt=custom_prompt, niche=niche
    )
    reasoning = ai_result['reasoning'] if ai_result else "Interesting setup here."

    tweet_text = create_tweet(agent, market, reasoning)
    print(f"[{agent_id}] Tweet:\n{tweet_text}\n")

    tweet_id = await TwitterAPI.post_tweet(twitter_cookie, tweet_text)

    if tweet_id:
        print(f"[{agent_id}] ✓ Posted! ID: {tweet_id}")
        if agent_id not in agent_runner_state["posted_events"]:
            agent_runner_state["posted_events"][agent_id] = set()
        agent_runner_state["posted_events"][agent_id].add(market['event_id'])
        return True
    else:
        print(f"[{agent_id}] ✗ Failed to post")
        new_cookie = await try_agent_relogin(agent)
        if new_cookie:
            agent["twitter_cookie"] = new_cookie
            running_agents[agent_id]["twitter_cookie"] = new_cookie
            save_agents()
        return False


async def process_agent_mentions(agent_id: str, agent: Dict) -> int:
    """Check for mentions and respond to them"""
    username = agent.get("twitter_username")
    twitter_cookie = agent.get("twitter_cookie")
    custom_prompt = agent.get("custom_prompt", "")
    ticker = agent.get("token_ticker", "")

    if not username or not twitter_cookie:
        return 0

    mention_start = agent_runner_state["mention_start_times"].get(agent_id, int(datetime.now().timestamp()))
    print(f"[{agent_id}] Checking mentions for @{username}...")

    mentions = await TwitterAPI.get_mentions(username, since_time=mention_start)
    answered = agent_runner_state["answered_mentions"].get(agent_id, set())

    real_mentions = [
        m for m in mentions
        if f"@{username.lower()}" in m.get("text", "").lower()
        and m.get("id") not in answered
        and m.get("author", {}).get("userName", "").lower() != username.lower()
    ]

    if not real_mentions:
        print(f"[{agent_id}] No new mentions")
        return 0

    print(f"[{agent_id}] Found {len(real_mentions)} new mentions!")
    answered_count = 0

    for mention in real_mentions[:5]:
        tweet_id = mention.get("id")
        text = mention.get("text", "")
        author = mention.get("author", {}).get("userName", "unknown")

        question = re.sub(r'@\w+\s*', '', text).strip()
        if len(question) < 5:
            continue

        print(f"[{agent_id}] Mention from @{author}: {question[:50]}...")

        facts = await AIAnalyzer.get_facts(question)
        reply = await AIAnalyzer.generate_reply(question=question, facts=facts or "", custom_prompt=custom_prompt, ticker=ticker)

        if not reply:
            print(f"[{agent_id}] Failed to generate reply")
            continue

        print(f"[{agent_id}] Reply: {reply[:100]}...")
        result = await TwitterAPI.post_reply(twitter_cookie, tweet_id, reply)

        if result:
            print(f"[{agent_id}] ✓ Replied to @{author}!")
            if agent_id not in agent_runner_state["answered_mentions"]:
                agent_runner_state["answered_mentions"][agent_id] = set()
            agent_runner_state["answered_mentions"][agent_id].add(tweet_id)
            answered_count += 1
        else:
            print(f"[{agent_id}] ✗ Failed to reply")
            new_cookie = await try_agent_relogin(agent)
            if new_cookie:
                agent["twitter_cookie"] = new_cookie
                running_agents[agent_id]["twitter_cookie"] = new_cookie
                save_agents()

        await asyncio.sleep(5)

    agent_runner_state["mention_start_times"][agent_id] = int(datetime.now().timestamp())
    return answered_count


async def agent_runner_loop():
    """Background task that runs all agents"""
    print("=" * 50)
    print("AGENT RUNNER STARTED (integrated)")
    print(f"Post interval: {POST_INTERVAL_HOURS} hours")
    print(f"Mention check interval: {MENTION_CHECK_INTERVAL} seconds")
    print("=" * 50)

    scanner = PolymarketScanner()
    agent_runner_state["answered_mentions"] = load_answered_mentions()

    while True:
        try:
            if not running_agents:
                print("[AgentRunner] No agents running. Waiting...")
                await asyncio.sleep(60)
                continue

            current_time = datetime.now().timestamp()
            should_post = (current_time - agent_runner_state["last_post_time"]) >= (POST_INTERVAL_HOURS * 3600)

            for agent_id, agent in list(running_agents.items()):
                if agent.get("status") == "cookie_expired":
                    print(f"[{agent_id}] Skipped - cookie expired")
                    continue
                if agent.get("status") != "running":
                    continue

                try:
                    # Check mentions
                    mentions_answered = await process_agent_mentions(agent_id, agent)
                    if mentions_answered > 0:
                        save_answered_mentions(agent_runner_state["answered_mentions"])

                    # Post new content
                    if should_post:
                        await process_agent_posting(agent_id, agent, scanner)

                    await asyncio.sleep(5)

                except Exception as e:
                    print(f"[{agent_id}] Error: {e}")
                    import traceback
                    traceback.print_exc()

            if should_post:
                agent_runner_state["last_post_time"] = current_time
                print(f"\n[AgentRunner] Next post in {POST_INTERVAL_HOURS} hours...")

            print(f"[AgentRunner] Next mention check in {MENTION_CHECK_INTERVAL} seconds...")
            print("-" * 50)

        except Exception as e:
            print(f"[AgentRunner] Main loop error: {e}")
            import traceback
            traceback.print_exc()

        await asyncio.sleep(MENTION_CHECK_INTERVAL)


# ============ FASTAPI LIFESPAN ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup: launch agent runner background task
    agent_task = asyncio.create_task(agent_runner_loop())
    print("[Launchpad] Agent runner background task started")
    yield
    # Shutdown: cancel agent runner
    agent_task.cancel()
    try:
        await agent_task
    except asyncio.CancelledError:
        print("[Launchpad] Agent runner stopped")


# ============ INITIALIZE FASTAPI APP ============

app = FastAPI(title="Polydictions Launchpad API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TwitterLoginRequest(BaseModel):
    username: str
    email: str
    password: str
    totp_secret: str  # Required by TwitterAPI.io


class TwitterLoginResponse(BaseModel):
    success: bool
    message: str
    login_cookie: Optional[str] = None


class LaunchResponse(BaseModel):
    success: bool
    message: str
    launch_id: Optional[str] = None
    payment_address: Optional[str] = None
    payment_amount: Optional[float] = None
    token_mint: Optional[str] = None
    tx_signature: Optional[str] = None


def get_launchpad_keypair() -> Keypair:
    """Load launchpad wallet keypair from env"""
    if not LAUNCHPAD_WALLET_PRIVATE_KEY:
        raise ValueError("LAUNCHPAD_WALLET_PRIVATE_KEY not set")
    return Keypair.from_base58_string(LAUNCHPAD_WALLET_PRIVATE_KEY)


async def twitter_login(username: str, email: str, password: str, totp_secret: str) -> Optional[str]:
    """
    Login to Twitter via TwitterAPI.io and get login_cookie
    totp_secret is required by the API for stable login
    """
    url = f"{TWITTERAPI_BASE}/twitter/user_login_v2"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": TWITTERAPI_KEY
    }

    data = {
        "user_name": username,
        "email": email,
        "password": password,
        "proxy": TWITTER_PROXY,
        "totp_secret": totp_secret
    }

    try:
        print(f"Twitter login attempt for @{username}")
        print(f"Using proxy: {TWITTER_PROXY[:30]}...")
        print(f"API Key: {TWITTERAPI_KEY[:20]}...")

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data, timeout=120) as resp:
                result = await resp.json()
                print(f"TwitterAPI response: {result}")

                if result.get("status") == "success":
                    print("Twitter login successful!")
                    # API returns login_cookies (with 's')
                    return result.get("login_cookies") or result.get("login_cookie")
                else:
                    error_msg = result.get("msg") or result.get("message") or str(result)
                    print(f"Twitter login failed: {error_msg}")
                    return None
    except Exception as e:
        print(f"Twitter login error: {e}")
        import traceback
        traceback.print_exc()
        return None


async def check_polyd_balance(wallet_address: str) -> float:
    """Check $POLYD token balance for a wallet"""
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
        print(f"Error checking POLYD balance: {e}")
        return 0


async def upload_metadata_to_ipfs(
    name: str,
    symbol: str,
    description: str,
    image_data: bytes,
    image_filename: str,
    website: str = None,
    twitter: str = None,
    telegram: str = None
) -> Optional[str]:
    """Upload token metadata to IPFS via pump.fun API"""
    try:
        print(f"Uploading metadata to IPFS:")
        print(f"  Name: {name}")
        print(f"  Symbol: {symbol}")
        print(f"  Description: {description[:50]}...")

        form_data = aiohttp.FormData()
        form_data.add_field('file', image_data, filename=image_filename, content_type='image/png')
        form_data.add_field('name', name)
        form_data.add_field('symbol', symbol)
        form_data.add_field('description', description)
        form_data.add_field('showName', 'true')

        if website and website.strip():
            form_data.add_field('website', website.strip())
            print(f"  Website: {website.strip()}")
        if twitter and twitter.strip():
            tw = twitter.strip()
            if not tw.startswith('http'):
                tw = f"https://x.com/{tw.replace('@', '')}"
            form_data.add_field('twitter', tw)
            print(f"  Twitter: {tw}")
        if telegram and telegram.strip():
            tg = telegram.strip()
            if not tg.startswith('http'):
                tg = f"https://t.me/{tg.replace('@', '')}"
            form_data.add_field('telegram', tg)
            print(f"  Telegram: {tg}")

        async with aiohttp.ClientSession() as session:
            async with session.post(PUMPFUN_IPFS, data=form_data) as resp:
                if resp.status != 200:
                    print(f"IPFS upload failed: {await resp.text()}")
                    return None

                result = await resp.json()
                return result.get("metadataUri")

    except Exception as e:
        print(f"Error uploading to IPFS: {e}")
        return None


async def create_token_on_pumpfun(
    name: str,
    symbol: str,
    description: str,
    image_data: bytes,
    website: str = None,
    twitter: str = None,
    telegram: str = None,
    dev_buy_sol: float = 0
) -> Optional[Dict[str, str]]:
    """Create token on pump.fun using Local Transaction API"""
    try:
        signer_keypair = get_launchpad_keypair()
        mint_keypair = Keypair()

        print(f"Creating token with mint: {mint_keypair.pubkey()}")
        print(f"Signer: {signer_keypair.pubkey()}")

        # Step 1: Upload metadata to IPFS
        print("Step 1: Uploading metadata to IPFS...")
        metadata_uri = await upload_metadata_to_ipfs(
            name=name,
            symbol=symbol,
            description=description,
            image_data=image_data,
            image_filename=f"{symbol.lower()}.png",
            website=website,
            twitter=twitter,
            telegram=telegram
        )

        if not metadata_uri:
            print("Failed to upload metadata")
            return None

        print(f"Metadata URI: {metadata_uri}")

        # Step 2: Get create transaction from PumpPortal
        print("Step 2: Getting create transaction...")

        create_payload = {
            "publicKey": str(signer_keypair.pubkey()),
            "action": "create",
            "tokenMetadata": {
                "name": name,
                "symbol": symbol,
                "uri": metadata_uri
            },
            "mint": str(mint_keypair.pubkey()),
            "denominatedInSol": "true",
            "amount": dev_buy_sol,
            "slippage": 10,
            "priorityFee": 0.0005,
            "pool": "pump"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                PUMPPORTAL_API,
                headers={"Content-Type": "application/json"},
                json=create_payload
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"PumpPortal API error: {error_text}")
                    return None

                tx_bytes = await resp.read()

        # Step 3: Sign transaction
        print("Step 3: Signing transaction...")

        tx = VersionedTransaction.from_bytes(tx_bytes)
        tx = VersionedTransaction(tx.message, [mint_keypair, signer_keypair])

        # Step 4: Send transaction
        print("Step 4: Sending transaction...")

        commitment = CommitmentLevel.Confirmed
        config = RpcSendTransactionConfig(preflight_commitment=commitment)
        tx_payload = SendVersionedTransaction(tx, config)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                HELIUS_RPC,
                headers={"Content-Type": "application/json"},
                data=tx_payload.to_json()
            ) as resp:
                result = await resp.json()

                if "error" in result:
                    print(f"Transaction error: {result['error']}")
                    return None

                tx_signature = result.get("result")

        print(f"Transaction sent: {tx_signature}")

        return {
            "token_mint": str(mint_keypair.pubkey()),
            "tx_signature": tx_signature
        }

    except Exception as e:
        print(f"Error creating token: {e}")
        import traceback
        traceback.print_exc()
        return None


async def refresh_agent_cookie(agent_id: str) -> Optional[str]:
    """
    Re-login to Twitter using stored credentials when cookie expires.
    Returns new cookie if successful, None otherwise.
    """
    if agent_id not in running_agents:
        print(f"Agent {agent_id} not found for re-login")
        return None

    agent = running_agents[agent_id]

    # Check if we have stored credentials
    username = agent.get("twitter_username_cred")
    email = agent.get("twitter_email")
    password = agent.get("twitter_password")
    totp_secret = agent.get("twitter_totp_secret")

    if not all([username, email, password, totp_secret]):
        print(f"Agent {agent_id} missing credentials for re-login")
        return None

    print(f"Re-logging in agent {agent_id} (@{username})...")

    new_cookie = await twitter_login(
        username=username,
        email=email,
        password=password,
        totp_secret=totp_secret
    )

    if new_cookie:
        agent["twitter_cookie"] = new_cookie
        agent["last_relogin"] = time.time()
        save_agents()
        print(f"Agent {agent_id} re-login successful!")
        return new_cookie
    else:
        print(f"Agent {agent_id} re-login failed")
        agent["status"] = "cookie_expired"
        save_agents()
        return None


async def start_agent(launch_id: str, launch: Dict[str, Any], credentials: Dict[str, str] = None):
    """
    Start the Twitter agent for a launch
    Stores encrypted credentials for re-login when cookie expires
    """
    agent_id = launch_id

    running_agents[agent_id] = {
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

    # Store credentials for re-login (will be encrypted when saved)
    if credentials:
        running_agents[agent_id].update({
            "twitter_username_cred": credentials.get("username"),
            "twitter_email": credentials.get("email"),
            "twitter_password": credentials.get("password"),
            "twitter_totp_secret": credentials.get("totp_secret")
        })

    save_agents()
    print(f"Agent {agent_id} started for @{launch.get('twitter_username')}")
    print(f"  Niche: {launch.get('agent_niche', 'general')}")
    if launch.get("custom_prompt"):
        print(f"  Custom prompt: {launch.get('custom_prompt')[:50]}...")
    if credentials:
        print(f"  Credentials stored (encrypted) for re-login")

    return agent_id


# ============ API ENDPOINTS ============

@app.get("/")
async def root():
    return {"status": "ok", "service": "Polydictions Launchpad"}


@app.get("/api/launchpad/config")
async def get_frontend_config():
    """Get frontend configuration (public values only)"""
    return {
        "helius_rpc": HELIUS_RPC,
        "polyd_mint": POLYD_MINT,
        "required_balance": REQUIRED_POLYD_BALANCE,
        "launch_fee": LAUNCH_FEE_SOL,
        "payment_address": LAUNCHPAD_WALLET_PUBLIC_KEY,
        "reown_project_id": os.getenv("REOWN_PROJECT_ID", "")
    }


@app.post("/api/launchpad/twitter-login", response_model=TwitterLoginResponse)
async def api_twitter_login(request: TwitterLoginRequest):
    """Login to Twitter and get login cookie"""

    if not TWITTERAPI_KEY:
        raise HTTPException(status_code=500, detail="TwitterAPI.io not configured")

    if not TWITTER_PROXY:
        raise HTTPException(status_code=500, detail="Twitter proxy not configured")

    login_cookie = await twitter_login(
        username=request.username,
        email=request.email,
        password=request.password,
        totp_secret=request.totp_secret
    )

    if login_cookie:
        return TwitterLoginResponse(
            success=True,
            message="Twitter login successful",
            login_cookie=login_cookie
        )
    else:
        raise HTTPException(
            status_code=400,
            detail="Twitter login failed. Check credentials and try again."
        )


@app.get("/api/launchpad/wallet")
async def get_payment_wallet():
    """Get the payment wallet address"""
    return {
        "address": LAUNCHPAD_WALLET_PUBLIC_KEY,
        "fee_sol": LAUNCH_FEE_SOL,
        "required_polyd": REQUIRED_POLYD_BALANCE
    }


@app.post("/api/launchpad/check-eligibility")
async def check_eligibility(user_wallet: str):
    """Check if user is eligible to launch (has enough $POLYD)"""
    balance = await check_polyd_balance(user_wallet)
    eligible = balance >= REQUIRED_POLYD_BALANCE

    return {
        "eligible": eligible,
        "balance": balance,
        "required": REQUIRED_POLYD_BALANCE,
        "missing": max(0, REQUIRED_POLYD_BALANCE - balance)
    }


@app.post("/api/launchpad/submit", response_model=LaunchResponse)
async def submit_launch(
    token_ticker: str = Form(...),
    agent_niche: str = Form(...),
    description: str = Form(...),
    custom_prompt: str = Form(""),
    profile_link: str = Form(...),
    wallet_address: str = Form(...),
    user_wallet: str = Form(...),
    twitter_username: str = Form(...),
    twitter_cookie: str = Form(...),
    twitter_email: str = Form(...),
    twitter_password: str = Form(...),
    twitter_totp_secret: str = Form(...),
    website: str = Form(None),
    telegram: str = Form(None),
    token_image: UploadFile = File(...)
):
    """Submit a new agent + token launch request"""

    # Check eligibility
    balance = await check_polyd_balance(user_wallet)
    if balance < REQUIRED_POLYD_BALANCE:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient $POLYD balance. Need {REQUIRED_POLYD_BALANCE:,}, have {balance:,.0f}"
        )

    # Read image
    image_data = await token_image.read()
    if len(image_data) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 5MB)")

    # Generate launch ID
    launch_id = hashlib.sha256(f"{user_wallet}{time.time()}".encode()).hexdigest()[:16]

    # Token name = ticker + "Agent"
    token_name = f"{token_ticker.replace('$', '')} Agent"

    # Store pending launch with encrypted credentials
    pending_launches[launch_id] = {
        "token_ticker": token_ticker.replace('$', '').upper(),
        "token_name": token_name,
        "agent_niche": agent_niche,
        "custom_prompt": custom_prompt,
        "profile_link": profile_link,
        "wallet_address": wallet_address,
        "description": description,
        "website": website,
        "telegram": telegram,
        "twitter_username": twitter_username.replace('@', ''),
        "twitter_cookie": twitter_cookie,
        # Store credentials for re-login (will be passed to agent)
        "twitter_credentials": {
            "username": twitter_username.replace('@', ''),
            "email": twitter_email,
            "password": twitter_password,
            "totp_secret": twitter_totp_secret
        },
        "user_wallet": user_wallet,
        "image_data": image_data,
        "status": "pending_payment",
        "created_at": time.time()
    }

    return LaunchResponse(
        success=True,
        message=f"Launch submitted. Send {LAUNCH_FEE_SOL} SOL to complete.",
        launch_id=launch_id,
        payment_address=LAUNCHPAD_WALLET_PUBLIC_KEY,
        payment_amount=LAUNCH_FEE_SOL
    )


@app.post("/api/launchpad/confirm/{launch_id}", response_model=LaunchResponse)
async def confirm_launch(launch_id: str):
    """Confirm payment, create token, and start agent"""

    if launch_id not in pending_launches:
        raise HTTPException(status_code=404, detail="Launch not found")

    launch = pending_launches[launch_id]

    if launch["status"] == "completed":
        return LaunchResponse(
            success=True,
            message="Token already created",
            launch_id=launch_id,
            token_mint=launch.get("token_mint"),
            tx_signature=launch.get("tx_signature")
        )

    if launch["status"] not in ["pending_payment", "failed"]:
        raise HTTPException(status_code=400, detail=f"Invalid status: {launch['status']}")

    # Update status
    launch["status"] = "creating"

    # Create token on pump.fun
    print(f"Creating token: {launch['token_name']} (${launch['token_ticker']})")

    twitter_handle = launch.get("twitter_username", "")

    result = await create_token_on_pumpfun(
        name=launch["token_name"],
        symbol=launch["token_ticker"],
        description=launch["description"],
        image_data=launch["image_data"],
        website=launch.get("website"),
        twitter=twitter_handle,
        telegram=launch.get("telegram"),
        dev_buy_sol=0
    )

    if result:
        launch["status"] = "completed"
        launch["token_mint"] = result["token_mint"]
        launch["tx_signature"] = result["tx_signature"]
        save_launches()

        # Start the Twitter agent with credentials for re-login
        credentials = launch.get("twitter_credentials")
        await start_agent(launch_id, launch, credentials=credentials)

        return LaunchResponse(
            success=True,
            message="Token created and agent started!",
            launch_id=launch_id,
            token_mint=result["token_mint"],
            tx_signature=result["tx_signature"]
        )
    else:
        launch["status"] = "failed"
        raise HTTPException(status_code=500, detail="Failed to create token. Check server logs.")


@app.get("/api/launchpad/status/{launch_id}")
async def get_launch_status(launch_id: str):
    """Get status of a launch"""

    if launch_id not in pending_launches:
        raise HTTPException(status_code=404, detail="Launch not found")

    launch = pending_launches[launch_id]

    return {
        "launch_id": launch_id,
        "status": launch["status"],
        "token_name": launch["token_name"],
        "token_ticker": launch["token_ticker"],
        "twitter_username": launch.get("twitter_username"),
        "token_mint": launch.get("token_mint"),
        "tx_signature": launch.get("tx_signature")
    }


@app.get("/api/launchpad/launches")
async def list_launches():
    """List all launches"""
    return {
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
    }


@app.get("/api/launchpad/stats")
async def get_stats():
    """Get launchpad statistics"""
    completed = sum(1 for l in pending_launches.values() if l["status"] == "completed")
    return {
        "total_launches": completed,
        "pending": sum(1 for l in pending_launches.values() if l["status"] == "pending_payment"),
        "failed": sum(1 for l in pending_launches.values() if l["status"] == "failed"),
        "running_agents": len(running_agents)
    }


@app.get("/api/launchpad/agents")
async def list_agents():
    """List all running agents (without sensitive data)"""
    safe_agents = []
    for agent in running_agents.values():
        safe_agent = {k: v for k, v in agent.items()
                      if k not in ["twitter_cookie", "twitter_password", "twitter_email",
                                   "twitter_username_cred", "twitter_totp_secret"]}
        # Add flag if credentials are stored
        safe_agent["has_credentials"] = bool(agent.get("twitter_password"))
        safe_agents.append(safe_agent)
    return {
        "total": len(running_agents),
        "agents": safe_agents
    }


@app.post("/api/launchpad/agents/{agent_id}/refresh")
async def refresh_agent(agent_id: str):
    """Re-login to Twitter using stored credentials"""
    if agent_id not in running_agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = running_agents[agent_id]

    # Check if agent has stored credentials
    if not agent.get("twitter_password"):
        raise HTTPException(
            status_code=400,
            detail="Agent has no stored credentials for re-login"
        )

    new_cookie = await refresh_agent_cookie(agent_id)

    if new_cookie:
        return {
            "success": True,
            "message": "Agent re-login successful",
            "agent_id": agent_id
        }
    else:
        raise HTTPException(
            status_code=400,
            detail="Re-login failed. Check credentials or Twitter account status."
        )


if __name__ == "__main__":
    import uvicorn
    print("Starting Polydictions Launchpad Server...")
    print(f"Launchpad wallet: {LAUNCHPAD_WALLET_PUBLIC_KEY}")
    print(f"Required $POLYD: {REQUIRED_POLYD_BALANCE:,}")
    print(f"Launch fee: {LAUNCH_FEE_SOL} SOL")
    print(f"TwitterAPI configured: {'Yes' if TWITTERAPI_KEY else 'No'}")
    uvicorn.run(app, host="0.0.0.0", port=5777)
