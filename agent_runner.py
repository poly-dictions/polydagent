"""
Agent Runner for Launchpad Agents
Runs all agents based on their niche and custom prompts
Uses TwitterAPI.io for posting (cookie-based auth)
Features:
- Posts about Polymarket events based on niche
- Responds to mentions with AI analysis
"""

import os
import json
import asyncio
import aiohttp
import random
import base64
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
from dotenv import load_dotenv
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

load_dotenv()

# Config
AGENTS_FILE = Path("agents.json")
ANSWERED_MENTIONS_FILE = Path("agent_answered_mentions.json")
POLYMARKET_API = "https://gamma-api.polymarket.com"
POLYFACTUAL_API_URL = "https://deep-research-api.thekid-solana.workers.dev/answer"
POLYFACTUAL_API_KEY = os.getenv("POLYFACTUAL_API_KEY", "wk_5rjSJp7hbIe6NBZ71-u4_2DUFOlDDaqj")

# FactsAI API (for mentions research)
FACTSAI_API_URL = "https://factsai.org/answer"
FACTSAI_API_KEY = os.getenv("FACTSAI_API_KEY", "polyd_0927dcb325cfd1a468417bedb260ff520d8a52afc9f9c06f25b4ac628416c03a")

# Claude API (for generating responses)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# Mention check interval (seconds)
MENTION_CHECK_INTERVAL = 60

# TwitterAPI.io config
TWITTERAPI_KEY = os.getenv("TWITTERAPI_KEY", "")
TWITTERAPI_BASE = "https://api.twitterapi.io"
TWITTER_PROXY = os.getenv("TWITTER_PROXY", "")

# Posting interval (hours)
POST_INTERVAL_HOURS = int(os.getenv("AGENT_POST_INTERVAL", "4"))

# Niche keywords for filtering Polymarket events
NICHE_KEYWORDS = {
    "crypto": [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol",
        "blockchain", "defi", "nft", "token", "coin", "binance", "coinbase",
        "memecoin", "doge", "shib", "xrp", "cardano", "polkadot"
    ],
    "politics": [
        "trump", "biden", "election", "president", "congress", "senate",
        "democrat", "republican", "vote", "poll", "governor", "mayor",
        "political", "white house", "cabinet", "impeach", "legislation"
    ],
    "sports": [
        "nba", "nfl", "mlb", "nhl", "soccer", "football", "basketball",
        "baseball", "hockey", "championship", "playoffs", "super bowl",
        "world series", "finals", "mvp", "team", "player", "coach",
        "premier league", "champions league", "world cup"
    ],
    "entertainment": [
        "movie", "film", "oscar", "emmy", "grammy", "netflix", "disney",
        "marvel", "actor", "actress", "director", "box office", "streaming",
        "celebrity", "hollywood", "tv show", "series", "album", "concert"
    ],
    "tech": [
        "ai", "artificial intelligence", "openai", "google", "apple",
        "microsoft", "amazon", "meta", "tesla", "elon musk", "spacex",
        "startup", "ipo", "tech", "software", "hardware", "chip", "nvidia"
    ],
    "finance": [
        "stock", "market", "fed", "interest rate", "inflation", "gdp",
        "recession", "economy", "s&p", "nasdaq", "dow", "bond", "treasury",
        "bank", "investment", "earnings", "revenue", "profit"
    ]
}


# Encryption (same as launchpad_server.py)
def get_encryption_key() -> bytes:
    secret = os.getenv("SESSION_SECRET", "default-secret-change-me")
    salt = b"polydictions-launchpad-salt"
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(secret.encode()))


ENCRYPTION_KEY = get_encryption_key()
fernet = Fernet(ENCRYPTION_KEY)


def decrypt_credential(encrypted_data: str) -> str:
    if not encrypted_data:
        return ""
    try:
        return fernet.decrypt(encrypted_data.encode()).decode()
    except Exception as e:
        print(f"Decryption error: {e}")
        return ""


def load_agents() -> Dict[str, Dict]:
    """Load agents from file and decrypt credentials"""
    if not AGENTS_FILE.exists():
        return {}

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
            return data
    except Exception as e:
        print(f"Error loading agents: {e}")
        return {}


def save_agents(agents: Dict[str, Dict]):
    """Save agents with encrypted credentials"""
    try:
        save_data = {}
        for aid, agent in agents.items():
            agent_data = {}
            for k, v in agent.items():
                if k in ["twitter_cookie", "twitter_password", "twitter_email",
                         "twitter_username_cred", "twitter_totp_secret"]:
                    agent_data[f"{k}_encrypted"] = fernet.encrypt(str(v).encode()).decode() if v else ""
                else:
                    agent_data[k] = v
            save_data[aid] = agent_data
        with open(AGENTS_FILE, "w") as f:
            json.dump(save_data, f, indent=2)
    except Exception as e:
        print(f"Error saving agents: {e}")


class TwitterAPI:
    """Post tweets via TwitterAPI.io using login cookies"""

    @staticmethod
    async def post_tweet(login_cookies: str, tweet_text: str) -> Optional[str]:
        """Post a tweet using login cookies"""
        url = f"{TWITTERAPI_BASE}/twitter/tweet/create_tweet_v2"
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": TWITTERAPI_KEY
        }
        data = {
            "tweet_text": tweet_text,
            "login_cookies": login_cookies,
            "proxy": TWITTER_PROXY
        }

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
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": TWITTERAPI_KEY
        }
        data = {
            "tweet_text": reply_text,
            "login_cookies": login_cookies,
            "proxy": TWITTER_PROXY,
            "reply_to_tweet_id": tweet_id
        }

        # Use note tweet for longer replies
        if len(reply_text) > 280:
            data["is_note_tweet"] = True

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data, timeout=60) as resp:
                    result = await resp.json()
                    if result.get("status") == "success":
                        return result.get("tweet_id") or "replied"
                    else:
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
    async def login(username: str, email: str, password: str, totp_secret: str) -> Optional[str]:
        """Login to Twitter and get new cookies"""
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
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data, timeout=120) as resp:
                    result = await resp.json()
                    if result.get("status") == "success":
                        return result.get("login_cookies") or result.get("login_cookie")
                    else:
                        print(f"Login failed: {result}")
                        return None
        except Exception as e:
            print(f"Login error: {e}")
            return None


class PolymarketScanner:
    """Scan Polymarket for opportunities filtered by niche"""

    SPAM_WORDS = [
        'up or down', 'higher or lower', 'above or below',
        'am-', 'pm-', 'am et', 'pm et', ':00', ':15', ':30', ':45'
    ]

    async def fetch_events(self, limit: int = 100) -> List[Dict]:
        """Fetch events sorted by volume"""
        url = f"{POLYMARKET_API}/events"
        params = {
            'limit': limit,
            'active': 'true',
            'closed': 'false',
            'order': 'volume',
            'ascending': 'false'
        }

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

        if volume < 50000:  # Lower threshold for niche markets
            return False

        if not markets or len(markets) > 3:
            return False

        market = markets[0]
        outcomes = market.get('outcomes', [])
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if len(outcomes) != 2:
            return False

        # Check odds (skip obvious markets)
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
    async def analyze_market(
        title: str,
        yes_odds: float,
        no_odds: float,
        volume: float,
        custom_prompt: str = "",
        niche: str = "general"
    ) -> Optional[Dict]:
        """Get AI analysis with custom personality"""

        # Base query
        query = f"""Analyze this Polymarket prediction market: [{title}]
Current odds: YES {yes_odds:.0f}% / NO {no_odds:.0f}%
Volume: ${volume:,.0f}

Provide a brief analysis with:
1) Key factors supporting YES
2) Key factors supporting NO
3) Your recommendation (YES or NO) with confidence level

Keep it concise - this is for a tweet."""

        # Add custom prompt if provided
        if custom_prompt:
            query += f"\n\nADDITIONAL INSTRUCTIONS: {custom_prompt}"

        headers = {
            'Content-Type': 'application/json',
            'X-API-Key': POLYFACTUAL_API_KEY
        }
        data = {
            'query': query,
            'text': True
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(POLYFACTUAL_API_URL, headers=headers, json=data, timeout=120) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        if result.get('success') and result.get('data'):
                            answer = result['data'].get('answer', '') or result['data'].get('text', '')
                            if isinstance(answer, dict):
                                answer = answer.get('text', str(answer))

                            # Clean up
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
        headers = {
            "Authorization": f"Bearer {FACTSAI_API_KEY}",
            "Content-Type": "application/json"
        }
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
            # Fallback: just use facts
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

        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        data = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}]
        }

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


class AgentRunner:
    """Run a single agent"""

    def __init__(self, agent_id: str, agent_data: Dict, answered_mentions: Set[str] = None):
        self.agent_id = agent_id
        self.agent = agent_data
        self.scanner = PolymarketScanner()
        self.posted_events: set = set()
        self.answered_mentions: Set[str] = answered_mentions or set()
        self.mention_start_time = int(datetime.now().timestamp())

    def format_volume(self, volume: float) -> str:
        if volume >= 1_000_000:
            return f"${volume/1_000_000:.1f}M"
        elif volume >= 1_000:
            return f"${volume/1_000:.0f}K"
        return f"${volume:.0f}"

    def create_tweet(self, market: Dict, reasoning: str) -> str:
        """Create tweet with market info and analysis"""

        # Niche-specific openers
        niche = self.agent.get("agent_niche", "general")
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

        # Truncate reasoning for tweet
        short_reasoning = reasoning[:200] + "..." if len(reasoning) > 200 else reasoning

        tweet = f"""{opener}

{market['title']}

YES {market['yes_odds']:.1f}% / NO {market['no_odds']:.1f}%
vol: {self.format_volume(market['volume'])}

{short_reasoning}

polymarket.com/event/{market['slug']}"""

        # Add token ticker mention
        ticker = self.agent.get("token_ticker", "")
        if ticker:
            tweet += f"\n\n${ticker} 🧡"

        return tweet[:280]  # Twitter limit

    async def find_and_post(self) -> bool:
        """Find a market and post about it"""
        niche = self.agent.get("agent_niche", "general")
        custom_prompt = self.agent.get("custom_prompt", "")
        twitter_cookie = self.agent.get("twitter_cookie")

        if not twitter_cookie:
            print(f"Agent {self.agent_id}: No twitter cookie")
            return False

        print(f"\n[{self.agent_id}] Scanning {niche} markets...")

        # Fetch and filter events
        events = await self.scanner.fetch_events(limit=100)
        events = self.scanner.filter_by_niche(events, niche)

        valid_markets = []
        for event in events:
            if self.scanner.is_valid_market(event, self.posted_events):
                market_data = self.scanner.parse_market_data(event)
                valid_markets.append(market_data)

        print(f"Found {len(valid_markets)} valid {niche} markets")

        if not valid_markets:
            print("No valid markets to post")
            return False

        # Pick random from top by volume
        top_markets = sorted(valid_markets, key=lambda x: x['volume'], reverse=True)[:10]
        market = random.choice(top_markets)

        print(f"Selected: {market['title']}")

        # Get AI analysis
        print("Getting AI analysis...")
        ai_result = await AIAnalyzer.analyze_market(
            market['title'], market['yes_odds'], market['no_odds'], market['volume'],
            custom_prompt=custom_prompt, niche=niche
        )

        reasoning = ai_result['reasoning'] if ai_result else "Interesting setup here."

        # Create tweet
        tweet_text = self.create_tweet(market, reasoning)
        print(f"\nTweet:\n{tweet_text}\n")

        # Post tweet
        tweet_id = await TwitterAPI.post_tweet(twitter_cookie, tweet_text)

        if tweet_id:
            print(f"✓ Posted! ID: {tweet_id}")
            self.posted_events.add(market['event_id'])
            return True
        else:
            print("✗ Failed to post")
            # Try re-login if failed
            await self.try_relogin()
            return False

    async def try_relogin(self):
        """Try to re-login if tweet failed"""
        username = self.agent.get("twitter_username_cred")
        email = self.agent.get("twitter_email")
        password = self.agent.get("twitter_password")
        totp = self.agent.get("twitter_totp_secret")

        if not all([username, email, password, totp]):
            print(f"Agent {self.agent_id}: Missing credentials for re-login")
            return

        print(f"Attempting re-login for @{username}...")
        new_cookie = await TwitterAPI.login(username, email, password, totp)

        if new_cookie:
            print("Re-login successful!")
            self.agent["twitter_cookie"] = new_cookie
            # Save will happen in main loop
        else:
            print("Re-login failed")
            self.agent["status"] = "cookie_expired"

    async def check_and_respond_mentions(self) -> int:
        """Check for mentions and respond to them"""
        username = self.agent.get("twitter_username")
        twitter_cookie = self.agent.get("twitter_cookie")
        custom_prompt = self.agent.get("custom_prompt", "")
        ticker = self.agent.get("token_ticker", "")

        if not username or not twitter_cookie:
            return 0

        print(f"[{self.agent_id}] Checking mentions for @{username}...")

        # Get mentions since last check
        mentions = await TwitterAPI.get_mentions(username, since_time=self.mention_start_time)

        # Filter to only real mentions (with @username in text)
        real_mentions = [
            m for m in mentions
            if f"@{username.lower()}" in m.get("text", "").lower()
            and m.get("id") not in self.answered_mentions
            and m.get("author", {}).get("userName", "").lower() != username.lower()
        ]

        if not real_mentions:
            print(f"[{self.agent_id}] No new mentions")
            return 0

        print(f"[{self.agent_id}] Found {len(real_mentions)} new mentions!")

        answered_count = 0
        for mention in real_mentions[:5]:  # Limit to 5 per check
            tweet_id = mention.get("id")
            text = mention.get("text", "")
            author = mention.get("author", {}).get("userName", "unknown")

            # Extract question (remove @mentions)
            question = re.sub(r'@\w+\s*', '', text).strip()
            if len(question) < 5:
                continue

            print(f"[{self.agent_id}] Mention from @{author}: {question[:50]}...")

            # Get facts and generate reply
            facts = await AIAnalyzer.get_facts(question)
            reply = await AIAnalyzer.generate_reply(
                question=question,
                facts=facts or "",
                custom_prompt=custom_prompt,
                ticker=ticker
            )

            if not reply:
                print(f"[{self.agent_id}] Failed to generate reply")
                continue

            print(f"[{self.agent_id}] Reply: {reply[:100]}...")

            # Post reply
            result = await TwitterAPI.post_reply(twitter_cookie, tweet_id, reply)

            if result:
                print(f"[{self.agent_id}] ✓ Replied to @{author}!")
                self.answered_mentions.add(tweet_id)
                answered_count += 1
            else:
                print(f"[{self.agent_id}] ✗ Failed to reply")
                # Try re-login if failed
                await self.try_relogin()

            # Small delay between replies
            await asyncio.sleep(5)

        # Update start time for next check
        self.mention_start_time = int(datetime.now().timestamp())

        return answered_count


async def run_all_agents():
    """Run all agents continuously with posting and mention responses"""
    print("=" * 50)
    print("AGENT RUNNER - Launchpad Agents")
    print(f"Post interval: {POST_INTERVAL_HOURS} hours")
    print(f"Mention check interval: {MENTION_CHECK_INTERVAL} seconds")
    print("=" * 50)

    # Track state
    last_post_time = 0
    answered_mentions = load_answered_mentions()
    agent_runners: Dict[str, AgentRunner] = {}

    while True:
        try:
            agents = load_agents()

            if not agents:
                print("No agents found. Waiting...")
                await asyncio.sleep(60)
                continue

            current_time = datetime.now().timestamp()
            should_post = (current_time - last_post_time) >= (POST_INTERVAL_HOURS * 3600)

            # Initialize runners for new agents
            for agent_id, agent_data in agents.items():
                if agent_id not in agent_runners:
                    agent_answered = answered_mentions.get(agent_id, set())
                    agent_runners[agent_id] = AgentRunner(agent_id, agent_data, agent_answered)
                else:
                    # Update agent data
                    agent_runners[agent_id].agent = agent_data

            # Process each agent
            for agent_id, runner in list(agent_runners.items()):
                if runner.agent.get("status") == "cookie_expired":
                    print(f"[{agent_id}] Skipped - cookie expired")
                    continue

                if runner.agent.get("status") != "running":
                    continue

                try:
                    # Check mentions (always)
                    mentions_answered = await runner.check_and_respond_mentions()
                    if mentions_answered > 0:
                        answered_mentions[agent_id] = runner.answered_mentions
                        save_answered_mentions(answered_mentions)

                    # Post new content (only when it's time)
                    if should_post:
                        await runner.find_and_post()

                    # Update agent data (in case of re-login)
                    agents[agent_id] = runner.agent

                    # Small delay between agents
                    await asyncio.sleep(5)

                except Exception as e:
                    print(f"[{agent_id}] Error: {e}")
                    import traceback
                    traceback.print_exc()

            # Save agents (with any updated cookies)
            save_agents(agents)

            if should_post:
                last_post_time = current_time
                print(f"\nNext post in {POST_INTERVAL_HOURS} hours...")

            print(f"Next mention check in {MENTION_CHECK_INTERVAL} seconds...")
            print("-" * 50)

        except Exception as e:
            print(f"Main loop error: {e}")
            import traceback
            traceback.print_exc()

        await asyncio.sleep(MENTION_CHECK_INTERVAL)


if __name__ == "__main__":
    asyncio.run(run_all_agents())
