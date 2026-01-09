"""
Pandora Market Creation Agent

Monitors Twitter mentions and creates prediction markets on Pandora (Sonic blockchain)
using AI to parse user requests.
"""

import os
import time
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from web3 import Web3
from anthropic import Anthropic
import requests

load_dotenv()

# Configuration
SONIC_RPC_URL = os.getenv("SONIC_RPC_URL", "https://rpc.soniclabs.com")
PANDORA_ORACLE_ADDRESS = os.getenv("PANDORA_ORACLE_ADDRESS")
PANDORA_MARKET_FACTORY_ADDRESS = os.getenv("PANDORA_MARKET_FACTORY_ADDRESS")
PANDORA_PRIVATE_KEY = os.getenv("PANDORA_PRIVATE_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TWITTERAPI_KEY = os.getenv("TWITTERAPI_KEY")

# Contract ABIs (minimal interfaces for the functions we need)
PREDICTION_ORACLE_ABI = [
    {
        "inputs": [
            {"name": "question", "type": "string"},
            {"name": "deadline", "type": "uint256"},
            {"name": "details", "type": "string"}
        ],
        "name": "createPoll",
        "outputs": [{"name": "pollId", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

MARKET_FACTORY_ABI = [
    {
        "inputs": [
            {"name": "pollId", "type": "uint256"},
            {"name": "initialLiquidity", "type": "uint256"}
        ],
        "name": "createAMMMarket",
        "outputs": [{"name": "marketAddress", "type": "address"}],
        "stateMutability": "payable",
        "type": "function"
    }
]


class PandoraAgent:
    def __init__(self):
        """Initialize the Pandora market creation agent"""
        # Web3 setup
        self.w3 = Web3(Web3.HTTPProvider(SONIC_RPC_URL))
        if not self.w3.is_connected():
            raise Exception("Failed to connect to Sonic RPC")

        # Load account from private key
        if not PANDORA_PRIVATE_KEY:
            raise Exception("PANDORA_PRIVATE_KEY not set in .env")

        self.account = self.w3.eth.account.from_key(PANDORA_PRIVATE_KEY)
        print(f"Agent wallet: {self.account.address}")

        # Contract instances
        self.oracle = self.w3.eth.contract(
            address=Web3.to_checksum_address(PANDORA_ORACLE_ADDRESS),
            abi=PREDICTION_ORACLE_ABI
        )
        self.factory = self.w3.eth.contract(
            address=Web3.to_checksum_address(PANDORA_MARKET_FACTORY_ADDRESS),
            abi=MARKET_FACTORY_ABI
        )

        # Claude AI client
        self.claude = Anthropic(api_key=ANTHROPIC_API_KEY)

        # Database for tracking created markets
        self.init_database()

        print("Pandora Agent initialized successfully")

    def init_database(self):
        """Initialize SQLite database for tracking markets"""
        conn = sqlite3.connect("pandora_markets.db")
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS markets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mention_id TEXT UNIQUE,
                mention_text TEXT,
                mention_author TEXT,
                question TEXT,
                deadline INTEGER,
                poll_id INTEGER,
                market_address TEXT,
                tx_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending'
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_status (
                id INTEGER PRIMARY KEY,
                running BOOLEAN DEFAULT 0,
                last_check TIMESTAMP,
                total_markets INTEGER DEFAULT 0,
                successful_markets INTEGER DEFAULT 0,
                failed_markets INTEGER DEFAULT 0
            )
        """)

        # Initialize status if not exists
        cursor.execute("INSERT OR IGNORE INTO agent_status (id, running) VALUES (1, 0)")

        conn.commit()
        conn.close()

    def parse_market_request(self, mention_text: str) -> Optional[Dict]:
        """Use Claude AI to parse a Twitter mention into market parameters"""
        prompt = f"""Parse this Twitter mention to extract prediction market parameters.

Mention text: "{mention_text}"

Extract:
1. Question: Clear yes/no question for the prediction market
2. Deadline: When should the market resolve? Extract date/time or relative time (e.g., "in 30 days")
3. Details: Any additional context or description

Return JSON format:
{{
    "question": "Clear yes/no question?",
    "deadline_description": "parsed deadline description",
    "deadline_days_from_now": number of days from now,
    "details": "additional context",
    "confidence": 0-100 (how confident you are this is a valid market request)
}}

If this doesn't look like a market creation request, set confidence to 0."""

        try:
            response = self.claude.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )

            result_text = response.content[0].text

            # Extract JSON from response
            json_start = result_text.find("{")
            json_end = result_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(result_text[json_start:json_end])

                if result.get("confidence", 0) < 60:
                    print(f"Low confidence ({result.get('confidence')}%) - skipping")
                    return None

                return result
            else:
                print("Failed to extract JSON from Claude response")
                return None

        except Exception as e:
            print(f"Error parsing with Claude: {e}")
            return None

    def create_poll(self, question: str, deadline_days: int, details: str) -> Optional[int]:
        """Create a poll on Pandora PredictionOracle contract"""
        try:
            # Calculate deadline timestamp
            deadline_timestamp = int((datetime.now() + timedelta(days=deadline_days)).timestamp())

            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.account.address)

            # Estimate gas
            gas_estimate = self.oracle.functions.createPoll(
                question,
                deadline_timestamp,
                details
            ).estimate_gas({'from': self.account.address})

            # Build transaction
            tx = self.oracle.functions.createPoll(
                question,
                deadline_timestamp,
                details
            ).build_transaction({
                'from': self.account.address,
                'nonce': nonce,
                'gas': int(gas_estimate * 1.2),  # 20% buffer
                'gasPrice': self.w3.eth.gas_price
            })

            # Sign and send
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

            print(f"Poll creation tx sent: {tx_hash.hex()}")

            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt['status'] == 1:
                # Parse pollId from logs
                # Assuming the first log contains the pollId
                if receipt['logs']:
                    poll_id = int(receipt['logs'][0]['topics'][1].hex(), 16)
                    print(f"Poll created successfully! Poll ID: {poll_id}")
                    return poll_id
                else:
                    print("Warning: No logs found, cannot extract pollId")
                    return None
            else:
                print("Transaction failed")
                return None

        except Exception as e:
            print(f"Error creating poll: {e}")
            return None

    def create_market(self, poll_id: int, initial_liquidity_s: float = 0.01) -> Optional[str]:
        """Create an AMM market for a poll"""
        try:
            # Convert S to wei (assuming 18 decimals)
            liquidity_wei = self.w3.to_wei(initial_liquidity_s, 'ether')

            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.account.address)

            # Estimate gas
            gas_estimate = self.factory.functions.createAMMMarket(
                poll_id,
                liquidity_wei
            ).estimate_gas({
                'from': self.account.address,
                'value': liquidity_wei
            })

            # Build transaction
            tx = self.factory.functions.createAMMMarket(
                poll_id,
                liquidity_wei
            ).build_transaction({
                'from': self.account.address,
                'nonce': nonce,
                'gas': int(gas_estimate * 1.2),
                'gasPrice': self.w3.eth.gas_price,
                'value': liquidity_wei
            })

            # Sign and send
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

            print(f"Market creation tx sent: {tx_hash.hex()}")

            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt['status'] == 1:
                # Parse market address from logs
                if receipt['logs']:
                    market_address = '0x' + receipt['logs'][0]['topics'][1].hex()[-40:]
                    print(f"Market created successfully! Address: {market_address}")
                    return market_address
                else:
                    print("Warning: No logs found, cannot extract market address")
                    return None
            else:
                print("Transaction failed")
                return None

        except Exception as e:
            print(f"Error creating market: {e}")
            return None

    def get_twitter_mentions(self) -> List[Dict]:
        """Fetch recent Twitter mentions using TwitterAPI.io"""
        if not TWITTERAPI_KEY:
            print("TWITTERAPI_KEY not set")
            return []

        try:
            # Get our user ID first (placeholder - would need to be implemented)
            # For now, return empty list as we need proper Twitter integration
            # In a real implementation, this would use TwitterAPI.io to fetch mentions
            return []
        except Exception as e:
            print(f"Error fetching Twitter mentions: {e}")
            return []

    def respond_to_mention(self, mention_id: str, market_address: str, question: str):
        """Respond to a Twitter mention with the market link"""
        market_url = f"https://thisispandora.netlify.app/market/{market_address}"
        response_text = f"✅ Market created!\n\n{question}\n\n🔗 {market_url}"

        print(f"Response to {mention_id}: {response_text}")

        # TODO: Implement actual Twitter response using TwitterAPI.io
        # For now, just log it

    def process_mention(self, mention: Dict) -> bool:
        """Process a single mention and create a market if valid"""
        mention_id = mention.get("id")
        mention_text = mention.get("text", "")
        mention_author = mention.get("author", "")

        print(f"\n{'='*60}")
        print(f"Processing mention from @{mention_author}")
        print(f"Text: {mention_text}")
        print(f"{'='*60}")

        # Check if already processed
        conn = sqlite3.connect("pandora_markets.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM markets WHERE mention_id = ?", (mention_id,))
        if cursor.fetchone():
            print("Already processed - skipping")
            conn.close()
            return False

        # Parse the mention with AI
        parsed = self.parse_market_request(mention_text)
        if not parsed:
            print("Failed to parse or not a valid market request")
            conn.close()
            return False

        print(f"Parsed: {json.dumps(parsed, indent=2)}")

        # Create the poll
        poll_id = self.create_poll(
            parsed["question"],
            parsed["deadline_days_from_now"],
            parsed.get("details", "")
        )

        if not poll_id:
            cursor.execute("""
                INSERT INTO markets (mention_id, mention_text, mention_author, status)
                VALUES (?, ?, ?, 'failed_poll')
            """, (mention_id, mention_text, mention_author))
            conn.commit()
            conn.close()
            return False

        # Create the market
        market_address = self.create_market(poll_id)

        if not market_address:
            cursor.execute("""
                INSERT INTO markets (mention_id, mention_text, mention_author, poll_id, status)
                VALUES (?, ?, ?, ?, 'failed_market')
            """, (mention_id, mention_text, mention_author, poll_id))
            conn.commit()
            conn.close()
            return False

        # Save to database
        cursor.execute("""
            INSERT INTO markets (
                mention_id, mention_text, mention_author,
                question, deadline, poll_id, market_address, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'success')
        """, (
            mention_id, mention_text, mention_author,
            parsed["question"], parsed["deadline_days_from_now"],
            poll_id, market_address
        ))

        # Update stats
        cursor.execute("""
            UPDATE agent_status
            SET total_markets = total_markets + 1,
                successful_markets = successful_markets + 1
            WHERE id = 1
        """)

        conn.commit()
        conn.close()

        # Respond to the mention
        self.respond_to_mention(mention_id, market_address, parsed["question"])

        print(f"\n✅ Successfully created market!")
        print(f"Poll ID: {poll_id}")
        print(f"Market: {market_address}")
        print(f"URL: https://thisispandora.netlify.app/market/{market_address}")

        return True

    def run_once(self):
        """Process mentions once"""
        print("\n" + "="*60)
        print("Checking for new mentions...")
        print("="*60)

        mentions = self.get_twitter_mentions()

        if not mentions:
            print("No new mentions found")
            return

        print(f"Found {len(mentions)} mentions to process")

        success_count = 0
        for mention in mentions:
            try:
                if self.process_mention(mention):
                    success_count += 1
            except Exception as e:
                print(f"Error processing mention: {e}")

        # Update last check time
        conn = sqlite3.connect("pandora_markets.db")
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE agent_status SET last_check = ? WHERE id = 1",
            (datetime.now().isoformat(),)
        )
        conn.commit()
        conn.close()

        print(f"\nProcessed {len(mentions)} mentions, {success_count} markets created")

    def run_continuous(self, check_interval: int = 300):
        """Run the agent continuously, checking every N seconds"""
        print(f"\n{'='*60}")
        print("Starting Pandora Agent in continuous mode")
        print(f"Check interval: {check_interval} seconds")
        print(f"{'='*60}\n")

        # Update status
        conn = sqlite3.connect("pandora_markets.db")
        cursor = conn.cursor()
        cursor.execute("UPDATE agent_status SET running = 1 WHERE id = 1")
        conn.commit()
        conn.close()

        try:
            while True:
                try:
                    self.run_once()
                except Exception as e:
                    print(f"Error in main loop: {e}")

                print(f"\nWaiting {check_interval} seconds until next check...")
                time.sleep(check_interval)
        except KeyboardInterrupt:
            print("\n\nStopping agent...")
        finally:
            # Update status
            conn = sqlite3.connect("pandora_markets.db")
            cursor = conn.cursor()
            cursor.execute("UPDATE agent_status SET running = 0 WHERE id = 1")
            conn.commit()
            conn.close()
            print("Agent stopped")

    def test_create_market(self, question: str, deadline_days: int = 30, details: str = ""):
        """Test market creation with manual input"""
        print(f"\n{'='*60}")
        print("TEST MODE: Creating market manually")
        print(f"Question: {question}")
        print(f"Deadline: {deadline_days} days from now")
        print(f"Details: {details}")
        print(f"{'='*60}\n")

        # Create poll
        poll_id = self.create_poll(question, deadline_days, details)
        if not poll_id:
            print("Failed to create poll")
            return

        # Create market
        market_address = self.create_market(poll_id)
        if not market_address:
            print("Failed to create market")
            return

        print(f"\n✅ Market created successfully!")
        print(f"Poll ID: {poll_id}")
        print(f"Market Address: {market_address}")
        print(f"URL: https://thisispandora.netlify.app/market/{market_address}")

        # Save to database
        conn = sqlite3.connect("pandora_markets.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO markets (
                mention_id, mention_text, mention_author,
                question, deadline, poll_id, market_address, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'success')
        """, (
            f"test_{int(time.time())}", question, "manual_test",
            question, deadline_days, poll_id, market_address
        ))
        cursor.execute("""
            UPDATE agent_status
            SET total_markets = total_markets + 1,
                successful_markets = successful_markets + 1
            WHERE id = 1
        """)
        conn.commit()
        conn.close()


def main():
    """Main entry point"""
    print("Pandora Market Creation Agent")
    print("="*60)

    try:
        agent = PandoraAgent()
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        return

    # Check wallet balance
    balance = agent.w3.eth.get_balance(agent.account.address)
    balance_s = agent.w3.from_wei(balance, 'ether')
    print(f"Wallet balance: {balance_s:.4f} S")

    if balance_s < 0.1:
        print(f"\n⚠️  WARNING: Low balance! You have {balance_s:.4f} S")
        print("You may need more S tokens to create markets")
        print("Each market requires a small amount of S for gas + initial liquidity")

    print("\nWhat would you like to do?")
    print("1. Run once (check mentions now)")
    print("2. Run continuously (check every 5 minutes)")
    print("3. Test market creation (manual input)")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        agent.run_once()
    elif choice == "2":
        agent.run_continuous(check_interval=300)
    elif choice == "3":
        question = input("Enter question: ").strip()
        if not question:
            print("Question required")
            return
        days = input("Enter deadline in days (default 30): ").strip()
        deadline_days = int(days) if days else 30
        details = input("Enter details (optional): ").strip()
        agent.test_create_market(question, deadline_days, details)
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
