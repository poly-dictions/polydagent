"""
$POLYD Token Payment System
- Solana SPL Token payments
- Burn mechanism (each payment burns tokens)
- Balance tracking per user
"""
import asyncio
import aiohttp
import sqlite3
import json
from datetime import datetime
from pathlib import Path

# $POLYD Token Config
POLYD_TOKEN_MINT = "iATcGSt9DhJF9ZiJ6dmR153N7bW2G4J9dSSDxWSpump"
HELIUS_RPC = "https://mainnet.helius-rpc.com/?api-key=fcdeff60-a054-4fc0-bdd5-0131925677ce"

# Treasury wallet (receives deposits, burns tokens)
TREASURY_WALLET = "5z5sxxhxWTGBy76BX3HzVmX54gc2ZSKKx5vm5EnoPRrA"

# Pricing (in $POLYD tokens)
PRICES = {
    'whales': 200,   # /whales command
    'flow': 200,     # /flow command
    'orders': 200,   # /orders command
}

# Subscription pricing
SUBSCRIPTIONS = {
    'arb_week': {'price': 7000, 'days': 7, 'name': 'Arb Alerts (1 week)'},
    'arb_month': {'price': 30000, 'days': 30, 'name': 'Arb Alerts (1 month)'},
}

DB_PATH = Path(__file__).parent / 'payments.db'


def init_db():
    """Initialize payments database"""
    conn = sqlite3.connect(DB_PATH)
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS user_balances (
            user_id INTEGER PRIMARY KEY,
            balance INTEGER DEFAULT 0,
            total_deposited INTEGER DEFAULT 0,
            total_burned INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            type TEXT,  -- 'deposit' | 'burn'
            amount INTEGER,
            command TEXT,
            tx_signature TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS pending_deposits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            expected_amount INTEGER,
            memo TEXT UNIQUE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            confirmed BOOLEAN DEFAULT FALSE
        );

        CREATE TABLE IF NOT EXISTS used_transactions (
            tx_signature TEXT PRIMARY KEY,
            user_id INTEGER,
            amount INTEGER,
            used_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            sub_type TEXT,
            expires_at DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS arb_alerts (
            user_id INTEGER PRIMARY KEY,
            enabled BOOLEAN DEFAULT TRUE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    conn.commit()
    conn.close()


class PaymentSystem:
    """Handle $POLYD token payments with burn mechanism"""

    def __init__(self):
        init_db()

    def get_balance(self, user_id: int) -> int:
        """Get user's $POLYD balance"""
        conn = sqlite3.connect(DB_PATH)
        result = conn.execute(
            'SELECT balance FROM user_balances WHERE user_id = ?',
            (user_id,)
        ).fetchone()
        conn.close()
        return result[0] if result else 0

    def get_user_stats(self, user_id: int) -> dict:
        """Get user's payment stats"""
        conn = sqlite3.connect(DB_PATH)
        result = conn.execute(
            'SELECT balance, total_deposited, total_burned FROM user_balances WHERE user_id = ?',
            (user_id,)
        ).fetchone()
        conn.close()

        if result:
            return {
                'balance': result[0],
                'total_deposited': result[1],
                'total_burned': result[2]
            }
        return {'balance': 0, 'total_deposited': 0, 'total_burned': 0}

    def check_balance(self, user_id: int, command: str) -> tuple[bool, int, int]:
        """
        Check if user has enough balance for command
        Returns: (has_enough, current_balance, required_amount)
        """
        required = PRICES.get(command, 0)
        balance = self.get_balance(user_id)
        return (balance >= required, balance, required)

    def deduct_and_burn(self, user_id: int, command: str) -> bool:
        """
        Deduct tokens from balance and record burn
        Returns True if successful
        """
        amount = PRICES.get(command, 0)
        if amount == 0:
            return True

        conn = sqlite3.connect(DB_PATH)
        try:
            # Check balance
            balance = conn.execute(
                'SELECT balance FROM user_balances WHERE user_id = ?',
                (user_id,)
            ).fetchone()

            if not balance or balance[0] < amount:
                conn.close()
                return False

            # Deduct balance
            conn.execute(
                'UPDATE user_balances SET balance = balance - ?, total_burned = total_burned + ? WHERE user_id = ?',
                (amount, amount, user_id)
            )

            # Record burn transaction
            conn.execute(
                'INSERT INTO transactions (user_id, type, amount, command) VALUES (?, ?, ?, ?)',
                (user_id, 'burn', amount, command)
            )

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            conn.close()
            raise e

    def is_tx_used(self, tx_signature: str) -> bool:
        """Check if transaction was already used"""
        conn = sqlite3.connect(DB_PATH)
        result = conn.execute(
            'SELECT 1 FROM used_transactions WHERE tx_signature = ?',
            (tx_signature,)
        ).fetchone()
        conn.close()
        return result is not None

    def mark_tx_used(self, tx_signature: str, user_id: int, amount: int):
        """Mark transaction as used"""
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            'INSERT OR IGNORE INTO used_transactions (tx_signature, user_id, amount) VALUES (?, ?, ?)',
            (tx_signature, user_id, amount)
        )
        conn.commit()
        conn.close()

    def add_balance(self, user_id: int, amount: int, tx_signature: str = None) -> int:
        """
        Add tokens to user balance (after deposit confirmed)
        Returns new balance, or -1 if tx already used
        """
        # Check if transaction already used
        if tx_signature and self.is_tx_used(tx_signature):
            return -1  # Already used

        conn = sqlite3.connect(DB_PATH)

        # Create user if not exists
        conn.execute('''
            INSERT OR IGNORE INTO user_balances (user_id, balance, total_deposited)
            VALUES (?, 0, 0)
        ''', (user_id,))

        # Add balance
        conn.execute(
            'UPDATE user_balances SET balance = balance + ?, total_deposited = total_deposited + ? WHERE user_id = ?',
            (amount, amount, user_id)
        )

        # Record deposit
        conn.execute(
            'INSERT INTO transactions (user_id, type, amount, tx_signature) VALUES (?, ?, ?, ?)',
            (user_id, 'deposit', amount, tx_signature)
        )

        # Mark transaction as used
        if tx_signature:
            conn.execute(
                'INSERT OR IGNORE INTO used_transactions (tx_signature, user_id, amount) VALUES (?, ?, ?)',
                (tx_signature, user_id, amount)
            )

        # Get new balance
        new_balance = conn.execute(
            'SELECT balance FROM user_balances WHERE user_id = ?',
            (user_id,)
        ).fetchone()[0]

        conn.commit()
        conn.close()
        return new_balance

    def generate_deposit_memo(self, user_id: int) -> str:
        """Generate unique memo for deposit tracking"""
        import hashlib
        data = f"{user_id}_{datetime.now().timestamp()}"
        return hashlib.sha256(data.encode()).hexdigest()[:8]

    def create_pending_deposit(self, user_id: int, amount: int) -> str:
        """Create pending deposit record, return memo"""
        memo = self.generate_deposit_memo(user_id)

        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            'INSERT INTO pending_deposits (user_id, expected_amount, memo) VALUES (?, ?, ?)',
            (user_id, amount, memo)
        )
        conn.commit()
        conn.close()

        return memo

    async def verify_deposit(self, tx_signature: str) -> dict:
        """
        Verify a deposit transaction using Helius RPC
        Returns: {success, user_id, amount} or {success: False, error}
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTransaction",
            "params": [
                tx_signature,
                {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}
            ]
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(HELIUS_RPC, json=payload, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        tx = data.get('result')

                        if not tx:
                            return {'success': False, 'error': 'Transaction not found'}

                        # Parse SPL token transfer
                        # Look for transfer to treasury with $POLYD token
                        instructions = tx.get('transaction', {}).get('message', {}).get('instructions', [])

                        for ix in instructions:
                            if ix.get('program') == 'spl-token':
                                parsed = ix.get('parsed', {})
                                tx_type = parsed.get('type', '')

                                # Support both 'transfer' and 'transferChecked'
                                if tx_type in ('transfer', 'transferChecked'):
                                    info = parsed.get('info', {})

                                    # Verify it's $POLYD token
                                    mint = info.get('mint', '')
                                    if mint and mint != POLYD_TOKEN_MINT:
                                        continue  # Wrong token

                                    # Get amount (different format for transferChecked)
                                    if tx_type == 'transferChecked':
                                        token_amount = info.get('tokenAmount', {})
                                        amount = int(token_amount.get('amount', 0))
                                        # Convert to whole tokens (6 decimals)
                                        decimals = token_amount.get('decimals', 6)
                                        amount = amount // (10 ** decimals)
                                    else:
                                        amount = int(info.get('amount', 0))

                                    # Verify destination is treasury (check postTokenBalances)
                                    meta = tx.get('meta', {})
                                    post_balances = meta.get('postTokenBalances', [])

                                    treasury_received = False
                                    for bal in post_balances:
                                        if bal.get('owner') == TREASURY_WALLET and bal.get('mint') == POLYD_TOKEN_MINT:
                                            treasury_received = True
                                            break

                                    if not treasury_received:
                                        return {'success': False, 'error': 'Transfer not to treasury wallet'}

                                    return {
                                        'success': True,
                                        'amount': amount,
                                        'tx_signature': tx_signature
                                    }

                        return {'success': False, 'error': 'No valid $POLYD transfer found'}
                    else:
                        return {'success': False, 'error': f'RPC error: {resp.status}'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_recent_transactions(self, user_id: int, limit: int = 10) -> list:
        """Get user's recent transactions"""
        conn = sqlite3.connect(DB_PATH)
        txs = conn.execute('''
            SELECT type, amount, command, tx_signature, created_at
            FROM transactions
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (user_id, limit)).fetchall()
        conn.close()

        return [
            {
                'type': tx[0],
                'amount': tx[1],
                'command': tx[2],
                'tx_signature': tx[3],
                'created_at': tx[4]
            }
            for tx in txs
        ]

    # ============ SUBSCRIPTIONS ============

    def subscribe(self, user_id: int, sub_type: str) -> bool:
        """
        Subscribe user to a service (arb_week, arb_month)
        Returns True if successful, False if insufficient balance
        """
        if sub_type not in SUBSCRIPTIONS:
            return False

        sub_info = SUBSCRIPTIONS[sub_type]
        price = sub_info['price']
        days = sub_info['days']

        conn = sqlite3.connect(DB_PATH)

        # Check balance
        balance = conn.execute(
            'SELECT balance FROM user_balances WHERE user_id = ?',
            (user_id,)
        ).fetchone()

        if not balance or balance[0] < price:
            conn.close()
            return False

        # Deduct balance
        conn.execute(
            'UPDATE user_balances SET balance = balance - ?, total_burned = total_burned + ? WHERE user_id = ?',
            (price, price, user_id)
        )

        # Record burn
        conn.execute(
            'INSERT INTO transactions (user_id, type, amount, command) VALUES (?, ?, ?, ?)',
            (user_id, 'burn', price, f'subscription:{sub_type}')
        )

        # Check if already has subscription
        existing = conn.execute(
            "SELECT expires_at FROM subscriptions WHERE user_id = ? AND sub_type LIKE 'arb%' AND expires_at > datetime('now')",
            (user_id,)
        ).fetchone()

        if existing:
            # Extend existing subscription
            conn.execute(
                "UPDATE subscriptions SET expires_at = datetime(expires_at, '+' || ? || ' days') WHERE user_id = ? AND sub_type LIKE 'arb%'",
                (days, user_id)
            )
        else:
            # Create new subscription
            conn.execute(
                "INSERT INTO subscriptions (user_id, sub_type, expires_at) VALUES (?, ?, datetime('now', '+' || ? || ' days'))",
                (user_id, sub_type, days)
            )

        conn.commit()
        conn.close()
        return True

    def get_subscription(self, user_id: int, sub_type_prefix: str = 'arb') -> dict:
        """Get user's subscription status"""
        conn = sqlite3.connect(DB_PATH)
        sub = conn.execute(
            "SELECT sub_type, expires_at FROM subscriptions WHERE user_id = ? AND sub_type LIKE ? AND expires_at > datetime('now') ORDER BY expires_at DESC LIMIT 1",
            (user_id, f'{sub_type_prefix}%')
        ).fetchone()
        conn.close()

        if sub:
            return {
                'active': True,
                'type': sub[0],
                'expires_at': sub[1]
            }
        return {'active': False}

    def get_all_arb_subscribers(self) -> list:
        """Get all users with active arb subscription (paid)"""
        conn = sqlite3.connect(DB_PATH)
        subs = conn.execute(
            "SELECT DISTINCT user_id FROM subscriptions WHERE sub_type LIKE 'arb%' AND expires_at > datetime('now')"
        ).fetchall()
        conn.close()
        return [s[0] for s in subs]

    # ============ FREE ARB ALERTS ============

    def enable_arb_alerts(self, user_id: int) -> bool:
        """Enable free arb alerts for user"""
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            'INSERT OR REPLACE INTO arb_alerts (user_id, enabled) VALUES (?, TRUE)',
            (user_id,)
        )
        conn.commit()
        conn.close()
        return True

    def disable_arb_alerts(self, user_id: int) -> bool:
        """Disable arb alerts for user"""
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            'DELETE FROM arb_alerts WHERE user_id = ?',
            (user_id,)
        )
        conn.commit()
        conn.close()
        return True

    def is_arb_alerts_enabled(self, user_id: int) -> bool:
        """Check if user has arb alerts enabled"""
        conn = sqlite3.connect(DB_PATH)
        result = conn.execute(
            'SELECT enabled FROM arb_alerts WHERE user_id = ?',
            (user_id,)
        ).fetchone()
        conn.close()
        return result is not None and result[0]

    def get_all_arb_alert_users(self) -> list:
        """Get all users with arb alerts enabled (free)"""
        conn = sqlite3.connect(DB_PATH)
        users = conn.execute(
            'SELECT user_id FROM arb_alerts WHERE enabled = TRUE'
        ).fetchall()
        conn.close()
        return [u[0] for u in users]


# Global instance
payments = PaymentSystem()


def format_prices() -> str:
    """Format prices for display"""
    descriptions = {
        'whales': 'Top traders leaderboard',
        'flow': 'Order flow sentiment',
        'orders': 'Large orders feed',
    }
    lines = ["<b>💰 $POLYD Command Prices</b>\n"]
    for cmd, price in PRICES.items():
        desc = descriptions.get(cmd, '')
        lines.append(f"/{cmd} — <b>{price}</b> $POLYD")
        if desc:
            lines.append(f"   <i>{desc}</i>")
    lines.append(f"\n🔥 <i>All payments are burned</i>")
    return "\n".join(lines)


def format_balance(user_id: int) -> str:
    """Format user balance for display"""
    stats = payments.get_user_stats(user_id)
    return f"""💎 <b>Your $POLYD Balance</b>

Balance: <b>{stats['balance']:,}</b> $POLYD
Total deposited: {stats['total_deposited']:,}
Total burned: {stats['total_burned']:,} 🔥

/deposit — Add more tokens
/prices — View command prices"""


def format_deposit_instructions(memo: str) -> str:
    """Format deposit instructions"""
    return f"""💳 <b>Deposit $POLYD</b>

Send $POLYD tokens to:
<code>{TREASURY_WALLET}</code>

Token: $POLYD
<code>{POLYD_TOKEN_MINT}</code>

After sending, use:
/confirm &lt;tx_signature&gt;

<i>Tokens are burned on each use 🔥</i>"""
