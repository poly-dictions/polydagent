"""
Polymarket Trading System
Automated trading with risk management and P&L tracking
"""

import os
import asyncio
import sqlite3
from datetime import datetime
from typing import Optional, Dict, List
from decimal import Decimal
from dotenv import load_dotenv

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, MarketOrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY, SELL
except ImportError:
    print("Warning: py-clob-client not installed. Run: pip install py-clob-client")
    ClobClient = None

load_dotenv()


class RiskManager:
    """Risk management for trading"""

    def __init__(
        self,
        max_position_size_usd: float = 100,
        max_total_exposure_usd: float = 500,
        max_single_market_pct: float = 0.20,  # 20% max per market
        min_edge_pct: float = 5.0,  # Minimum 5% edge to trade
        max_loss_per_trade: float = 50,  # Max $50 loss per trade
    ):
        self.max_position_size_usd = max_position_size_usd
        self.max_total_exposure_usd = max_total_exposure_usd
        self.max_single_market_pct = max_single_market_pct
        self.min_edge_pct = min_edge_pct
        self.max_loss_per_trade = max_loss_per_trade

    def calculate_position_size(
        self,
        confidence: float,
        market_price: float,
        current_exposure: float
    ) -> float:
        """Calculate position size based on confidence and risk limits"""

        # Base size from confidence (0-100%)
        confidence_multiplier = min(confidence / 100, 1.0)
        base_size = self.max_position_size_usd * confidence_multiplier

        # Check total exposure limit
        remaining_exposure = self.max_total_exposure_usd - current_exposure
        size = min(base_size, remaining_exposure)

        # Ensure we don't exceed max loss
        max_shares = self.max_loss_per_trade / market_price
        size = min(size, max_shares * market_price)

        return max(size, 0)

    def should_trade(
        self,
        ai_price: float,
        market_price: float,
        confidence: float,
        current_exposure: float
    ) -> tuple[bool, str]:
        """Check if we should trade based on edge and risk limits"""

        # Check edge requirement
        edge_pct = abs(ai_price - market_price) / market_price * 100
        if edge_pct < self.min_edge_pct:
            return False, f"insufficient edge: {edge_pct:.1f}% < {self.min_edge_pct}%"

        # Check exposure limit
        if current_exposure >= self.max_total_exposure_usd:
            return False, f"max exposure reached: ${current_exposure:.2f}"

        # Check confidence
        if confidence < 60:  # Minimum 60% confidence
            return False, f"confidence too low: {confidence:.1f}%"

        return True, "risk checks passed"


class PositionTracker:
    """Track positions and calculate P&L"""

    def __init__(self, db_path: str = "trading.db"):
        self.db_path = db_path
        self.setup_database()

    def setup_database(self):
        """Create tables for tracking trades and positions"""
        conn = sqlite3.connect(self.db_path)

        # Trades table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE,
                token_id TEXT,
                market_title TEXT,
                side TEXT,
                size REAL,
                price REAL,
                amount_usd REAL,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                filled_at TIMESTAMP,
                tweet_id TEXT
            )
        """)

        # Positions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_id TEXT UNIQUE,
                market_title TEXT,
                side TEXT,
                shares REAL,
                avg_price REAL,
                amount_invested REAL,
                current_price REAL,
                current_value REAL,
                unrealized_pnl REAL,
                realized_pnl REAL DEFAULT 0,
                status TEXT DEFAULT 'open',
                opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                closed_at TIMESTAMP
            )
        """)

        # P&L summary table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pnl_summary (
                id INTEGER PRIMARY KEY,
                total_invested REAL DEFAULT 0,
                total_realized_pnl REAL DEFAULT 0,
                total_unrealized_pnl REAL DEFAULT 0,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Initialize P&L summary if empty
        if not conn.execute("SELECT * FROM pnl_summary").fetchone():
            conn.execute("INSERT INTO pnl_summary (id) VALUES (1)")

        conn.commit()
        conn.close()

    def add_trade(
        self,
        order_id: str,
        token_id: str,
        market_title: str,
        side: str,
        size: float,
        price: float,
        amount_usd: float,
        status: str = "pending",
        tweet_id: str = None
    ):
        """Record a new trade"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO trades (order_id, token_id, market_title, side, size, price, amount_usd, status, tweet_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (order_id, token_id, market_title, side, size, price, amount_usd, status, tweet_id))
            conn.commit()
        except sqlite3.IntegrityError:
            # Order already exists
            pass
        finally:
            conn.close()

    def update_trade_status(self, order_id: str, status: str):
        """Update trade status"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE trades SET status = ?, filled_at = CURRENT_TIMESTAMP
            WHERE order_id = ?
        """, (status, order_id))
        conn.commit()
        conn.close()

    def update_position(
        self,
        token_id: str,
        market_title: str,
        side: str,
        shares: float,
        price: float,
        amount: float
    ):
        """Update or create position"""
        conn = sqlite3.connect(self.db_path)

        # Check if position exists
        existing = conn.execute(
            "SELECT * FROM positions WHERE token_id = ? AND status = 'open'",
            (token_id,)
        ).fetchone()

        if existing:
            # Update existing position
            old_shares = existing[4]
            old_avg_price = existing[5]
            old_invested = existing[6]

            new_shares = old_shares + shares
            new_invested = old_invested + amount
            new_avg_price = new_invested / new_shares if new_shares > 0 else 0

            conn.execute("""
                UPDATE positions
                SET shares = ?, avg_price = ?, amount_invested = ?
                WHERE token_id = ? AND status = 'open'
            """, (new_shares, new_avg_price, new_invested, token_id))
        else:
            # Create new position
            conn.execute("""
                INSERT INTO positions (token_id, market_title, side, shares, avg_price, amount_invested)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (token_id, market_title, side, shares, price, amount))

        # Update P&L summary
        conn.execute("""
            UPDATE pnl_summary
            SET total_invested = total_invested + ?,
                total_trades = total_trades + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
        """, (amount,))

        conn.commit()
        conn.close()

    def update_position_prices(self, token_id: str, current_price: float):
        """Update current price and calculate unrealized P&L"""
        conn = sqlite3.connect(self.db_path)

        position = conn.execute(
            "SELECT shares, avg_price FROM positions WHERE token_id = ? AND status = 'open'",
            (token_id,)
        ).fetchone()

        if position:
            shares, avg_price = position
            current_value = shares * current_price
            unrealized_pnl = current_value - (shares * avg_price)

            conn.execute("""
                UPDATE positions
                SET current_price = ?, current_value = ?, unrealized_pnl = ?
                WHERE token_id = ? AND status = 'open'
            """, (current_price, current_value, unrealized_pnl, token_id))

            # Update total unrealized P&L
            total_unrealized = conn.execute(
                "SELECT SUM(unrealized_pnl) FROM positions WHERE status = 'open'"
            ).fetchone()[0] or 0

            conn.execute("""
                UPDATE pnl_summary
                SET total_unrealized_pnl = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = 1
            """, (total_unrealized,))

        conn.commit()
        conn.close()

    def close_position(self, token_id: str, exit_price: float):
        """Close a position and realize P&L"""
        conn = sqlite3.connect(self.db_path)

        position = conn.execute(
            "SELECT shares, avg_price, unrealized_pnl FROM positions WHERE token_id = ? AND status = 'open'",
            (token_id,)
        ).fetchone()

        if position:
            shares, avg_price, unrealized_pnl = position
            realized_pnl = unrealized_pnl

            conn.execute("""
                UPDATE positions
                SET status = 'closed',
                    realized_pnl = ?,
                    closed_at = CURRENT_TIMESTAMP
                WHERE token_id = ? AND status = 'open'
            """, (realized_pnl, token_id))

            # Update P&L summary
            is_win = realized_pnl > 0
            conn.execute("""
                UPDATE pnl_summary
                SET total_realized_pnl = total_realized_pnl + ?,
                    winning_trades = winning_trades + ?,
                    losing_trades = losing_trades + ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = 1
            """, (realized_pnl, 1 if is_win else 0, 0 if is_win else 1))

        conn.commit()
        conn.close()

    def get_current_exposure(self) -> float:
        """Get total current exposure (invested capital)"""
        conn = sqlite3.connect(self.db_path)
        result = conn.execute(
            "SELECT SUM(amount_invested) FROM positions WHERE status = 'open'"
        ).fetchone()
        conn.close()
        return result[0] or 0.0

    def get_pnl_summary(self) -> Dict:
        """Get overall P&L summary"""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT * FROM pnl_summary WHERE id = 1").fetchone()
        conn.close()

        if not row:
            return {}

        total_invested = row[1]
        total_realized = row[2]
        total_unrealized = row[3]
        total_trades = row[4]
        wins = row[5]
        losses = row[6]

        total_pnl = total_realized + total_unrealized
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        roi = (total_pnl / total_invested * 100) if total_invested > 0 else 0

        return {
            'total_invested': total_invested,
            'total_realized_pnl': total_realized,
            'total_unrealized_pnl': total_unrealized,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': wins,
            'losing_trades': losses,
            'win_rate': win_rate,
            'roi': roi
        }

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT token_id, market_title, side, shares, avg_price,
                   amount_invested, current_price, current_value, unrealized_pnl
            FROM positions WHERE status = 'open'
            ORDER BY opened_at DESC
        """).fetchall()
        conn.close()

        return [
            {
                'token_id': r[0],
                'market_title': r[1],
                'side': r[2],
                'shares': r[3],
                'avg_price': r[4],
                'amount_invested': r[5],
                'current_price': r[6],
                'current_value': r[7],
                'unrealized_pnl': r[8]
            }
            for r in rows
        ]


class PolymarketTrader:
    """Main trading interface for Polymarket"""

    def __init__(
        self,
        private_key: str = None,
        funder_address: str = None,
        signature_type: int = 1,  # 1 = email/Magic wallet
        chain_id: int = 137,  # Polygon
        host: str = "https://clob.polymarket.com",
        db_path: str = "trading.db"
    ):
        self.host = host
        self.chain_id = chain_id
        self.private_key = private_key or os.getenv("POLYMARKET_PRIVATE_KEY")
        self.funder_address = funder_address or os.getenv("POLYMARKET_FUNDER_ADDRESS")
        self.signature_type = signature_type

        if not ClobClient:
            raise ImportError("py-clob-client not installed")

        # Initialize client
        self.client = ClobClient(
            self.host,
            key=self.private_key,
            chain_id=self.chain_id,
            signature_type=self.signature_type,
            funder=self.funder_address
        )

        # Set API credentials
        self.client.set_api_creds(self.client.create_or_derive_api_creds())

        # Initialize risk manager and position tracker
        self.risk_manager = RiskManager()
        self.position_tracker = PositionTracker(db_path)

        print(f"✓ Polymarket trader initialized")
        print(f"  Funder: {self.funder_address}")
        print(f"  Chain: {self.chain_id}")

    def get_market_price(self, token_id: str, side: str = "BUY") -> Optional[float]:
        """Get current market price for a token"""
        try:
            price = self.client.get_price(token_id, side=side)
            return float(price)
        except Exception as e:
            print(f"Error getting price for {token_id}: {e}")
            return None

    def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get midpoint price"""
        try:
            mid = self.client.get_midpoint(token_id)
            return float(mid)
        except Exception as e:
            print(f"Error getting midpoint for {token_id}: {e}")
            return None

    def place_market_order(
        self,
        token_id: str,
        amount_usd: float,
        side: str,  # BUY or SELL
        market_title: str = "",
        tweet_id: str = None
    ) -> Optional[Dict]:
        """Place a market order (FOK - Fill or Kill)"""
        try:
            print(f"\n{'='*50}")
            print(f"Placing market order:")
            print(f"  Token: {token_id}")
            print(f"  Side: {side}")
            print(f"  Amount: ${amount_usd:.2f}")

            # Create market order
            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount_usd,
                side=side,
                order_type=OrderType.FOK
            )

            signed_order = self.client.create_market_order(order_args)
            resp = self.client.post_order(signed_order, OrderType.FOK)

            if resp.get('success'):
                order_id = resp.get('orderId')
                price = self.get_market_price(token_id, side)
                shares = amount_usd / price if price else 0

                print(f"✓ Order placed! ID: {order_id}")
                print(f"  Shares: {shares:.2f}")
                print(f"  Price: ${price:.4f}")

                # Record trade
                self.position_tracker.add_trade(
                    order_id=order_id,
                    token_id=token_id,
                    market_title=market_title,
                    side=side,
                    size=shares,
                    price=price,
                    amount_usd=amount_usd,
                    status="filled",
                    tweet_id=tweet_id
                )

                # Update position
                self.position_tracker.update_position(
                    token_id=token_id,
                    market_title=market_title,
                    side=side,
                    shares=shares,
                    price=price,
                    amount=amount_usd
                )

                return {
                    'success': True,
                    'order_id': order_id,
                    'shares': shares,
                    'price': price,
                    'amount': amount_usd
                }
            else:
                error_msg = resp.get('errorMsg', 'Unknown error')
                print(f"✗ Order failed: {error_msg}")
                return {'success': False, 'error': error_msg}

        except Exception as e:
            print(f"✗ Error placing order: {e}")
            return {'success': False, 'error': str(e)}

    def place_limit_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str,
        market_title: str = "",
        tweet_id: str = None
    ) -> Optional[Dict]:
        """Place a limit order (GTC - Good Till Cancelled)"""
        try:
            print(f"\n{'='*50}")
            print(f"Placing limit order:")
            print(f"  Token: {token_id}")
            print(f"  Side: {side}")
            print(f"  Price: ${price:.4f}")
            print(f"  Size: {size:.2f} shares")

            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=side
            )

            signed_order = self.client.create_order(order_args)
            resp = self.client.post_order(signed_order, OrderType.GTC)

            if resp.get('success'):
                order_id = resp.get('orderId')
                amount_usd = price * size

                print(f"✓ Limit order placed! ID: {order_id}")

                # Record trade
                self.position_tracker.add_trade(
                    order_id=order_id,
                    token_id=token_id,
                    market_title=market_title,
                    side=side,
                    size=size,
                    price=price,
                    amount_usd=amount_usd,
                    status="open",
                    tweet_id=tweet_id
                )

                return {
                    'success': True,
                    'order_id': order_id,
                    'price': price,
                    'size': size,
                    'amount': amount_usd
                }
            else:
                error_msg = resp.get('errorMsg', 'Unknown error')
                print(f"✗ Order failed: {error_msg}")
                return {'success': False, 'error': error_msg}

        except Exception as e:
            print(f"✗ Error placing order: {e}")
            return {'success': False, 'error': str(e)}

    def trade_with_ai_signal(
        self,
        token_id: str,
        market_title: str,
        ai_signal: str,  # "YES" or "NO"
        ai_confidence: float,  # 0-100
        market_yes_price: float,
        market_no_price: float,
        tweet_id: str = None
    ) -> Optional[Dict]:
        """Execute trade based on AI signal with risk management"""

        print(f"\n{'='*50}")
        print(f"AI Trading Signal:")
        print(f"  Market: {market_title}")
        print(f"  Signal: {ai_signal}")
        print(f"  Confidence: {ai_confidence:.1f}%")
        print(f"  Market prices: YES {market_yes_price:.2%} / NO {market_no_price:.2%}")

        # Determine what we want to buy based on AI signal
        if ai_signal == "YES":
            side = BUY
            ai_price = ai_confidence / 100
            market_price = market_yes_price
        else:  # NO
            side = BUY  # We buy NO tokens
            ai_price = (100 - ai_confidence) / 100
            market_price = market_no_price

        # Check if we should trade
        current_exposure = self.position_tracker.get_current_exposure()
        should_trade, reason = self.risk_manager.should_trade(
            ai_price=ai_price,
            market_price=market_price,
            confidence=ai_confidence,
            current_exposure=current_exposure
        )

        if not should_trade:
            print(f"✗ Trade blocked: {reason}")
            return {'success': False, 'error': reason}

        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            confidence=ai_confidence,
            market_price=market_price,
            current_exposure=current_exposure
        )

        if position_size < 5:  # Minimum $5 trade
            print(f"✗ Position size too small: ${position_size:.2f}")
            return {'success': False, 'error': 'position size too small'}

        print(f"✓ Risk checks passed")
        print(f"  Position size: ${position_size:.2f}")
        print(f"  Current exposure: ${current_exposure:.2f}")

        # Place market order
        result = self.place_market_order(
            token_id=token_id,
            amount_usd=position_size,
            side=side,
            market_title=market_title,
            tweet_id=tweet_id
        )

        return result

    def get_open_orders(self) -> List[Dict]:
        """Get all open orders"""
        try:
            from py_clob_client.clob_types import OpenOrderParams
            orders = self.client.get_orders(OpenOrderParams())
            return orders
        except Exception as e:
            print(f"Error getting orders: {e}")
            return []

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order"""
        try:
            self.client.cancel(order_id)
            print(f"✓ Order {order_id} cancelled")
            return True
        except Exception as e:
            print(f"✗ Error cancelling order: {e}")
            return False

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders"""
        try:
            self.client.cancel_all()
            print(f"✓ All orders cancelled")
            return True
        except Exception as e:
            print(f"✗ Error cancelling orders: {e}")
            return False

    def update_all_position_prices(self):
        """Update prices for all open positions"""
        positions = self.position_tracker.get_open_positions()

        for pos in positions:
            token_id = pos['token_id']
            current_price = self.get_midpoint(token_id)

            if current_price:
                self.position_tracker.update_position_prices(token_id, current_price)
                print(f"Updated {pos['market_title'][:40]}: ${current_price:.4f}")

    def print_portfolio_status(self):
        """Print current portfolio status"""
        print(f"\n{'='*50}")
        print("PORTFOLIO STATUS")
        print(f"{'='*50}")

        # P&L summary
        pnl = self.position_tracker.get_pnl_summary()
        print(f"\n📊 P&L Summary:")
        print(f"  Total Invested: ${pnl.get('total_invested', 0):.2f}")
        print(f"  Realized P&L: ${pnl.get('total_realized_pnl', 0):.2f}")
        print(f"  Unrealized P&L: ${pnl.get('total_unrealized_pnl', 0):.2f}")
        print(f"  Total P&L: ${pnl.get('total_pnl', 0):.2f}")
        print(f"  ROI: {pnl.get('roi', 0):.2f}%")
        print(f"  Win Rate: {pnl.get('win_rate', 0):.1f}%")
        print(f"  Trades: {pnl.get('total_trades', 0)} ({pnl.get('winning_trades', 0)}W/{pnl.get('losing_trades', 0)}L)")

        # Open positions
        positions = self.position_tracker.get_open_positions()
        if positions:
            print(f"\n📈 Open Positions ({len(positions)}):")
            for pos in positions:
                pnl_str = f"+${pos['unrealized_pnl']:.2f}" if pos['unrealized_pnl'] >= 0 else f"-${abs(pos['unrealized_pnl']):.2f}"
                print(f"  • {pos['market_title'][:50]}")
                print(f"    {pos['shares']:.2f} shares @ ${pos['avg_price']:.4f} | Current: ${pos['current_price']:.4f} | P&L: {pnl_str}")
        else:
            print("\n📈 No open positions")

        print(f"{'='*50}\n")


# CLI for testing
if __name__ == "__main__":
    # Example usage
    trader = PolymarketTrader()

    # Print portfolio status
    trader.print_portfolio_status()

    # Example: Update all position prices
    # trader.update_all_position_prices()
    # trader.print_portfolio_status()
