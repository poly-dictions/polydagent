"""
Trading System Examples
Show how to use the Polymarket trading system
"""

import asyncio
from polymarket_trader import PolymarketTrader, RiskManager


def example_1_check_portfolio():
    """Example 1: Check portfolio status"""
    print("="*50)
    print("EXAMPLE 1: Check Portfolio Status")
    print("="*50)

    trader = PolymarketTrader()

    # Update all position prices
    trader.update_all_position_prices()

    # Print portfolio
    trader.print_portfolio_status()


def example_2_manual_trade():
    """Example 2: Place a manual trade"""
    print("="*50)
    print("EXAMPLE 2: Manual Trade")
    print("="*50)

    trader = PolymarketTrader()

    # Example token ID (you need to get real ones from Polymarket API)
    token_id = "71321045679252212594626385532706912750332728571942532289631379312455583992833"

    # Place a $25 market buy order
    result = trader.place_market_order(
        token_id=token_id,
        amount_usd=25.0,
        side="BUY",
        market_title="Example Market"
    )

    if result.get('success'):
        print(f"✓ Trade executed!")
        print(f"  Order ID: {result['order_id']}")
        print(f"  Shares: {result['shares']:.2f}")
        print(f"  Price: ${result['price']:.4f}")
    else:
        print(f"✗ Trade failed: {result.get('error')}")


def example_3_ai_signal_trade():
    """Example 3: Trade based on AI signal"""
    print("="*50)
    print("EXAMPLE 3: AI Signal Trade")
    print("="*50)

    trader = PolymarketTrader()

    # Simulate AI analysis results
    result = trader.trade_with_ai_signal(
        token_id="example-token-id",
        market_title="Will Bitcoin hit $100k in 2025?",
        ai_signal="YES",
        ai_confidence=75.0,
        market_yes_price=0.45,  # Current market price for YES
        market_no_price=0.55,   # Current market price for NO
        tweet_id="123456"
    )

    if result.get('success'):
        print(f"✓ AI trade executed!")
        print(f"  Amount: ${result['amount']:.2f}")
    else:
        print(f"✗ Trade not executed: {result.get('error')}")


def example_4_risk_check():
    """Example 4: Check if a trade passes risk requirements"""
    print("="*50)
    print("EXAMPLE 4: Risk Check")
    print("="*50)

    trader = PolymarketTrader()

    # AI thinks market should be at 65%, but it's at 45%
    # That's a 20% edge (44% edge ratio)
    should_trade, reason = trader.risk_manager.should_trade(
        ai_price=0.65,
        market_price=0.45,
        confidence=75.0,
        current_exposure=0  # No current exposure
    )

    print(f"Should trade: {should_trade}")
    print(f"Reason: {reason}")

    if should_trade:
        # Calculate position size
        size = trader.risk_manager.calculate_position_size(
            confidence=75.0,
            market_price=0.45,
            current_exposure=0
        )
        print(f"Position size: ${size:.2f}")


def example_5_manage_orders():
    """Example 5: View and manage open orders"""
    print("="*50)
    print("EXAMPLE 5: Manage Orders")
    print("="*50)

    trader = PolymarketTrader()

    # Get all open orders
    orders = trader.get_open_orders()
    print(f"Open orders: {len(orders)}")

    if orders:
        # Show first order
        order = orders[0]
        print(f"\nFirst order:")
        print(f"  ID: {order.get('id')}")
        print(f"  Token: {order.get('asset_id')}")
        print(f"  Price: ${order.get('price')}")
        print(f"  Size: {order.get('size')}")

        # Cancel it (commented out for safety)
        # trader.cancel_order(order['id'])

    # Cancel all orders (commented out for safety)
    # trader.cancel_all_orders()


def example_6_custom_risk_settings():
    """Example 6: Use custom risk settings"""
    print("="*50)
    print("EXAMPLE 6: Custom Risk Settings")
    print("="*50)

    # Create trader with custom risk settings
    trader = PolymarketTrader()

    # Override risk manager with conservative settings
    trader.risk_manager = RiskManager(
        max_position_size_usd=50,      # Lower position size
        max_total_exposure_usd=200,    # Lower total exposure
        max_single_market_pct=0.15,    # Max 15% per market
        min_edge_pct=10.0,             # Require 10% edge
        max_loss_per_trade=25          # Max $25 loss per trade
    )

    print("Custom risk settings:")
    print(f"  Max position: ${trader.risk_manager.max_position_size_usd}")
    print(f"  Max exposure: ${trader.risk_manager.max_total_exposure_usd}")
    print(f"  Min edge: {trader.risk_manager.min_edge_pct}%")

    # Now use trader as normal
    # trader.trade_with_ai_signal(...)


def example_7_monitor_pnl():
    """Example 7: Monitor P&L over time"""
    print("="*50)
    print("EXAMPLE 7: Monitor P&L")
    print("="*50)

    trader = PolymarketTrader()

    # Get P&L summary
    pnl = trader.position_tracker.get_pnl_summary()

    print("P&L Summary:")
    print(f"  Total Invested: ${pnl.get('total_invested', 0):.2f}")
    print(f"  Realized P&L: ${pnl.get('total_realized_pnl', 0):.2f}")
    print(f"  Unrealized P&L: ${pnl.get('total_unrealized_pnl', 0):.2f}")
    print(f"  Total P&L: ${pnl.get('total_pnl', 0):.2f}")
    print(f"  ROI: {pnl.get('roi', 0):.2f}%")
    print(f"  Win Rate: {pnl.get('win_rate', 0):.1f}%")
    print(f"  Trades: {pnl.get('total_trades', 0)} total")
    print(f"  W/L: {pnl.get('winning_trades', 0)}W / {pnl.get('losing_trades', 0)}L")

    # Get open positions
    positions = trader.position_tracker.get_open_positions()
    print(f"\nOpen positions: {len(positions)}")

    for pos in positions[:3]:  # Show first 3
        print(f"\n  {pos['market_title'][:50]}")
        print(f"    Shares: {pos['shares']:.2f}")
        print(f"    Avg Price: ${pos['avg_price']:.4f}")
        print(f"    Current: ${pos['current_price']:.4f}")
        pnl_str = f"+${pos['unrealized_pnl']:.2f}" if pos['unrealized_pnl'] >= 0 else f"-${abs(pos['unrealized_pnl']):.2f}"
        print(f"    P&L: {pnl_str}")


async def example_8_async_trading():
    """Example 8: Async trading with monitoring"""
    print("="*50)
    print("EXAMPLE 8: Async Trading")
    print("="*50)

    trader = PolymarketTrader()

    # Simulate trading loop
    for i in range(3):
        print(f"\nIteration {i+1}:")

        # Update positions
        trader.update_all_position_prices()

        # Check portfolio
        pnl = trader.position_tracker.get_pnl_summary()
        print(f"  Total P&L: ${pnl.get('total_pnl', 0):.2f}")
        print(f"  ROI: {pnl.get('roi', 0):.2f}%")

        # Wait before next iteration
        await asyncio.sleep(2)

    print("\n✓ Done!")


def main():
    """Run examples"""
    print("\n" + "="*50)
    print("POLYMARKET TRADING EXAMPLES")
    print("="*50 + "\n")

    # Run examples (comment out ones you don't want to run)

    # Safe examples (read-only)
    example_1_check_portfolio()
    # example_4_risk_check()
    # example_5_manage_orders()
    # example_6_custom_risk_settings()
    # example_7_monitor_pnl()

    # Trading examples (commented out for safety)
    # example_2_manual_trade()
    # example_3_ai_signal_trade()

    # Async example
    # asyncio.run(example_8_async_trading())


if __name__ == "__main__":
    main()
