# Polymarket AI Trading System Setup

Complete guide to set up automated AI trading on Polymarket.

## Overview

The system combines:
- AI predictions via Polyfactual Deep Research API
- Automated order execution via Polymarket CLOB API
- Risk management (position sizing, exposure limits)
- P&L tracking and portfolio monitoring
- Twitter integration for transparency

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Your Polymarket Credentials

You need two things:
1. **Private Key**: The key that signs your orders
2. **Funder Address**: The address that holds your USDC

#### Option A: Email/Magic Wallet (Easiest)

1. Go to [reveal.polymarket.com](https://reveal.polymarket.com)
2. Login with your Polymarket email
3. Click "Reveal Private Key" → Copy it
4. Copy your "Proxy Wallet Address" (this is your funder address)

#### Option B: MetaMask/Hardware Wallet

1. Export your private key from MetaMask
2. Your funder address is your wallet address
3. **IMPORTANT**: You must set token allowances before trading (see below)

### 3. Configure Environment Variables

Create/update `.env` file:

```bash
# Polymarket Trading
POLYMARKET_PRIVATE_KEY="your-private-key-here"
POLYMARKET_FUNDER_ADDRESS="your-funder-address-here"
TRADING_ENABLED=true  # Set to false for prediction-only mode

# Twitter API (for posting)
TWITTER_API_KEY="..."
TWITTER_API_SECRET="..."
TWITTER_ACCESS_TOKEN="..."
TWITTER_ACCESS_SECRET="..."
TWITTER_BEARER_TOKEN="..."

# AI Analysis
POLYFACTUAL_API_KEY="..."
```

### 4. Set Token Allowances (MetaMask users only)

If you use MetaMask or hardware wallet, you need to approve tokens before trading.

**What needs approval:**
- USDC: `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174`
- Conditional Tokens: `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045`

**Contracts to approve:**
- Exchange: `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E`
- Neg Risk: `0xC5d563A36AE78145C45a50134d48A1215220f80a`
- Neg Risk Adapter: `0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296`

**How to set allowances:**

```python
# Use this script: https://gist.github.com/poly-rodr/44313920481de58d5a3f6d1f8226bd5e
# Or approve via Polygonscan
```

### 5. Test the System

Run in test mode first (predictions only):

```bash
# Set TRADING_ENABLED=false in .env
python agent_with_trading.py
```

Then enable trading:

```bash
# Set TRADING_ENABLED=true in .env
python agent_with_trading.py
```

## Architecture

### Core Components

1. **`polymarket_trader.py`** - Main trading engine
   - Order execution (market & limit orders)
   - Risk management
   - Position tracking
   - P&L calculation

2. **`agent_with_trading.py`** - AI agent with trading
   - Scans Polymarket for opportunities
   - Gets AI analysis
   - Posts predictions on Twitter
   - Executes trades

3. **Trading Database** (`trading.db`)
   - Trades table: All executed trades
   - Positions table: Open/closed positions
   - P&L summary: Overall performance

### Risk Management

Built-in risk controls:

- **Max position size**: $100 per trade (configurable)
- **Max total exposure**: $500 across all trades (configurable)
- **Max single market**: 20% of total capital
- **Min edge requirement**: 5% edge to trade
- **Max loss per trade**: $50

Configure in `polymarket_trader.py`:

```python
risk_manager = RiskManager(
    max_position_size_usd=100,
    max_total_exposure_usd=500,
    max_single_market_pct=0.20,
    min_edge_pct=5.0,
    max_loss_per_trade=50
)
```

### How It Works

1. **Scan**: Agent scans Polymarket every 4 hours
2. **Analyze**: AI analyzes market with confidence score
3. **Risk Check**: System checks if trade meets risk requirements
4. **Size**: Calculates position size based on confidence
5. **Execute**: Places market order on Polymarket
6. **Track**: Records trade and updates P&L
7. **Post**: Tweets prediction with "💰 trading this live"

## Manual Trading

You can also trade manually:

```python
from polymarket_trader import PolymarketTrader

trader = PolymarketTrader()

# Trade based on AI signal
result = trader.trade_with_ai_signal(
    token_id="<token-id>",
    market_title="Will Bitcoin hit $100k in 2025?",
    ai_signal="YES",
    ai_confidence=75.0,
    market_yes_price=0.45,
    market_no_price=0.55,
    tweet_id="123456"
)

# Check portfolio
trader.update_all_position_prices()
trader.print_portfolio_status()
```

## Monitoring

### View Portfolio Status

```python
from polymarket_trader import PolymarketTrader

trader = PolymarketTrader()
trader.update_all_position_prices()
trader.print_portfolio_status()
```

Output:
```
==================================================
PORTFOLIO STATUS
==================================================

📊 P&L Summary:
  Total Invested: $250.00
  Realized P&L: $12.50
  Unrealized P&L: $8.30
  Total P&L: $20.80
  ROI: 8.32%
  Win Rate: 66.7%
  Trades: 6 (4W/2L)

📈 Open Positions (3):
  • Will Bitcoin hit $100k in 2025?
    50.00 shares @ $0.4500 | Current: $0.4800 | P&L: +$1.50
...
==================================================
```

### Check Open Orders

```python
trader = PolymarketTrader()
orders = trader.get_open_orders()
print(f"Open orders: {len(orders)}")
```

### Cancel Orders

```python
trader.cancel_order("order-id")  # Cancel specific order
trader.cancel_all_orders()       # Cancel all orders
```

## API Reference

### PolymarketTrader

```python
trader = PolymarketTrader(
    private_key=None,          # Or from env
    funder_address=None,       # Or from env
    signature_type=1,          # 1=email, 0=EOA, 2=browser
    chain_id=137,              # Polygon
    host="https://clob.polymarket.com"
)
```

### Place Market Order

```python
result = trader.place_market_order(
    token_id="71321045679252212594626385532706912750332728571942532289631379312455583992833",
    amount_usd=25.0,
    side=BUY,  # or SELL
    market_title="Market title",
    tweet_id="optional"
)
```

### Place Limit Order

```python
result = trader.place_limit_order(
    token_id="token-id",
    price=0.45,
    size=100.0,
    side=BUY,
    market_title="Market title"
)
```

### Trade with AI Signal

```python
result = trader.trade_with_ai_signal(
    token_id="token-id",
    market_title="Market title",
    ai_signal="YES",           # or "NO"
    ai_confidence=75.0,        # 0-100
    market_yes_price=0.45,
    market_no_price=0.55,
    tweet_id="optional"
)
```

## Safety Features

✅ **Risk limits**: Built-in position sizing and exposure caps
✅ **Confidence thresholds**: Only trades high-confidence signals
✅ **Edge requirements**: Minimum edge required to trade
✅ **Confirmation prompt**: Requires "YES" to enable trading
✅ **Database tracking**: Full audit trail of all trades
✅ **P&L monitoring**: Real-time P&L tracking
✅ **Test mode**: Can run predictions without trading

## Troubleshooting

### "Error placing order: not enough balance"

- Check your USDC balance on Polygon
- Ensure funder address is correct
- For MetaMask: Check token allowances are set

### "Order failed: INVALID_ORDER_NOT_ENOUGH_BALANCE"

- You need to approve USDC spending
- For email wallets: Should work automatically
- For MetaMask: Set allowances (see section 4 above)

### "Could not initialize trader"

- Check `POLYMARKET_PRIVATE_KEY` is set
- Check `POLYMARKET_FUNDER_ADDRESS` is set
- Ensure `py-clob-client` is installed

### Trades not executing

- Check `TRADING_ENABLED=true` in `.env`
- Check risk limits aren't blocking trades
- Check you have sufficient balance
- Check confidence threshold (min 60%)

## Testing

### Test Without Trading

```bash
# predictions only, no real money
TRADING_ENABLED=false python agent_with_trading.py
```

### Test With Small Amounts

```python
# Edit polymarket_trader.py
risk_manager = RiskManager(
    max_position_size_usd=10,   # Small test size
    max_total_exposure_usd=50,  # Low exposure
    min_edge_pct=10.0           # Higher edge required
)
```

### Dry Run

```python
# Check what would happen without trading
trader = PolymarketTrader()

should_trade, reason = trader.risk_manager.should_trade(
    ai_price=0.65,
    market_price=0.45,
    confidence=75.0,
    current_exposure=0
)
print(f"Would trade: {should_trade}, reason: {reason}")
```

## Production Deployment

### Railway/Heroku

1. Add to `Procfile`:
```
worker: python agent_with_trading.py
```

2. Set environment variables in dashboard

3. Deploy

### Docker

```dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "agent_with_trading.py"]
```

## Support

- [Polymarket CLOB Docs](https://docs.polymarket.com/developers/CLOB/introduction)
- [py-clob-client GitHub](https://github.com/Polymarket/py-clob-client)
- [Polymarket Discord](https://discord.gg/polymarket)

## Disclaimer

⚠️ **This is experimental software for educational purposes.**

- Trading cryptocurrency markets involves significant risk
- Past performance does not guarantee future results
- Only trade with money you can afford to lose
- Always test thoroughly before using real funds
- No guarantees or warranties provided
- Use at your own risk

## License

MIT
