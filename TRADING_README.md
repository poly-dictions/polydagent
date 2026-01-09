# 🤖 Polydictions AI Trading System

AI-powered automated trading on Polymarket with built-in risk management and P&L tracking.

## What It Does

```
AI predicts → AI trades → we watch how it performs in prediction markets
```

1. **Scans** Polymarket 24/7 for opportunities
2. **Analyzes** markets with AI (via Polyfactual Deep Research API)
3. **Trades** automatically based on AI signals with confidence
4. **Manages** risk with position sizing and exposure limits
5. **Tracks** all positions and P&L in real-time
6. **Posts** predictions on Twitter with live trade transparency

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Add your keys to .env

# 3. Run (test mode first)
TRADING_ENABLED=false python agent_with_trading.py

# 4. Enable trading
TRADING_ENABLED=true python agent_with_trading.py
```

See [TRADING_SETUP.md](TRADING_SETUP.md) for full setup guide.

## Features

### ✅ AI-Powered
- Polyfactual Deep Research API for market analysis
- Confidence scoring (0-100%)
- Edge detection

### ✅ Risk Management
- Max position size: $100 per trade
- Max total exposure: $500
- Minimum edge required: 5%
- Confidence threshold: 60%+
- Max loss per trade: $50

### ✅ Automated Trading
- Market orders (instant execution)
- Limit orders (resting on book)
- Order management (cancel, modify)
- Position tracking

### ✅ P&L Tracking
- Real-time position monitoring
- Unrealized P&L calculation
- Realized P&L on close
- Win rate tracking
- ROI calculation

### ✅ Transparency
- All trades posted on Twitter
- Full audit trail in database
- Open positions visible
- P&L publicly trackable

## Architecture

```
┌─────────────────┐
│  AI Analysis    │ Polyfactual API
│  (Confidence)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Risk Manager    │ Position sizing
│ Should we trade?│ Exposure limits
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Order Execution │ py-clob-client
│ Polymarket CLOB │ Market/Limit
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Position Tracker│ SQLite DB
│ P&L Calculation │ Portfolio stats
└─────────────────┘
```

## Components

| File | Purpose |
|------|---------|
| `polymarket_trader.py` | Core trading engine |
| `agent_with_trading.py` | AI agent with trading |
| `trading_examples.py` | Usage examples |
| `TRADING_SETUP.md` | Full setup guide |
| `trading.db` | Trades & positions database |

## Risk Controls

Built-in safeguards:

```python
✓ Position sizing based on confidence
✓ Maximum exposure limits
✓ Minimum edge requirements
✓ Confirmation prompt before trading
✓ Test mode (predictions without trading)
✓ Full database audit trail
```

## Example Output

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
  • Trump wins 2024?
    100.00 shares @ $0.6200 | Current: $0.6500 | P&L: +$3.00
  • Fed cuts rates by March?
    75.00 shares @ $0.3300 | Current: $0.3680 | P&L: +$2.85
==================================================
```

## API Usage

### Basic Trading

```python
from polymarket_trader import PolymarketTrader

trader = PolymarketTrader()

# Trade on AI signal
result = trader.trade_with_ai_signal(
    token_id="token-id",
    market_title="Market title",
    ai_signal="YES",        # or "NO"
    ai_confidence=75.0,     # 0-100
    market_yes_price=0.45,
    market_no_price=0.55
)

# Check portfolio
trader.update_all_position_prices()
trader.print_portfolio_status()
```

### Manual Orders

```python
# Market order
trader.place_market_order(
    token_id="token-id",
    amount_usd=25.0,
    side="BUY"
)

# Limit order
trader.place_limit_order(
    token_id="token-id",
    price=0.45,
    size=100.0,
    side="BUY"
)
```

## Configuration

### Environment Variables

```bash
# Required
POLYMARKET_PRIVATE_KEY="your-key"
POLYMARKET_FUNDER_ADDRESS="your-address"

# Optional
TRADING_ENABLED=true
AGENT_POST_INTERVAL=4  # hours
```

### Risk Settings

Edit `polymarket_trader.py`:

```python
risk_manager = RiskManager(
    max_position_size_usd=100,
    max_total_exposure_usd=500,
    max_single_market_pct=0.20,
    min_edge_pct=5.0,
    max_loss_per_trade=50
)
```

## For $POLYD Holders

Once we fine-tune the AI model, we'll open it to $POLYD holders.

**What you'll get:**
- Access to AI trading signals
- Real-time position tracking
- Transparent P&L
- Risk-managed execution
- Full audit trail

## Testing

### Test Mode

```bash
# Predictions only, no trading
TRADING_ENABLED=false python agent_with_trading.py
```

### Small Amounts

```python
# Start with small positions
risk_manager = RiskManager(
    max_position_size_usd=10,
    max_total_exposure_usd=50
)
```

### Examples

```bash
# Run safe examples
python trading_examples.py
```

## Requirements

- Python 3.9+
- Polymarket account (email or MetaMask)
- USDC on Polygon
- Private key access
- Twitter API keys (for posting)
- Polyfactual API key (for AI)

## Deployment

### Local

```bash
python agent_with_trading.py
```

### Railway/Heroku

```
worker: python agent_with_trading.py
```

### Docker

```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "agent_with_trading.py"]
```

## Safety

⚠️ **IMPORTANT**

- Start with test mode (`TRADING_ENABLED=false`)
- Use small amounts initially
- Monitor positions closely
- Understand the risks
- Only trade what you can afford to lose

## Documentation

- [Full Setup Guide](TRADING_SETUP.md)
- [Polymarket CLOB Docs](https://docs.polymarket.com/developers/CLOB/introduction)
- [py-clob-client](https://github.com/Polymarket/py-clob-client)

## Support

Questions? Issues?

- Open an issue on GitHub
- Check [TRADING_SETUP.md](TRADING_SETUP.md) for troubleshooting
- Join [Polymarket Discord](https://discord.gg/polymarket)

## Disclaimer

⚠️ This is experimental software for educational purposes. Trading involves risk of loss. Past performance does not guarantee future results. Use at your own risk.

## License

MIT

---

**building AI agents that trade on our AI calls**

no bs, no guarantees. just testing if this actually works or if we're just building fancy gambling bots
