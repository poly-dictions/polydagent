# Simple Trader - Ready to Use

Your ultra-conservative auto-trader is ready. Here's what's been set up:

## What You Have

### 1. Simple Trader ([simple_trader.py](simple_trader.py))
- Reads predictions from `polydictions-agent/calls.db`
- Uses Claude API to parse signal + confidence
- **Budget: $100 total**
- **Max bet: $5 per trade**
- **Min confidence: 70%**
- **Min edge: 10%**
- **Max positions: 10**

### 2. Verification Script ([test_simple_trader.py](test_simple_trader.py))
Test your setup without placing trades:
```bash
python test_simple_trader.py
```

### 3. Documentation
- [SIMPLE_TRADER_GUIDE.md](SIMPLE_TRADER_GUIDE.md) - Complete usage guide
- [.env.trading.example](.env.trading.example) - Configuration template

## Current Status

Just ran verification:

**✓ Working:**
- Claude API connected
- Polymarket API accessible
- Database found with 3 unresolved predictions:
  - 2026 NBA Champion (OKC Thunder) - 45.5% YES
  - Xi Jinping out before 2027? - 11.5% YES
  - Russia x Ukraine ceasefire by March 31, 2026? - 24.5% YES

**✗ Need to Setup:**
1. **Install py-clob-client**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add trading keys to .env**
   ```bash
   # Add these lines to your .env file:
   POLYMARKET_PRIVATE_KEY=your-private-key-here
   POLYMARKET_FUNDER_ADDRESS=your-wallet-address-here
   ```

   **How to get these:**
   - Go to https://reveal.polymarket.com
   - Login with your Polymarket email
   - Get your private key and proxy address

   **OR use MetaMask:**
   - Export private key from MetaMask
   - Use your wallet address as funder address

3. **Builder attribution (already set)**
   Your existing `POLYMARKET_BUILDERS_KEY` in .env will be used automatically for attribution.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Trading Keys
Edit `.env` and add the two lines above.

### 3. Verify Setup
```bash
python test_simple_trader.py
```
All checks should pass.

### 4. Run Trader (Once)
```bash
python simple_trader.py
# Choose "once" when prompted
```

This will:
- Process your 3 unresolved predictions
- Show what would be traded
- Execute trades that meet criteria

### 5. Run Continuously
```bash
python simple_trader.py
# Choose "continuous" when prompted
```

Checks for new predictions every hour.

## What Will Happen

Based on your current predictions, here's what the trader will analyze:

### Prediction 1: 2026 NBA Champion (OKC Thunder)
- Market: 45.5% YES
- Will parse with Claude to get AI signal + confidence
- If confidence > 70% and edge > 10%: Trade
- Max bet: $5.00

### Prediction 2: Xi Jinping out before 2027?
- Market: 11.5% YES
- Same analysis
- Conservative limits apply

### Prediction 3: Russia x Ukraine ceasefire by March 31, 2026?
- Market: 24.5% YES
- Same analysis
- Conservative limits apply

## Risk Management

Every trade must pass these checks:

```python
✓ Confidence >= 70%
✓ Edge >= 10%
✓ Capital >= $5
✓ Positions < 10
✓ Bet <= $5
```

If any check fails, trade is skipped with reason shown.

## Tracking

All trades tracked in `simple_trading.db`:

```bash
# View trades
sqlite3 simple_trading.db "SELECT market_title, signal, confidence, bet_size, status FROM trades;"

# Check capital
sqlite3 simple_trading.db "SELECT * FROM capital;"
```

## Example Output

```
============================================================
Processing prediction: 2026 NBA Champion (OKC Thunder)
  Signal: YES
  Confidence: 78.5%
  AI price: 78.50%
  Market price: 45.50%
  Edge: 72.5%
  Capital: $100.00
  Open positions: 0
  [OK] TRADE: $4.50
  [OK] ORDER FILLED: abc123...

============================================================
Session complete: 1/3 trades executed
Capital remaining: $95.50
============================================================
```

## Safety Features

1. **Ultra-conservative limits**: $5 max, 70%+ confidence only
2. **Dry-run first**: Test with verification script
3. **Full audit trail**: Every trade in database
4. **Can stop anytime**: Ctrl+C to stop
5. **Builder attribution**: Orders tracked for rewards

## Next Steps

1. `pip install -r requirements.txt`
2. Add keys to `.env`
3. `python test_simple_trader.py` (verify)
4. `python simple_trader.py` (trade!)

## Files Created

- [simple_trader.py](simple_trader.py) - Main trading bot (600 lines)
- [test_simple_trader.py](test_simple_trader.py) - Verification script
- [SIMPLE_TRADER_GUIDE.md](SIMPLE_TRADER_GUIDE.md) - Full documentation
- [requirements.txt](requirements.txt) - Updated with dependencies

## What's Different from Earlier Systems

Earlier you had a complex system with:
- Multiple files (polymarket_trader.py, agent_with_trading.py)
- Twitter integration
- Complex risk management
- Word-counting AI parser

**Simple trader:**
- ✓ Single file - simple_trader.py
- ✓ No Twitter - just reads from calls.db
- ✓ Uses Claude API for smart parsing
- ✓ Ultra-conservative: $100 budget, $5 max bet
- ✓ Builder code attribution included

## Architecture

```
polydictions-agent/calls.db  (your predictions)
           ↓
    simple_trader.py
           ↓
    Claude API (parse signal + confidence)
           ↓
    Risk Manager (70%+ confidence, 10%+ edge)
           ↓
    Polymarket CLOB (place $5 order)
           ↓
    simple_trading.db (track position)
```

Clean, simple, conservative.

---

**Ready to trade on your AI predictions with $100 budget and $5 max bets.**
