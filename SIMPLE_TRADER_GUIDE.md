# Simple Trader - Quick Start

Ultra-conservative auto-trader for your AI predictions.

**Budget: $100 total | Max bet: $5**

## What It Does

```
Reads your AI predictions → Parses with Claude → Trades automatically
```

- Reads from `polydictions-agent/calls.db`
- Uses Claude API to parse signal + confidence
- Only trades high-confidence calls (70%+)
- Only trades strong edge (10%+)
- Max 10 open positions at once

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Trading Keys to .env

You need to add these to your `.env` file:

```bash
# Polymarket Trading (Required)
POLYMARKET_PRIVATE_KEY=your-private-key-here
POLYMARKET_FUNDER_ADDRESS=your-proxy-or-wallet-address-here

# Optional: Builder code for attribution
POLYMARKET_BUILDER_CODE=your-builder-code-here

# Already in .env
ANTHROPIC_API_KEY=sk-ant-api03-...
```

#### Getting Your Keys

**Option 1: Email/Magic Wallet (Recommended)**
1. Go to https://reveal.polymarket.com
2. Login with your Polymarket email
3. Get your private key and proxy address

**Option 2: MetaMask/EOA Wallet**
1. Export private key from MetaMask
2. Your wallet address is your funder address

### 3. Verify Setup

```bash
# Check that prediction DB exists
ls polydictions-agent/calls.db

# Should see the file
```

## Usage

### Run Once

Process all untraded predictions:

```bash
python simple_trader.py
# Select "once" when prompted
```

### Run Continuously

Check for new predictions every hour:

```bash
python simple_trader.py
# Select "continuous" when prompted
```

### What You'll See

```
==================================================
SIMPLE TRADER - Processing predictions
==================================================

Found 5 untraded predictions

============================================================
Processing prediction: Will Bitcoin hit $100k in 2025?
  Signal: YES
  Confidence: 78.5%
  AI price: 78.50%
  Market price: 65.00%
  Edge: 20.8%
  Capital: $100.00
  Open positions: 0
  ✓ TRADE: $4.50
  ✓ ORDER FILLED: abc123...

============================================================
Session complete: 3/5 trades executed
Capital remaining: $86.50
==================================================
```

## Risk Settings

Current ultra-conservative settings:

```python
total_budget: $100        # Your total bankroll
max_bet: $5               # Max per trade
max_positions: 10         # Max open positions
min_confidence: 70%       # Only high confidence
min_edge: 10%             # Only strong edge
```

To adjust, edit [simple_trader.py:139-151](simple_trader.py#L139-L151)

## Tracking

All trades stored in `simple_trading.db`:

```bash
sqlite3 simple_trading.db

# View all trades
SELECT market_title, signal, confidence, bet_size, status FROM trades;

# Check capital
SELECT * FROM capital;
```

## Safety

- **Test first**: Start with just a few predictions to verify it works
- **Monitor closely**: Check the output and database regularly
- **Small amounts**: $5 max bet means low risk
- **Can stop anytime**: Just Ctrl+C to stop the trader

## How It Works

### 1. Finds Untraded Predictions

```sql
SELECT * FROM calls
WHERE created_at > (now - 7 days)
AND resolved = FALSE
```

### 2. Parses with Claude

Sends reasoning to Claude API:
```
"Parse this prediction... Extract signal (YES/NO) and confidence (0-100)"
```

### 3. Risk Check

```python
if confidence < 70%: skip
if edge < 10%: skip
if capital < $5: skip
if positions >= 10: skip
```

### 4. Calculate Bet

```python
# Scale with confidence
70% confidence → $3.50
80% confidence → $4.00
90% confidence → $4.50
100% confidence → $5.00

# Never more than 5% of capital
```

### 5. Execute Trade

```python
# Market order (instant fill)
order = client.create_market_order(token_id, bet_size)
```

### 6. Track Position

```sql
INSERT INTO trades (token_id, bet_size, market_price, ...)
UPDATE capital SET current_capital = current_capital - bet_size
```

## Files

| File | Purpose |
|------|---------|
| `simple_trader.py` | Main trading bot |
| `simple_trading.db` | Trades & capital tracking |
| `polydictions-agent/calls.db` | Your predictions (read-only) |

## Troubleshooting

### "Missing POLYMARKET_PRIVATE_KEY"

Add the keys to `.env` (see Setup step 2)

### "Could not find polydictions-agent/calls.db"

Make sure your prediction agent has created this database.

### "py-clob-client not installed"

```bash
pip install -r requirements.txt
```

### "Claude API error"

Check that `ANTHROPIC_API_KEY` is set in `.env`

## Advanced

### Custom Risk Settings

Edit [simple_trader.py:223](simple_trader.py#L223) before initializing:

```python
trader = SimpleTrader()
trader.risk = ConservativeRiskManager(
    total_budget=50.0,      # Lower budget
    max_bet=2.0,            # Lower max bet
    min_confidence=80.0,    # Higher confidence required
    min_edge=15.0           # Higher edge required
)
```

### Run Without Claude

If Claude API fails, it falls back to simple parsing. To force fallback:

```python
# Remove or comment out ANTHROPIC_API_KEY in .env
```

## Next Steps

Once it's working well:
- Monitor performance in `simple_trading.db`
- Adjust risk settings based on results
- Track which types of markets perform best
- Consider increasing budget if profitable

---

**Ultra-conservative by design. $100 budget, $5 max bet, 70%+ confidence only.**
