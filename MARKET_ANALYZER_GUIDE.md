# Market Analyzer - Quick Start Guide

Randomly selects Polymarket markets and displays detailed AI reasoning in terminal.

## What It Does

1. Fetches random active binary markets from Polymarket
2. Calls Polyfactual API for deep research analysis
3. Uses Claude Sonnet 3.5 to format and parse reasoning
4. Displays beautiful terminal output with:
   - Market title and URL
   - Current YES/NO prices
   - AI signal (YES/NO) and confidence
   - Edge calculation
   - Detailed reasoning
   - Key factors

## Requirements

Already installed if you set up simple_trader:
- Python 3.8+
- aiohttp
- requests
- python-dotenv

API Keys needed in `.env`:
```bash
ANTHROPIC_API_KEY=your-key-here
POLYFACTUAL_API_KEY=your-key-here
POLYFACTUAL_API_URL=https://deep-research-api.thekid-solana.workers.dev/answer
```

## Quick Start

### Run Once (Analyze N Markets)

```bash
python market_analyzer.py
# Choose: once
# Enter number of markets (e.g., 5)
```

### Run Continuously (Every 5 Minutes)

```bash
python market_analyzer.py
# Choose: continuous
```

## Example Output

```
======================================================================
MARKET: Will Donald Trump win the 2024 Presidential Election?
======================================================================
URL: https://polymarket.com/event/presidential-election-2024
Current Prices: YES 62.3% | NO 37.7%

SIGNAL: YES
CONFIDENCE: 75%
AI PRICE: 75.0%
MARKET PRICE: 62.3%
EDGE: 20.4%

REASONING:
----------------------------------------------------------------------
Based on current polling data and electoral college projections,
Trump maintains a significant lead in key swing states. Recent
economic indicators favor the incumbent party, and historical
patterns suggest a high probability of victory given current trends.
----------------------------------------------------------------------

KEY FACTORS:
  1. Leading in 5/7 swing states by average of 3.2 points
  2. Strong economic fundamentals favor incumbent
  3. Historical re-election patterns align with current scenario
======================================================================
```

## How It Works

1. **Random Market Selection**
   - Fetches 100 active markets from Polymarket API
   - Filters for binary YES/NO markets only
   - Randomly selects specified number

2. **Deep Research (Polyfactual)**
   - Sends market question to Polyfactual API
   - Gets comprehensive research and analysis
   - Typically 500-2000 words of detailed research

3. **AI Formatting (Claude Sonnet 3.5)**
   - Processes Polyfactual research
   - Extracts signal (YES/NO)
   - Calculates confidence (0-100)
   - Formats clear reasoning (2-4 sentences)
   - Identifies 3 key supporting factors

4. **Edge Calculation**
   - AI Price = Confidence (for YES) or 100-Confidence (for NO)
   - Market Price = Current YES or NO price
   - Edge = |AI Price - Market Price| / Market Price * 100%

## Configuration

Edit `market_analyzer.py` to customize:

```python
# Change model (line 119)
"model": "claude-3-5-sonnet-20241022",  # or "claude-3-opus-20240229"

# Change continuous interval (line 269)
await asyncio.sleep(300)  # 300 seconds = 5 minutes

# Change number of markets per run
await analyzer.run(num_markets=3)  # default: 3 in continuous, 5 in once
```

## Use Cases

### Research Mode
Run once to analyze specific number of markets for research:
```bash
python market_analyzer.py
# Choose: once
# Enter: 10
```

### Monitoring Mode
Run continuously to monitor random market analysis:
```bash
python market_analyzer.py
# Choose: continuous
```

### Integration with Trading
Use the output to inform trading decisions (manual or automated):
- High confidence (>80%) + High edge (>15%) = Strong signal
- Compare AI reasoning with your own research
- Track which markets have persistent edge

## Differences from simple_trader.py

| Feature | market_analyzer.py | simple_trader.py |
|---------|-------------------|------------------|
| Purpose | Display market analysis | Auto-trade positions |
| Markets | Random from API | From calls.db only |
| Actions | Display only | Places real trades |
| Model | Claude Sonnet 3.5 | Claude Haiku |
| Output | Terminal reasoning | Trade execution logs |
| Frequency | Every 5 min (continuous) | Every 2 hours |

## Tips

1. **For Research**: Run once with 10-20 markets to find interesting opportunities
2. **For Monitoring**: Run continuous mode to keep discovering new markets
3. **For Trading**: Use high-confidence signals from analyzer to add to calls.db
4. **For Learning**: Study the reasoning to understand how AI evaluates markets

## Troubleshooting

### No markets found
- Polymarket API might be down
- Check internet connection
- Try again in a few minutes

### Polyfactual API errors
- Check POLYFACTUAL_API_KEY in .env
- API might be rate limited
- Verify POLYFACTUAL_API_URL is correct

### Claude API errors
- Check ANTHROPIC_API_KEY in .env
- Verify API key is valid and has credits
- Rate limits: 50 requests/minute for Sonnet

### JSON parsing errors
- Claude sometimes returns extra text
- Script handles this automatically
- If persistent, try different model

## Advanced Usage

### Save Analysis to File

Redirect output to file:
```bash
python market_analyzer.py > analysis_$(date +%Y%m%d_%H%M%S).txt
```

### Filter by Edge

Modify script to only show high-edge opportunities:
```python
if edge < 15.0:
    print("[*] Edge too low, skipping display...")
    return None
```

### Custom Market Selection

Replace random selection with specific categories:
```python
# In get_random_markets(), add filter:
params={
    "limit": 100,
    "active": "true",
    "closed": "false",
    "tag": "politics"  # or "sports", "crypto", etc.
}
```

---

**Simple, powerful market analysis in your terminal.**
