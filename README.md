# Polydictions Agent

Automated Twitter bot that posts free Polymarket trading signals.

## Features

- Scans Polymarket for high-quality trading opportunities
- Filters by volume, liquidity, and odds
- Generates branded banners for each post
- Auto-posts to Twitter with signals
- Avoids duplicate posts

## Signal Types

- **HIGH_CONVICTION_YES/NO** - Markets with >75% or <25% odds
- **CONTRARIAN** - Potential value plays against the crowd
- **CLOSE_RACE** - High-volume markets with 45-55% odds

## Setup

1. Clone and install:
```bash
pip install -r requirements.txt
```

2. Create config:
```bash
cp config.example.py config.py
```

3. Add Twitter API credentials to `config.py`

4. Run:
```bash
python agent.py
```

## Configuration

Edit `agent.py` to adjust:
- `CHECK_INTERVAL` - How often to scan (default: 30 min)
- `MIN_VOLUME` - Minimum market volume (default: $50k)
- `MIN_LIQUIDITY` - Minimum liquidity (default: $10k)

## Twitter Setup

1. Go to https://developer.twitter.com/
2. Create a project and app
3. Enable OAuth 1.0a with Read and Write permissions
4. Generate API keys and access tokens
5. Add to config.py

## Links

- Main Twitter: https://twitter.com/polydictions
- Chrome Extension: https://github.com/poly-dictions/polydictions-chrome-extension
