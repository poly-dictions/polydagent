# Polydictions

CA: iATcGSt9DhJF9ZiJ6dmR153N7bW2G4J9dSSDxWSpumpㅤ

A Telegram bot for tracking and analyzing Polymarket prediction markets in real-time.

## Features

- **Real-time Monitoring**: Automatic notifications for new Polymarket events
- **Market Analysis**: Detailed event statistics including liquidity, volume, and current odds
- **AI Context**: Market context powered by Polymarket's API
- **Smart Filtering**: Filter events by custom keywords
- **Pause/Resume**: Control notifications on-demand

## Commands

- `/start` - Subscribe to event notifications
- `/deal <link>` - Analyze a specific Polymarket event
- `/keywords` - Set keyword filters for events
- `/pause` - Pause notifications
- `/resume` - Resume notifications
- `/help` - Show help information

## Installation

### Prerequisites

- Python 3.8 or higher
- A Telegram Bot Token from [@BotFather](https://t.me/BotFather)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/polydictions.git
cd polydictions
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a configuration file:
```bash
# Create config.py with your bot token
echo BOT_TOKEN = "your_bot_token_here" > config.py
```

Alternatively, create a `.env` file:
```bash
BOT_TOKEN=your_bot_token_here
```

4. Run the bot:
```bash
python bot.py
```

## Configuration

The bot uses the following files for data persistence:

- `users.json` - Stores subscribed user IDs
- `seen_events.json` - Tracks processed events
- `keywords.json` - Stores user keyword filters
- `paused_users.json` - Tracks users who paused notifications

These files are automatically created and managed by the bot.

## Usage Examples

### Analyze an Event
```
/deal https://polymarket.com/event/presidential-election-winner-2024
```

### Set Keyword Filters
```
/keywords btc, eth, crypto
/keywords "artificial intelligence", tech
/keywords clear
```

The bot supports:
- Simple keywords: `btc`, `election`, `sports`
- Phrases with quotes: `"united states"`, `"world cup"`
- Multiple keywords (OR logic): `btc, eth, sports`

### Pause and Resume
```
/pause    # Stop receiving notifications
/resume   # Start receiving notifications again
```

## API Integration

The bot integrates with:
- **Polymarket Gamma API**: For event data and market information
- **Polymarket Grok API**: For AI-generated market context

## Project Structure

```
polydictions/
├── bot.py              # Main bot application
├── config.py           # Configuration (token)
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore rules
└── README.md          # Documentation
```

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Disclaimer

This bot is for informational purposes only. Always do your own research before making any predictions or financial decisions.
