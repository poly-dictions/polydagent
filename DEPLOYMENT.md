# Deployment Guide

## Deploying to GitHub

### Step 1: Remove Sensitive Data

Before pushing to GitHub, ensure no sensitive data is committed:

```bash
# Make sure config.py is not tracked
git rm --cached config.py 2>/dev/null || true

# Verify .gitignore is working
git status
```

Your `.gitignore` should already exclude:
- `config.py` (contains bot token)
- `.env` (environment variables)
- `users.json`, `seen_events.json`, `keywords.json`, `paused_users.json` (user data)

### Step 2: Initialize Git Repository

```bash
# Navigate to project directory
cd polydictions

# Initialize git repository
git init

# Add all files (sensitive files will be ignored by .gitignore)
git add .

# Create initial commit
git commit -m "Initial commit"
```

### Step 3: Create GitHub Repository

1. Go to [GitHub](https://github.com) and log in
2. Click the **+** icon in the top right and select **New repository**
3. Name your repository (e.g., `polydictions`)
4. Choose **Public** or **Private**
5. Do NOT initialize with README (you already have one)
6. Click **Create repository**

### Step 4: Push to GitHub

```bash
# Add GitHub remote (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 5: Verify Deployment

Visit your repository on GitHub and verify:
- README.md displays correctly
- config.py is NOT visible (only config.py.example)
- No sensitive data is exposed

## Deploying the Bot

### Local Deployment

```bash
# Clone your repository
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
cd REPO_NAME

# Install dependencies
pip install -r requirements.txt

# Create config file from example
cp config.py.example config.py

# Edit config.py and add your bot token
# Get token from @BotFather on Telegram

# Run the bot
python bot.py
```

### VPS/Server Deployment

For production deployment on a VPS:

```bash
# Install Python 3.8+
sudo apt update
sudo apt install python3 python3-pip

# Clone repository
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
cd REPO_NAME

# Install dependencies
pip3 install -r requirements.txt

# Create config
cp config.py.example config.py
nano config.py  # Add your token

# Run with nohup (keeps running after logout)
nohup python3 bot.py > bot.log 2>&1 &

# Or use screen
screen -S polydictions
python3 bot.py
# Press Ctrl+A then D to detach
```

### Using systemd (Recommended for VPS)

Create a service file `/etc/systemd/system/polydictions.service`:

```ini
[Unit]
Description=Polydictions Telegram Bot
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/polydictions
ExecStart=/usr/bin/python3 /path/to/polydictions/bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable polydictions
sudo systemctl start polydictions
sudo systemctl status polydictions
```

### Docker Deployment (Optional)

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "bot.py"]
```

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  bot:
    build: .
    restart: unless-stopped
    volumes:
      - ./data:/app/data
    environment:
      - BOT_TOKEN=${BOT_TOKEN}
```

Run:

```bash
docker-compose up -d
```

## Environment Variables

Instead of `config.py`, you can use environment variables:

```bash
export BOT_TOKEN="your_token_here"
python bot.py
```

Or create a `.env` file:

```bash
BOT_TOKEN=your_token_here
```

The bot will automatically load from `.env` if `config.py` doesn't exist.

## Monitoring

Check bot logs:

```bash
# If using systemd
sudo journalctl -u polydictions -f

# If using nohup
tail -f bot.log

# If using screen
screen -r polydictions
```

## Updating

```bash
# Pull latest changes
git pull origin main

# Restart the bot
sudo systemctl restart polydictions

# Or if using screen/nohup, kill and restart
pkill -f bot.py
nohup python3 bot.py > bot.log 2>&1 &
```

## Security Notes

- Never commit `config.py` or `.env` to GitHub
- Keep your bot token private
- Regularly update dependencies: `pip install --upgrade -r requirements.txt`
- Use HTTPS for production deployments
- Consider using secrets management for production (e.g., AWS Secrets Manager, HashiCorp Vault)

## Troubleshooting

If the bot doesn't start:

1. Check token is correct: `grep BOT_TOKEN config.py`
2. Check dependencies: `pip install -r requirements.txt`
3. Check Python version: `python --version` (should be 3.8+)
4. Check logs for errors
5. Verify internet connectivity to Telegram API
