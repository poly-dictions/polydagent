# Quantish MCP Server Setup Guide

Connect AI agents to prediction markets (Polymarket & Kalshi) via MCP (Model Context Protocol).

---

## Available MCP Servers

| Server | Purpose | Signup Required |
|--------|---------|-----------------|
| **Discovery** | Market search, trending, arbitrage | No |
| **Polymarket** | Trading on Polygon/EVM | Yes |
| **Kalshi** | Trading on Solana via DFlow | Yes |

---

## 1. Quantish Discovery MCP

**Purpose**: AI-powered semantic search across Polymarket and Kalshi markets.

**URL**: `https://quantish.live/mcp`

### No Signup Required

Add directly to your MCP config:

```json
{
  "quantish-discovery": {
    "type": "http",
    "url": "https://quantish.live/mcp",
    "headers": {
      "X-API-Key": "qm_dyvSSGPYLcRwLXBNmW5VPa94sjCVAe-6"
    }
  }
}
```

### Available Tools

- `search_markets` - Semantic search across both platforms
- `get_trending_markets` - Hot markets by volume
- `find_arbitrage` - Detect arbitrage opportunities
- `get_market_details` - Detailed market info
- `xtracker_tweet_count` - Elon Musk tweet tracking (for tweet count markets)

---

## 2. Quantish Polymarket MCP

**Purpose**: Trade prediction markets on Polymarket (Polygon blockchain).

**URL**: `https://quantish-sdk-production.up.railway.app/mcp`

### Signup Flow (2 Steps)

#### Step 1: Request API Key

Call the `request_api_key` tool:

```
Tool: request_api_key

Parameters:
  externalId: "your-unique-id"    # Required: email, telegram ID, or any unique string
  keyName: "My Trading Bot"       # Optional: friendly name for this key

Returns:
  - apiKey: "pk_live_xxx..."      # Use this in your MCP config
  - apiSecret: "..."              # For optional HMAC signing
  - message: "API key created"
```

#### Step 2: Setup Wallet

Call the `setup_wallet` tool (uses your API key from Step 1):

```
Tool: setup_wallet

Parameters: none

This will:
  - Deploy a Safe wallet on Polygon (gasless)
  - Set up USDC and CTF token approvals
  - Generate CLOB trading credentials

Returns:
  - safeAddress: "0x..."          # Your trading wallet
  - status: "READY"
```

### Add to MCP Config

```json
{
  "quantish-polymarket": {
    "type": "http",
    "url": "https://quantish-sdk-production.up.railway.app/mcp",
    "headers": {
      "x-api-key": "pk_live_YOUR_API_KEY_HERE"
    }
  }
}
```

### Available Tools

**Trading:**
- `place_order` - Buy/sell shares (GTC, FOK, FAK order types)
- `cancel_order` - Cancel open orders
- `get_orders` - View order history
- `get_positions` - View current holdings
- `execute_atomic_orders` - Multi-order atomic execution

**Wallet:**
- `get_balances` - USDC and MATIC balances
- `get_deposit_addresses` - Fund your wallet
- `transfer_usdc` / `transfer_shares` - Send assets
- `swap_tokens` - Swap MATIC/USDC via LI.FI

**Markets:**
- `search_markets` - Find markets by keyword
- `get_market` - Market details by condition ID
- `get_orderbook` - Live bids/asks
- `get_price` - Current midpoint price

---

## 3. Quantish Kalshi MCP

**Purpose**: Trade Kalshi prediction markets via DFlow on Solana.

**URL**: `https://kalshi-mcp-production-7c2c.up.railway.app/mcp`

### Signup Flow (1 Step)

Call the `kalshi_signup` tool:

```
Tool: kalshi_signup

Parameters:
  externalId: "your-unique-id"    # Required: email, telegram ID, or any unique string
  keyName: "My Kalshi Bot"        # Optional: friendly name

Returns:
  - apiKey: "pk_kalshi_xxx..."    # Use this in your MCP config
  - publicKey: "9mNa..."          # Your Solana wallet address
  - message: "Account created"
```

This single call:
- Creates your API credentials
- Generates a Solana wallet
- Sets up everything needed for trading

### Add to MCP Config

```json
{
  "quantish-kalshi": {
    "type": "http",
    "url": "https://kalshi-mcp-production-7c2c.up.railway.app/mcp",
    "headers": {
      "x-api-key": "pk_kalshi_YOUR_API_KEY_HERE"
    }
  }
}
```

### Available Tools

**Trading:**
- `kalshi_buy_yes` / `kalshi_buy_no` - Buy outcome tokens
- `kalshi_sell_position` - Sell back to USDC
- `kalshi_get_positions` - Current holdings
- `kalshi_redeem_winnings` - Claim from settled markets

**Wallet:**
- `kalshi_get_balances` - SOL and USDC balances
- `kalshi_get_deposit_address` - Fund your wallet
- `kalshi_swap_sol_to_usdc` / `kalshi_swap_usdc_to_sol` - Jupiter swaps
- `kalshi_send_sol` / `kalshi_send_usdc` - Withdrawals

**Markets:**
- `kalshi_search_markets` - Find markets
- `kalshi_get_market` - Market details
- `kalshi_get_event` - Event with all nested markets
- `kalshi_get_live_data` - Real-time pricing

---

## Full MCP Config Example

For Claude Code, add to `~/.claude.json` under your project:

```json
{
  "projects": {
    "/path/to/your/project": {
      "mcpServers": {
        "quantish-discovery": {
          "type": "http",
          "url": "https://quantish.live/mcp",
          "headers": {
            "X-API-Key": "qm_dyvSSGPYLcRwLXBNmW5VPa94sjCVAe-6"
          }
        },
        "quantish-polymarket": {
          "type": "http",
          "url": "https://quantish-sdk-production.up.railway.app/mcp",
          "headers": {
            "x-api-key": "pk_live_YOUR_POLYMARKET_KEY"
          }
        },
        "quantish-kalshi": {
          "type": "http",
          "url": "https://kalshi-mcp-production-7c2c.up.railway.app/mcp",
          "headers": {
            "x-api-key": "pk_kalshi_YOUR_KALSHI_KEY"
          }
        }
      }
    }
  }
}
```

---

## Funding Your Wallets

### Polymarket (Polygon)
1. Get your deposit address: `get_deposit_addresses`
2. Send USDC (or ETH/MATIC) to your Polygon address
3. Deposits auto-convert to USDC.e for trading

### Kalshi (Solana)
1. Get your deposit address: `kalshi_get_deposit_address`
2. Send USDC to your Solana address
3. Keep ~0.01 SOL for transaction fees
4. Use `kalshi_swap_usdc_to_sol` if you need SOL

---

## Support

- GitHub Issues: [quantish-platform](https://github.com/boshjerns/quantish-platform-private/issues)
- Documentation: [quantish.live](https://quantish.live)
