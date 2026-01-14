"""
Quantish Multi-User Wallet Management
Each user gets their own Quantish API key and wallet
"""
import asyncio
import aiohttp
import sqlite3
import json
from pathlib import Path
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
import os
import base64
from hashlib import sha256

# Quantish MCP endpoints
QUANTISH_POLYMARKET_URL = "https://quantish-sdk-production.up.railway.app/mcp"
QUANTISH_KALSHI_URL = "https://kalshi-mcp-production-7c2c.up.railway.app/mcp"
QUANTISH_DISCOVERY_URL = "https://quantish.live/mcp"
QUANTISH_DISCOVERY_KEY = "qm_dyvSSGPYLcRwLXBNmW5VPa94sjCVAe-6"

DB_PATH = Path("quantish_users.db")

# Encryption for API keys
def get_encryption_key() -> bytes:
    secret = os.getenv("SESSION_SECRET", "default-quantish-secret-change-me")
    return base64.urlsafe_b64encode(sha256(secret.encode()).digest())

fernet = Fernet(get_encryption_key())

def encrypt(data: str) -> str:
    return fernet.encrypt(data.encode()).decode()

def decrypt(data: str) -> str:
    return fernet.decrypt(data.encode()).decode()


def init_db():
    """Initialize database for user wallets"""
    conn = sqlite3.connect(DB_PATH)
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS user_wallets (
            user_id TEXT NOT NULL,
            platform TEXT NOT NULL,
            api_key_encrypted TEXT NOT NULL,
            api_secret_encrypted TEXT,
            wallet_address TEXT,
            wallet_status TEXT DEFAULT 'pending',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, platform)
        );
        
        CREATE INDEX IF NOT EXISTS idx_user_platform ON user_wallets(user_id, platform);
    ''')
    conn.commit()
    conn.close()

init_db()


class QuantishUserManager:
    """Manage per-user Quantish wallets"""
    
    async def _call_mcp(self, url: str, api_key: str, tool: str, args: dict) -> dict:
        """Make MCP call"""
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool, "arguments": args},
            "id": 1
        }
        headers = {"Content-Type": "application/json", "x-api-key": api_key}
        timeout = aiohttp.ClientTimeout(total=120)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                data = await resp.json()
                if "error" in data:
                    return {"error": data["error"]}
                content = data.get("result", {}).get("content", [{}])
                if content:
                    text = content[0].get("text", "{}")
                    return json.loads(text)
                return data.get("result", {})
    
    # =========================================================================
    # POLYMARKET
    # =========================================================================
    
    async def create_polymarket_wallet(self, user_id: str) -> Dict[str, Any]:
        """Create new Polymarket wallet for user"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if already exists
        cursor.execute(
            "SELECT wallet_address, wallet_status FROM user_wallets WHERE user_id=? AND platform='polymarket'",
            (user_id,)
        )
        existing = cursor.fetchone()
        if existing and existing[1] == 'ready':
            conn.close()
            return {"success": True, "wallet": existing[0], "status": "already_exists"}
        
        # Use timestamp suffix to ensure unique externalId
        import time
        timestamp = int(time.time())
        external_id = f"polydictions_{user_id}_{timestamp}"
        
        try:
            # Step 1: Request API key
            result = await self._call_mcp(
                QUANTISH_POLYMARKET_URL,
                "temp",  # No key needed for signup
                "request_api_key",
                {"externalId": external_id, "keyName": f"User {user_id[:8]}"}
            )
            
            if "error" in result:
                conn.close()
                return {"error": f"Failed to create API key: {result['error']}"}
            
            api_key = result.get("apiKey")
            api_secret = result.get("apiSecret", "")
            
            if not api_key:
                conn.close()
                return {"error": "No API key returned"}
            
            # Save API key (encrypted)
            cursor.execute('''
                INSERT OR REPLACE INTO user_wallets (user_id, platform, api_key_encrypted, api_secret_encrypted, wallet_status)
                VALUES (?, 'polymarket', ?, ?, 'pending')
            ''', (user_id, encrypt(api_key), encrypt(api_secret) if api_secret else None))
            conn.commit()
            
            # Step 2: Setup wallet
            wallet_result = await self._call_mcp(
                QUANTISH_POLYMARKET_URL,
                api_key,
                "setup_wallet",
                {}
            )
            
            if "error" in wallet_result:
                conn.close()
                return {"error": f"Failed to setup wallet: {wallet_result['error']}", "api_key_created": True}
            
            wallet_address = wallet_result.get("safeAddress", wallet_result.get("address", ""))
            
            # Update with wallet address
            cursor.execute('''
                UPDATE user_wallets SET wallet_address=?, wallet_status='ready'
                WHERE user_id=? AND platform='polymarket'
            ''', (wallet_address, user_id))
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "platform": "polymarket",
                "wallet": wallet_address,
                "network": "Polygon",
                "deposit_tokens": ["USDC.e", "MATIC"]
            }
            
        except Exception as e:
            conn.close()
            return {"error": str(e)}
    
    async def create_kalshi_wallet(self, user_id: str) -> Dict[str, Any]:
        """Create new Kalshi wallet for user"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if already exists in our DB
        cursor.execute(
            "SELECT wallet_address, wallet_status FROM user_wallets WHERE user_id=? AND platform='kalshi'",
            (user_id,)
        )
        existing = cursor.fetchone()
        if existing and existing[1] == 'ready':
            conn.close()
            return {"success": True, "wallet": existing[0], "status": "already_exists"}
        
        # Try with timestamp suffix to ensure unique externalId
        import time
        timestamp = int(time.time())
        external_id = f"polydictions_{user_id}_{timestamp}"
        
        try:
            # Kalshi signup
            result = await self._call_mcp(
                QUANTISH_KALSHI_URL,
                "temp",
                "kalshi_signup",
                {"externalId": external_id, "keyName": f"User {user_id[:8]}"}
            )
            
            if "error" in result:
                conn.close()
                return {"error": f"Failed to create Kalshi account: {result['error']}"}
            
            api_key = result.get("apiKey")
            wallet_address = result.get("publicKey", "")
            
            if not api_key:
                conn.close()
                return {"error": "No API key returned"}
            
            # Save to DB
            cursor.execute('''
                INSERT OR REPLACE INTO user_wallets (user_id, platform, api_key_encrypted, wallet_address, wallet_status)
                VALUES (?, 'kalshi', ?, ?, 'ready')
            ''', (user_id, encrypt(api_key), wallet_address))
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "platform": "kalshi",
                "wallet": wallet_address,
                "network": "Solana",
                "deposit_tokens": ["USDC", "SOL (for gas)"]
            }
            
        except Exception as e:
            conn.close()
            return {"error": str(e)}
    
    def get_user_wallet(self, user_id: str, platform: str) -> Optional[Dict[str, Any]]:
        """Get user's wallet info"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT wallet_address, wallet_status, created_at FROM user_wallets WHERE user_id=? AND platform=?",
            (user_id, platform)
        )
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        return {
            "wallet": row[0],
            "status": row[1],
            "created_at": row[2]
        }
    
    def _get_api_key(self, user_id: str, platform: str) -> Optional[str]:
        """Get decrypted API key for user"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT api_key_encrypted FROM user_wallets WHERE user_id=? AND platform=? AND wallet_status='ready'",
            (user_id, platform)
        )
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        return decrypt(row[0])
    
    # =========================================================================
    # TRADING (per-user)
    # =========================================================================
    
    async def get_balances(self, user_id: str, platform: str) -> Dict[str, Any]:
        """Get user's balances"""
        api_key = self._get_api_key(user_id, platform)
        if not api_key:
            return {"error": "Wallet not found. Create one first."}
        
        if platform == "polymarket":
            return await self._call_mcp(QUANTISH_POLYMARKET_URL, api_key, "get_balances", {})
        elif platform == "kalshi":
            return await self._call_mcp(QUANTISH_KALSHI_URL, api_key, "kalshi_get_balances", {})
        return {"error": "Unknown platform"}
    
    async def get_positions(self, user_id: str, platform: str) -> Dict[str, Any]:
        """Get user's positions"""
        api_key = self._get_api_key(user_id, platform)
        if not api_key:
            return {"error": "Wallet not found"}
        
        if platform == "polymarket":
            return await self._call_mcp(QUANTISH_POLYMARKET_URL, api_key, "get_positions", {})
        elif platform == "kalshi":
            return await self._call_mcp(QUANTISH_KALSHI_URL, api_key, "kalshi_get_positions", {})
        return {"error": "Unknown platform"}
    
    async def place_order(self, user_id: str, platform: str, **kwargs) -> Dict[str, Any]:
        """Place order for user"""
        api_key = self._get_api_key(user_id, platform)
        if not api_key:
            return {"error": "Wallet not found"}
        
        if platform == "polymarket":
            return await self._call_mcp(QUANTISH_POLYMARKET_URL, api_key, "place_order", kwargs)
        elif platform == "kalshi":
            action = kwargs.get("action", "buy_yes")
            ticker = kwargs.get("ticker", "")
            amount = kwargs.get("amount", 1)
            
            if not ticker:
                return {"error": "ticker required for Kalshi trades"}
            
            # Use kalshi_buy which auto-fetches outcomeMint
            if action in ["buy_yes", "buy_no"]:
                side = "YES" if action == "buy_yes" else "NO"
                return await self.kalshi_buy(user_id, ticker, side, amount)
            elif action == "sell":
                outcome_mint = kwargs.get("outcomeMint", "")
                token_amount = kwargs.get("tokenAmount", amount)
                return await self.kalshi_sell(user_id, outcome_mint, token_amount)
        return {"error": "Unknown platform"}
    
    async def transfer(self, user_id: str, platform: str, to_address: str, amount: float, token: str = "USDC") -> Dict[str, Any]:
        """Transfer funds from user's wallet"""
        api_key = self._get_api_key(user_id, platform)
        if not api_key:
            return {"error": "Wallet not found"}
        
        if platform == "polymarket":
            tool = "transfer_usdc" if token == "USDC" else f"transfer_{token.lower()}"
            return await self._call_mcp(QUANTISH_POLYMARKET_URL, api_key, tool, {"toAddress": to_address, "amount": amount})
        elif platform == "kalshi":
            tool = "kalshi_send_usdc" if token == "USDC" else "kalshi_send_sol"
            return await self._call_mcp(QUANTISH_KALSHI_URL, api_key, tool, {"toAddress": to_address, "amount": amount})
        return {"error": "Unknown platform"}
    
    # =========================================================================
    # DISCOVERY (shared, no per-user key needed)
    # =========================================================================
    
    async def search_markets(self, query: str, platform: str = "all", limit: int = 10) -> Dict[str, Any]:
        """Search markets (uses shared discovery key)"""
        return await self._call_mcp(
            QUANTISH_DISCOVERY_URL,
            QUANTISH_DISCOVERY_KEY,
            "search_markets",
            {"query": query, "platform": platform, "limit": limit}
        )
    
    async def get_trending(self, platform: str = "all", limit: int = 10) -> Dict[str, Any]:
        """Get trending markets"""
        return await self._call_mcp(
            QUANTISH_DISCOVERY_URL,
            QUANTISH_DISCOVERY_KEY,
            "get_trending_markets",
            {"platform": platform, "limit": limit}
        )
    
    # =========================================================================
    # KALSHI-SPECIFIC METHODS
    # =========================================================================
    
    async def kalshi_search_markets(self, user_id: str, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search Kalshi markets"""
        api_key = self._get_api_key(user_id, "kalshi")
        if not api_key:
            return {"error": "Kalshi wallet not found"}
        return await self._call_mcp(QUANTISH_KALSHI_URL, api_key, "kalshi_search_markets", {"query": query, "limit": limit})
    
    async def kalshi_get_market(self, user_id: str, ticker: str) -> Dict[str, Any]:
        """Get Kalshi market details"""
        api_key = self._get_api_key(user_id, "kalshi")
        if not api_key:
            return {"error": "Kalshi wallet not found"}
        return await self._call_mcp(QUANTISH_KALSHI_URL, api_key, "kalshi_get_market", {"ticker": ticker})
    
    async def kalshi_get_quote(self, user_id: str, ticker: str, side: str, amount: float) -> Dict[str, Any]:
        """Get quote for Kalshi trade"""
        api_key = self._get_api_key(user_id, "kalshi")
        if not api_key:
            return {"error": "Kalshi wallet not found"}
        return await self._call_mcp(QUANTISH_KALSHI_URL, api_key, "kalshi_get_quote", {
            "marketTicker": ticker,
            "side": side,
            "usdcAmount": amount
        })
    
    async def kalshi_buy(self, user_id: str, ticker: str, side: str, amount: float) -> Dict[str, Any]:
        """Buy on Kalshi market"""
        api_key = self._get_api_key(user_id, "kalshi")
        if not api_key:
            return {"error": "Kalshi wallet not found"}
        
        # Get market details to find outcome mint
        market = await self._call_mcp(QUANTISH_KALSHI_URL, api_key, "kalshi_get_market", {"ticker": ticker})
        if "error" in market:
            return market
        
        if side.upper() == "YES":
            mint = market.get("yesOutcomeMint", "")
            if not mint:
                return {"error": "Could not find YES outcome mint"}
            return await self._call_mcp(QUANTISH_KALSHI_URL, api_key, "kalshi_buy_yes", {
                "marketTicker": ticker,
                "yesOutcomeMint": mint,
                "usdcAmount": amount
            })
        else:
            mint = market.get("noOutcomeMint", "")
            if not mint:
                return {"error": "Could not find NO outcome mint"}
            return await self._call_mcp(QUANTISH_KALSHI_URL, api_key, "kalshi_buy_no", {
                "marketTicker": ticker,
                "noOutcomeMint": mint,
                "usdcAmount": amount
            })
    
    async def kalshi_sell(self, user_id: str, outcome_mint: str, token_amount: float) -> Dict[str, Any]:
        """Sell Kalshi position"""
        api_key = self._get_api_key(user_id, "kalshi")
        if not api_key:
            return {"error": "Kalshi wallet not found"}
        return await self._call_mcp(QUANTISH_KALSHI_URL, api_key, "kalshi_sell_position", {
            "outcomeMint": outcome_mint,
            "tokenAmount": token_amount
        })
    
    async def kalshi_redeem(self, user_id: str) -> Dict[str, Any]:
        """Redeem all winning positions on Kalshi"""
        api_key = self._get_api_key(user_id, "kalshi")
        if not api_key:
            return {"error": "Kalshi wallet not found"}
        return await self._call_mcp(QUANTISH_KALSHI_URL, api_key, "kalshi_redeem_all_positions", {})


# Global instance
quantish_users = QuantishUserManager()
