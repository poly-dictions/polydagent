# Wallet Gate Setup - Arbitrage Page

## Overview
The arbitrage page now requires users to hold $POLY tokens and connect their Phantom wallet to access the content.

## Features

### 🔐 Token Gating
- **Required Token**: `iATcGSt9DhJF9ZiJ6dmR153N7bW2G4J9dSSDxWSpump`
- **Minimum Balance**: 1 token
- **Verification**: Real-time check via Solana RPC

### 👻 Phantom Wallet Integration
- Auto-detect Phantom wallet extension
- Secure connection via Solana web3.js
- Token balance verification
- 24-hour session cache

### 💾 Session Persistence
- Verified wallets cached for 24 hours in localStorage
- No need to reconnect within 24h period
- Automatic expiry and re-verification

## User Flow

1. **User visits `/arbitrage/`**
2. **Wallet Gate appears** (if not previously verified)
3. **User clicks "Connect Phantom Wallet"**
4. **Phantom popup opens** requesting connection approval
5. **Backend verifies** token holdings via Solana RPC
6. **If successful**: Gate disappears, content unlocked, wallet indicator shown
7. **If failed**: Error message with reason (no tokens, insufficient balance, etc.)

## Developer Testing

### Bypass Gate (Development Only)
Access the page with bypass parameter:
```
http://localhost:8765/arbitrage/?bypass=true
```

### Test with Real Wallet
1. Install Phantom wallet extension
2. Fund wallet with at least 1 $POLY token
3. Visit `/arbitrage/` normally
4. Click "Connect Phantom Wallet"

### Clear Cached Verification
```javascript
// In browser console:
localStorage.removeItem('wallet_verified');
location.reload();
```

## Technical Implementation

### Token Verification Flow
```javascript
1. Connect to Phantom wallet
   └─> Get wallet public key

2. Query Solana RPC
   └─> connection.getParsedTokenAccountsByOwner(publicKey, { mint: TOKEN_MINT })

3. Check token balance
   └─> tokenAmount.uiAmount >= MIN_TOKEN_BALANCE

4. If valid:
   └─> Cache wallet address + 24h expiry
   └─> Unlock content
```

### Security Features
- ✅ Client-side token verification via Solana RPC
- ✅ No backend API keys exposed
- ✅ Session expiry after 24 hours
- ✅ Direct blockchain verification (no trusted third party)

### Dependencies
- **Solana Web3.js**: `https://unpkg.com/@solana/web3.js@latest/lib/index.iife.min.js`
- **Phantom Wallet**: Browser extension required

## Configuration

### Change Required Token
Edit in `/arbitrage/index.html`:
```javascript
const REQUIRED_TOKEN = 'YOUR_TOKEN_MINT_ADDRESS';
const MIN_TOKEN_BALANCE = 1; // Minimum tokens required
```

### Change RPC Endpoint
```javascript
const RPC_ENDPOINT = 'https://api.mainnet-beta.solana.com';
// or use custom RPC like QuickNode, Alchemy, etc.
```

### Change Session Duration
```javascript
// In unlockContent() function
const expiry = Date.now() + 24 * 60 * 60 * 1000; // 24 hours
```

## UI Elements

### Wallet Gate Overlay
- Full-screen overlay when not verified
- Phantom logo icon (👻)
- Connect button with gradient purple styling
- Token address display
- Status messages (connecting, checking, errors)

### Wallet Indicator (Top Right)
- Green pulsing dot (verified status)
- Shortened wallet address (e.g., `iATc...pump`)
- Only shown when wallet connected

## Error Messages

| Error | Reason |
|-------|--------|
| "Phantom wallet not found" | Extension not installed |
| "insufficient $POLY balance" | Balance < MIN_TOKEN_BALANCE |
| "connection failed" | User rejected or timeout |
| "No token account found" | Wallet has no token accounts for this mint |

## Troubleshooting

### Gate won't disappear
- Check console for errors
- Verify token mint address is correct
- Ensure RPC endpoint is accessible
- Try clearing localStorage and reconnecting

### "Insufficient balance" but I have tokens
- Verify you're checking the correct token mint
- Check decimals (some tokens have 6, 9 decimals)
- Ensure token account is initialized

### Phantom not detected
- Ensure Phantom extension is installed
- Refresh page after installing
- Check that `window.solana.isPhantom === true`

## Production Deployment

Before deploying to production:

1. **Remove bypass parameter** or restrict to dev environments
2. **Consider rate limiting** RPC calls (avoid spam)
3. **Add analytics** for wallet connections
4. **Monitor RPC costs** if using paid provider
5. **Test on mobile** (Phantom mobile app support)

## Future Improvements

- [ ] Support for multiple wallets (Solflare, Backpack, etc.)
- [ ] NFT gating as alternative to tokens
- [ ] Tiered access based on token amount
- [ ] Staking verification (locked tokens)
- [ ] Discord/Telegram verification alternative
- [ ] QR code for mobile wallet connection

## Support

For issues or questions:
- Twitter: [@polydictions](https://x.com/polydictions)
- Telegram: [polydictions](https://t.me/polydictions)
- GitHub: [Create an issue](https://github.com/poly-dictions/polydictions/issues)
