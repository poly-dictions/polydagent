/**
 * Dynamic.xyz Wallet Integration for Polydictions
 * Solana wallet connection using Dynamic SDK
 */

import { createClient } from '@dynamic-labs/client';
import { SolanaExtension } from '@dynamic-labs/solana-extension';
import { SolanaWalletConnectors } from '@dynamic-labs/solana';
import { WebExtension } from '@dynamic-labs/web-extension';

// Dynamic.xyz Configuration
const DYNAMIC_ENV_ID = 'ea528e4a-63e9-4d21-97e5-1ff2b8a24a85';

// Initialize Dynamic client with Solana support
const dynamicClient = createClient({
  environmentId: DYNAMIC_ENV_ID,
  appName: 'Polydictions',
  appLogoUrl: 'https://polydictions.xyz/images/logo.png',
  walletConnectors: [SolanaWalletConnectors],
})
  .extend(WebExtension())
  .extend(SolanaExtension());

// State
let isInitialized = false;
let currentWallet = null;

/**
 * Initialize Dynamic SDK
 */
async function initialize() {
  if (isInitialized) return;

  try {
    console.log('[Dynamic] Initializing SDK...');

    // Listen for auth state changes
    dynamicClient.auth.on('authChanged', (event) => {
      console.log('[Dynamic] Auth changed:', event);
      if (event.isAuthenticated) {
        handleAuthenticated();
      } else {
        handleLogout();
      }
    });

    // Listen for wallet connection
    dynamicClient.wallets.on('primaryChanged', (wallet) => {
      console.log('[Dynamic] Primary wallet changed:', wallet);
      currentWallet = wallet;
      if (wallet) {
        window.dispatchEvent(new CustomEvent('dynamicWalletConnected', {
          detail: { address: wallet.address, chain: wallet.chain }
        }));
      }
    });

    isInitialized = true;
    console.log('[Dynamic] SDK initialized successfully');

    // Check if already authenticated
    if (dynamicClient.auth.isAuthenticated) {
      handleAuthenticated();
    }

  } catch (error) {
    console.error('[Dynamic] Initialization error:', error);
    throw error;
  }
}

/**
 * Handle successful authentication
 */
function handleAuthenticated() {
  const wallet = dynamicClient.wallets.primary;
  if (wallet) {
    currentWallet = wallet;
    console.log('[Dynamic] Authenticated with wallet:', wallet.address);
    window.dispatchEvent(new CustomEvent('dynamicWalletConnected', {
      detail: {
        address: wallet.address,
        chain: wallet.chain,
        authToken: dynamicClient.auth.token
      }
    }));
  }
}

/**
 * Handle logout
 */
function handleLogout() {
  currentWallet = null;
  console.log('[Dynamic] Logged out');
  window.dispatchEvent(new CustomEvent('dynamicWalletDisconnected'));
}

/**
 * Show Dynamic connect modal
 */
async function connect() {
  await initialize();

  try {
    console.log('[Dynamic] Opening connect modal...');
    await dynamicClient.ui.auth.show();
  } catch (error) {
    console.error('[Dynamic] Connect error:', error);
    throw error;
  }
}

/**
 * Disconnect wallet
 */
async function disconnect() {
  try {
    console.log('[Dynamic] Disconnecting...');
    await dynamicClient.auth.logout();
    currentWallet = null;
  } catch (error) {
    console.error('[Dynamic] Disconnect error:', error);
    throw error;
  }
}

/**
 * Get connected wallet address
 */
function getWalletAddress() {
  return currentWallet?.address || null;
}

/**
 * Get JWT auth token for backend verification
 */
function getAuthToken() {
  return dynamicClient.auth.token || null;
}

/**
 * Check if wallet is connected
 */
function isConnected() {
  return dynamicClient.auth.isAuthenticated && currentWallet !== null;
}

/**
 * Get Solana connection for RPC calls
 */
function getSolanaConnection() {
  if (!currentWallet) return null;
  return dynamicClient.solana.getConnection({ commitment: 'confirmed' });
}

/**
 * Get Solana signer for transactions
 */
function getSolanaSigner() {
  if (!currentWallet) return null;
  return dynamicClient.solana.getSigner({ wallet: currentWallet });
}

// Export to global scope for vanilla JS usage
window.DynamicWallet = {
  initialize,
  connect,
  disconnect,
  getWalletAddress,
  getAuthToken,
  isConnected,
  getSolanaConnection,
  getSolanaSigner,
  client: dynamicClient
};

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initialize);
} else {
  initialize();
}

export {
  initialize,
  connect,
  disconnect,
  getWalletAddress,
  getAuthToken,
  isConnected,
  getSolanaConnection,
  getSolanaSigner,
  dynamicClient
};
