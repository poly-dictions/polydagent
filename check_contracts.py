#!/usr/bin/env python3
"""
Check if Pandora contracts exist on Sonic blockchain
"""

from web3 import Web3
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
SONIC_RPC_URL = os.getenv("SONIC_RPC_URL", "https://rpc.soniclabs.com")
PANDORA_ORACLE_ADDRESS = os.getenv("PANDORA_ORACLE_ADDRESS")
PANDORA_MARKET_FACTORY_ADDRESS = os.getenv("PANDORA_MARKET_FACTORY_ADDRESS")

print("="*60)
print("CHECKING PANDORA CONTRACTS ON SONIC")
print("="*60 + "\n")

# Connect to Sonic
w3 = Web3(Web3.HTTPProvider(SONIC_RPC_URL))
print(f"Connected to Sonic: {w3.is_connected()}")
print(f"Chain ID: {w3.eth.chain_id}")
print(f"Latest block: {w3.eth.block_number}\n")

# Check PredictionOracle
print(f"PredictionOracle Address: {PANDORA_ORACLE_ADDRESS}")
oracle_code = w3.eth.get_code(Web3.to_checksum_address(PANDORA_ORACLE_ADDRESS))
print(f"Contract code length: {len(oracle_code)} bytes")
if len(oracle_code) > 2:
    print("[OK] Contract exists!\n")
else:
    print("[ERROR] No contract found at this address!\n")

# Check MarketFactory
print(f"MarketFactory Address: {PANDORA_MARKET_FACTORY_ADDRESS}")
factory_code = w3.eth.get_code(Web3.to_checksum_address(PANDORA_MARKET_FACTORY_ADDRESS))
print(f"Contract code length: {len(factory_code)} bytes")
if len(factory_code) > 2:
    print("[OK] Contract exists!\n")
else:
    print("[ERROR] No contract found at this address!\n")

print("="*60)
