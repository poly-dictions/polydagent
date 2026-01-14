#!/usr/bin/env python3
"""
Inspect PredictionOracle contract to find the correct function signatures
"""

from web3 import Web3
import os
from dotenv import load_dotenv

load_dotenv()

w3 = Web3(Web3.HTTPProvider(os.getenv("SONIC_RPC_URL")))
oracle_addr = Web3.to_checksum_address(os.getenv("PANDORA_ORACLE_ADDRESS"))

# Get contract bytecode
code = w3.eth.get_code(oracle_addr)
print(f"Oracle contract: {oracle_addr}")
print(f"Contract size: {len(code)} bytes\n")

# Try to extract function selectors from bytecode
print("Looking for 'createPoll' function selector...")

# The function selector for createPoll(string,uint256,string) should be:
# First 4 bytes of keccak256("createPoll(string,uint256,string)")
from eth_utils import keccak

sig = "createPoll(string,uint256,string)"
selector = keccak(text=sig).hex()[:10]
print(f"Expected selector for {sig}: {selector}")

# Try alternative signatures
alternatives = [
    "createPoll(string,uint256,string,address)",
    "createPoll(string,uint256,string,address,uint256)",
    "createPoll(string,uint256)",
    "createMarket(string,uint256,string)",
]

for alt in alternatives:
    sel = keccak(text=alt).hex()[:10]
    print(f"Alternative {alt}: {sel}")

print("\n" + "="*60)
print("Trying to find recent createPoll transactions...")
print("="*60)
