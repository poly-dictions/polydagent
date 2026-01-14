#!/usr/bin/env python3
"""
Simple test - just create a poll without market
"""

from web3 import Web3
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

# Setup
w3 = Web3(Web3.HTTPProvider(os.getenv("SONIC_RPC_URL")))
account = w3.eth.account.from_key(os.getenv("PANDORA_PRIVATE_KEY"))
oracle_addr = Web3.to_checksum_address(os.getenv("PANDORA_ORACLE_ADDRESS"))

print("="*60)
print("SIMPLE POLL CREATION TEST")
print("="*60)
print(f"Wallet: {account.address}")
print(f"Balance: {w3.from_wei(w3.eth.get_balance(account.address), 'ether')} S")
print(f"Oracle: {oracle_addr}\n")

# Minimal ABI - just createPoll (PAYABLE - requires 1.1 S)
ORACLE_ABI = [{
    "inputs": [
        {"name": "question", "type": "string"},
        {"name": "deadline", "type": "uint256"},
        {"name": "details", "type": "string"}
    ],
    "name": "createPoll",
    "outputs": [{"name": "pollId", "type": "uint256"}],
    "stateMutability": "payable",
    "type": "function"
}]

# Poll creation fee (observed from blockchain)
POLL_FEE = w3.to_wei(1.1, 'ether')  # 1.1 S

oracle = w3.eth.contract(address=oracle_addr, abi=ORACLE_ABI)

# Test poll
question = "Will Bitcoin reach $150k by Q1 2026?"
deadline = int((datetime.now() + timedelta(days=90)).timestamp())
details = "Test poll from Polydictions agent"

print(f"Question: {question}")
print(f"Deadline: {deadline} ({datetime.fromtimestamp(deadline)})")
print(f"Details: {details}\n")

try:
    print("Estimating gas...")
    gas_estimate = oracle.functions.createPoll(
        question, deadline, details
    ).estimate_gas({'from': account.address, 'value': POLL_FEE})

    print(f"Gas estimate: {gas_estimate}")
    print(f"Poll fee: {w3.from_wei(POLL_FEE, 'ether')} S")

    print("\nBuilding transaction...")
    tx = oracle.functions.createPoll(
        question, deadline, details
    ).build_transaction({
        'from': account.address,
        'nonce': w3.eth.get_transaction_count(account.address),
        'gas': int(gas_estimate * 1.5),
        'gasPrice': w3.eth.gas_price,
        'value': POLL_FEE
    })

    print("Signing...")
    signed_tx = account.sign_transaction(tx)

    print("Sending transaction...")
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    print(f"TX Hash: {tx_hash.hex()}")

    print("\nWaiting for receipt...")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    if receipt['status'] == 1:
        print("\n[SUCCESS] Poll created!")
        print(f"Block: {receipt['blockNumber']}")
        print(f"Gas used: {receipt['gasUsed']}")

        if receipt['logs']:
            print(f"\nLogs found: {len(receipt['logs'])}")
            for i, log in enumerate(receipt['logs']):
                print(f"  Log {i}: {log['topics'][0].hex() if log['topics'] else 'no topics'}")
        else:
            print("No logs found")
    else:
        print("\n[FAILED] Transaction reverted")

except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
