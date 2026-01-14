#!/usr/bin/env python3
from web3 import Web3
import os
from dotenv import load_dotenv

load_dotenv()

w3 = Web3(Web3.HTTPProvider(os.getenv("SONIC_RPC_URL")))
account = w3.eth.account.from_key(os.getenv("PANDORA_PRIVATE_KEY"))

print(f"Wallet: {account.address}")
balance = w3.eth.get_balance(account.address)
print(f"Balance: {w3.from_wei(balance, 'ether')} S")
print(f"Gas Price: {w3.from_wei(w3.eth.gas_price, 'gwei')} Gwei")
