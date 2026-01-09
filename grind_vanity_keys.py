#!/usr/bin/env python3
"""
Vanity Keypair Generator for pump.fun tokens
Generates keypairs ending with "pump" suffix

Usage:
    python grind_vanity_keys.py [count]

Example:
    python grind_vanity_keys.py 100  # Generate 100 vanity keypairs
"""

import json
import os
import sys
import time
from pathlib import Path

try:
    from solders.keypair import Keypair
except ImportError:
    print("Error: solders not installed. Run: pip install solders")
    sys.exit(1)

VANITY_KEYS_FILE = "vanity_keypairs.json"
TARGET_SUFFIX = "pump"


def load_existing_keys() -> list:
    """Load existing vanity keypairs from file"""
    if os.path.exists(VANITY_KEYS_FILE):
        try:
            with open(VANITY_KEYS_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []


def save_keys(keys: list):
    """Save vanity keypairs to file"""
    with open(VANITY_KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=2)


def grind_vanity_keypair(suffix: str = TARGET_SUFFIX, log_interval: int = 100000) -> dict:
    """Generate a single vanity keypair ending with the given suffix"""
    attempts = 0
    start_time = time.time()

    while True:
        keypair = Keypair()
        pubkey = str(keypair.pubkey())

        if pubkey.endswith(suffix):
            elapsed = time.time() - start_time
            print(f"  Found: {pubkey} (after {attempts:,} attempts, {elapsed:.1f}s)")
            return {
                "pubkey": pubkey,
                "private_key": str(keypair),  # base58 encoded
                "used": False
            }

        attempts += 1
        if attempts % log_interval == 0:
            elapsed = time.time() - start_time
            rate = attempts / elapsed if elapsed > 0 else 0
            print(f"  ... {attempts:,} attempts ({rate:,.0f}/sec)")


def main():
    # Parse arguments
    count = 10  # Default
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            print(f"Invalid count: {sys.argv[1]}")
            sys.exit(1)

    print(f"Vanity Keypair Generator")
    print(f"========================")
    print(f"Target suffix: '{TARGET_SUFFIX}'")
    print(f"Generating: {count} keypairs")
    print()

    # Load existing keys
    keys = load_existing_keys()
    existing_count = len(keys)
    unused_count = sum(1 for k in keys if not k.get("used", False))

    print(f"Existing keypairs: {existing_count} ({unused_count} unused)")
    print()

    # Generate new keypairs
    total_start = time.time()

    for i in range(count):
        print(f"[{i+1}/{count}] Generating keypair...")
        keypair_data = grind_vanity_keypair()
        keys.append(keypair_data)

        # Save after each keypair (in case of interruption)
        save_keys(keys)

    total_elapsed = time.time() - total_start

    print()
    print(f"Done! Generated {count} keypairs in {total_elapsed:.1f}s")
    print(f"Total keypairs: {len(keys)} ({sum(1 for k in keys if not k.get('used', False))} unused)")
    print(f"Saved to: {VANITY_KEYS_FILE}")


if __name__ == "__main__":
    main()
