#!/usr/bin/env python3
"""
Quick test script for Pandora market creation
"""

from pandora_agent import PandoraAgent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("\n" + "="*60)
    print("PANDORA MARKET CREATOR - TEST MODE")
    print("="*60 + "\n")

    # Initialize agent
    agent = PandoraAgent()

    # Test market parameters
    question = "Will Bitcoin reach $150,000 by end of Q1 2026?"
    deadline_days = 90  # 3 months
    details = "Test market for Pandora platform. This is a prediction market about Bitcoin price reaching $150k by March 31, 2026."

    print(f"Creating test market:")
    print(f"  Question: {question}")
    print(f"  Deadline: {deadline_days} days from now")
    print(f"  Details: {details}")
    print()

    # Create the market
    agent.test_create_market(question, deadline_days, details)

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
