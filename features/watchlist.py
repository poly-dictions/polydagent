"""
Watchlist feature - allows users to save favorite events for quick access
"""
import json
from pathlib import Path
from typing import Dict, List, Set

WATCHLIST_FILE = "watchlist.json"

class Watchlist:
    def __init__(self):
        self.user_watchlists: Dict[int, List[str]] = {}
        self.load()

    def load(self):
        """Load watchlists from file"""
        if Path(WATCHLIST_FILE).exists():
            try:
                with open(WATCHLIST_FILE, 'r') as f:
                    data = json.load(f)
                    self.user_watchlists = {int(k): v for k, v in data.items()}
            except Exception as e:
                print(f"Error loading watchlist: {e}")

    def save(self):
        """Save watchlists to file"""
        try:
            with open(WATCHLIST_FILE, 'w') as f:
                json.dump(self.user_watchlists, f, indent=2)
        except Exception as e:
            print(f"Error saving watchlist: {e}")

    def add(self, user_id: int, event_slug: str) -> bool:
        """Add event to user's watchlist"""
        if user_id not in self.user_watchlists:
            self.user_watchlists[user_id] = []

        if event_slug in self.user_watchlists[user_id]:
            return False  # Already in watchlist

        self.user_watchlists[user_id].append(event_slug)
        self.save()
        return True

    def remove(self, user_id: int, event_slug: str) -> bool:
        """Remove event from user's watchlist"""
        if user_id not in self.user_watchlists:
            return False

        if event_slug not in self.user_watchlists[user_id]:
            return False

        self.user_watchlists[user_id].remove(event_slug)
        self.save()
        return True

    def get(self, user_id: int) -> List[str]:
        """Get user's watchlist"""
        return self.user_watchlists.get(user_id, [])

    def clear(self, user_id: int):
        """Clear user's watchlist"""
        if user_id in self.user_watchlists:
            del self.user_watchlists[user_id]
            self.save()
