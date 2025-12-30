"""
Category filters - filter events by category (crypto, politics, sports, etc.)
"""
import json
from pathlib import Path
from typing import Dict, List, Set

CATEGORIES_FILE = "user_categories.json"

# Category keywords mapping
CATEGORY_KEYWORDS = {
    "crypto": ["btc", "bitcoin", "eth", "ethereum", "crypto", "solana", "xrp", "blockchain", "defi", "nft", "token", "coin"],
    "politics": ["election", "president", "senate", "congress", "vote", "trump", "biden", "political", "government", "democrat", "republican"],
    "sports": ["nfl", "nba", "mlb", "nhl", "football", "basketball", "baseball", "hockey", "soccer", "vs.", "vs", "game", "match", "championship"],
    "finance": ["stock", "market", "fed", "rate", "inflation", "gdp", "economy", "treasury", "dollar", "recession"],
    "tech": ["ai", "apple", "google", "meta", "tesla", "microsoft", "amazon", "tech", "software", "app"],
    "entertainment": ["movie", "oscar", "grammy", "emmy", "celebrity", "actor", "music", "album", "box office"]
}

class Categories:
    def __init__(self):
        self.user_categories: Dict[int, List[str]] = {}
        self.load()

    def load(self):
        """Load user categories from file"""
        if Path(CATEGORIES_FILE).exists():
            try:
                with open(CATEGORIES_FILE, 'r') as f:
                    data = json.load(f)
                    self.user_categories = {int(k): v for k, v in data.items()}
            except Exception as e:
                print(f"Error loading categories: {e}")

    def save(self):
        """Save user categories to file"""
        try:
            with open(CATEGORIES_FILE, 'w') as f:
                json.dump(self.user_categories, f, indent=2)
        except Exception as e:
            print(f"Error saving categories: {e}")

    def set_categories(self, user_id: int, categories: List[str]) -> bool:
        """Set user's category filters"""
        valid_categories = [c.lower() for c in categories if c.lower() in CATEGORY_KEYWORDS]

        if not valid_categories:
            return False

        self.user_categories[user_id] = valid_categories
        self.save()
        return True

    def get_categories(self, user_id: int) -> List[str]:
        """Get user's category filters"""
        return self.user_categories.get(user_id, [])

    def clear_categories(self, user_id: int):
        """Clear user's category filters"""
        if user_id in self.user_categories:
            del self.user_categories[user_id]
            self.save()

    def matches_category(self, event_data: dict, user_id: int) -> bool:
        """Check if event matches user's category filters"""
        user_cats = self.get_categories(user_id)

        if not user_cats:
            return True  # No filters = show all

        title = event_data.get('title', '').lower()
        markets = event_data.get('markets', [])
        market_text = ' '.join([m.get('question', '').lower() for m in markets])
        searchable = f"{title} {market_text}"

        # Check if event matches any of user's categories
        for category in user_cats:
            keywords = CATEGORY_KEYWORDS.get(category, [])
            for keyword in keywords:
                if keyword in searchable:
                    return True

        return False

    @staticmethod
    def get_all_categories() -> List[str]:
        """Get list of all available categories"""
        return list(CATEGORY_KEYWORDS.keys())
