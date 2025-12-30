"""
News Tracker - monitors Market Context updates for watchlist events
Notifies users when there are significant news/context changes
"""
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

NEWS_CACHE_FILE = "news_cache.json"
USER_INTERVALS_FILE = "user_intervals.json"

# Default and minimum interval in seconds
DEFAULT_INTERVAL = 300  # 5 minutes
MIN_INTERVAL = 180  # 3 minutes minimum

class NewsTracker:
    def __init__(self):
        # Structure: {event_slug: {"hash": str, "context": str, "updated": str}}
        self.context_cache: Dict[str, Dict] = {}
        # Structure: {user_id: interval_seconds}
        self.user_intervals: Dict[int, int] = {}
        self.load()
        self.load_intervals()

    def load(self):
        """Load cached contexts from file"""
        if Path(NEWS_CACHE_FILE).exists():
            try:
                with open(NEWS_CACHE_FILE, 'r', encoding='utf-8') as f:
                    self.context_cache = json.load(f)
                logger.info(f"Loaded news cache with {len(self.context_cache)} events")
            except Exception as e:
                logger.error(f"Error loading news cache: {e}")

    def load_intervals(self):
        """Load user intervals from file"""
        if Path(USER_INTERVALS_FILE).exists():
            try:
                with open(USER_INTERVALS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.user_intervals = {int(k): v for k, v in data.items()}
                logger.info(f"Loaded intervals for {len(self.user_intervals)} users")
            except Exception as e:
                logger.error(f"Error loading user intervals: {e}")

    def save_intervals(self):
        """Save user intervals to file"""
        try:
            with open(USER_INTERVALS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.user_intervals, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving user intervals: {e}")

    def set_interval(self, user_id: int, minutes: int) -> bool:
        """Set user's update interval in minutes. Returns True if valid."""
        seconds = minutes * 60
        if seconds < MIN_INTERVAL:
            return False
        self.user_intervals[user_id] = seconds
        self.save_intervals()
        logger.info(f"User {user_id} set interval to {minutes} minutes")
        return True

    def get_interval(self, user_id: int) -> int:
        """Get user's update interval in seconds"""
        return self.user_intervals.get(user_id, DEFAULT_INTERVAL)

    def get_interval_minutes(self, user_id: int) -> int:
        """Get user's update interval in minutes"""
        return self.get_interval(user_id) // 60

    def save(self):
        """Save cached contexts to file"""
        try:
            with open(NEWS_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.context_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving news cache: {e}")

    def _hash_context(self, context: str) -> str:
        """Create hash of context for comparison"""
        # Normalize: lowercase, remove extra whitespace, remove punctuation variations
        normalized = ' '.join(context.lower().split())
        # Remove common varying words/phrases that AI changes between requests
        import re
        # Remove time references that change
        normalized = re.sub(r'\b(today|yesterday|this week|last week|recently|currently)\b', '', normalized)
        # Remove filler words
        normalized = re.sub(r'\b(the|a|an|is|are|was|were|has|have|had|been|being)\b', '', normalized)
        # Keep only first 200 chars for comparison (main content)
        normalized = normalized[:200]
        return hashlib.md5(normalized.encode()).hexdigest()

    def check_for_update(self, event_slug: str, new_context: str) -> Optional[Dict]:
        """
        Check if context has changed significantly.
        Returns dict with update info if changed, None if no change.
        """
        if not new_context or len(new_context) < 50:
            return None

        new_hash = self._hash_context(new_context)

        if event_slug not in self.context_cache:
            # First time seeing this event - save but don't notify
            self.context_cache[event_slug] = {
                "hash": new_hash,
                "context": new_context[:500],  # Store truncated for reference
                "updated": datetime.now().isoformat()
            }
            self.save()
            logger.info(f"First context cached for {event_slug}")
            return None

        old_data = self.context_cache[event_slug]
        old_hash = old_data.get("hash", "")

        if new_hash != old_hash:
            # Context changed - update cache and return change info
            old_context = old_data.get("context", "")

            self.context_cache[event_slug] = {
                "hash": new_hash,
                "context": new_context[:500],
                "updated": datetime.now().isoformat()
            }
            self.save()

            logger.info(f"Context updated for {event_slug}")
            return {
                "event_slug": event_slug,
                "old_context": old_context,
                "new_context": new_context,
                "updated": datetime.now().isoformat()
            }

        return None

    def get_cached_context(self, event_slug: str) -> Optional[str]:
        """Get cached context for an event"""
        if event_slug in self.context_cache:
            return self.context_cache[event_slug].get("context")
        return None

    def remove_event(self, event_slug: str):
        """Remove event from cache"""
        if event_slug in self.context_cache:
            del self.context_cache[event_slug]
            self.save()
