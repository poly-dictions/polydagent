"""
Features module for Polydictions bot
"""
from .watchlist import Watchlist
from .categories import Categories, CATEGORY_KEYWORDS
from .alerts import Alerts, PriceAlert
from .news_tracker import NewsTracker

__all__ = ['Watchlist', 'Categories', 'CATEGORY_KEYWORDS', 'Alerts', 'PriceAlert', 'NewsTracker']
