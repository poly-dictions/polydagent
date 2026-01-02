/**
 * Polydictions API Client
 */

// Auto-detect API URL: use localhost for dev, production URL otherwise
const API_BASE = window.location.hostname === 'localhost'
    ? 'http://localhost:8765'
    : 'https://polydictions-production.up.railway.app';

const api = {
    /**
     * Fetch events from Polymarket
     */
    async getEvents(limit = 2000, sort = 'createdAt') {
        try {
            const url = `${API_BASE}/api/events?limit=${limit}&sort=${sort}&filter=false`;
            console.log(`Fetching events from ${url}`);
            const response = await fetch(url);
            console.log('Response status:', response.status);
            if (!response.ok) throw new Error(`Failed to fetch events: ${response.status} ${response.statusText}`);
            const data = await response.json();
            console.log(`Loaded ${data.length} events`);
            return data;
        } catch (error) {
            console.error('API Error:', error);
            throw error; // Re-throw to show error in UI
        }
    },

    /**
     * Get user's watchlist
     */
    async getWatchlist(userId) {
        try {
            const response = await fetch(`${API_BASE}/api/watchlist/${userId}`);
            const data = await response.json();
            return data.success ? data.watchlist : [];
        } catch (error) {
            console.error('Watchlist Error:', error);
            return [];
        }
    },

    /**
     * Update user's watchlist
     */
    async updateWatchlist(userId, slugs) {
        try {
            const response = await fetch(`${API_BASE}/api/watchlist/${userId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ slugs })
            });
            const data = await response.json();
            return data.success;
        } catch (error) {
            console.error('Watchlist Update Error:', error);
            return false;
        }
    },

    /**
     * Get new markets
     */
    async getNewMarkets() {
        try {
            const response = await fetch(`${API_BASE}/api/new-markets`);
            const data = await response.json();
            return data.success ? data.events : [];
        } catch (error) {
            console.error('New Markets Error:', error);
            return [];
        }
    }
};

/**
 * Utility functions
 */
const utils = {
    /**
     * Format currency
     */
    formatCurrency(value) {
        if (value >= 1000000) {
            return '$' + (value / 1000000).toFixed(1) + 'M';
        } else if (value >= 1000) {
            return '$' + (value / 1000).toFixed(1) + 'K';
        }
        return '$' + value.toFixed(0);
    },

    /**
     * Format percentage
     */
    formatPercent(value) {
        return (value * 100).toFixed(0) + '%';
    },

    /**
     * Format date
     */
    formatDate(dateStr) {
        if (!dateStr) return 'N/A';
        const date = new Date(dateStr);
        const now = new Date();
        const diff = date - now;

        if (diff < 0) return 'Ended';

        const days = Math.floor(diff / (1000 * 60 * 60 * 24));
        if (days > 30) {
            return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        } else if (days > 0) {
            return days + 'd left';
        } else {
            const hours = Math.floor(diff / (1000 * 60 * 60));
            return hours + 'h left';
        }
    },

    /**
     * Debounce function
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * Get category from tags
     */
    getCategory(event) {
        // Tags can be objects with label/slug or strings - all lowercase for case-insensitive matching
        const tags = (event.tags || []).map(t => {
            if (typeof t === 'string') return t.toLowerCase();
            return (t.label || t.slug || '').toLowerCase();
        });
        const title = (event.title || '').toLowerCase();
        const slug = (event.slug || '').toLowerCase();
        const description = (event.description || '').toLowerCase();

        // Helper function for case-insensitive keyword matching
        const matchesKeyword = (keywords) => keywords.some(k =>
            title.includes(k) || slug.includes(k) || description.includes(k)
        );

        // Sports - check first as it has specific keywords
        const sportsKeywords = ['super bowl', 'nfl', 'nba', 'nhl', 'mlb', 'ufc', 'mma', 'boxing',
            'premier league', 'champions league', 'world cup', 'tennis', 'golf', 'f1', 'formula',
            'olympics', 'championship', 'playoff', 'finals', 'champion', 'uefa', 'fifa',
            'basketball', 'football', 'soccer', 'baseball', 'hockey', 'counter-strike', 'cs2',
            'dota', 'league of legends', 'valorant', 'esports', ' vs '];
        if (tags.some(t => t.includes('sport')) || matchesKeyword(sportsKeywords)) {
            return 'sports';
        }

        // Culture - check early because of explicit Culture tag from Polymarket
        const cultureKeywords = ['movie', 'oscar', 'grammy', 'emmy', 'album', 'song', 'netflix',
            'tv show', 'celebrity', 'kardashian', 'taylor swift', 'kanye', 'mrbeast', 'youtube',
            'influencer', 'viral', 'tiktok', 'streaming', 'box office', 'award', 'gta', 'game of',
            'disney', 'marvel', 'dc ', 'anime', 'spotify', 'podcast', 'reality tv', 'squid game',
            'weather', 'temperature', 'precipitation', 'earthquake', 'tsa passenger'];
        if (tags.some(t => t.includes('culture') || t.includes('entertainment') || t.includes('pop culture')) || matchesKeyword(cultureKeywords)) {
            return 'culture';
        }

        // Tech - check before finance (some overlap with crypto)
        const techKeywords = ['ai ', 'artificial intelligence', 'openai', 'chatgpt', 'gpt-', 'grok',
            'spacex', 'starship', 'tesla', 'apple', 'google', 'microsoft', 'amazon', 'meta',
            'iphone', 'android', 'software', 'hardware', 'robot', 'tech',
            'startup', 'silicon valley', 'neuralink', 'quantum', 'semiconductor', 'chip'];
        if (tags.some(t => t.includes('tech') || t.includes('science') || t.includes('ai')) || matchesKeyword(techKeywords)) {
            return 'tech';
        }

        // Finance - check before politics (fed/rate cut are finance)
        const financeKeywords = ['stock', 'fed ', 'federal reserve', 'rate cut', 'inflation', 'gdp',
            'economy', 'treasury', 'dollar', 'recession', 'indices', 'equities', 'ipo', 's&p',
            'nasdaq', 'dow jones', 'bank', 'central bank', 'interest rate', 'fdv', 'market cap',
            'acquisitions', 'commodities', 'trading', 'tariff', 'trade war', 'jobs report',
            'unemployment', 'cpi', 'ppi', 'yield', 'bond', 'forex', 'currency', 'company',
            'largest company', 'market value', 'valuation', 'revenue', 'earnings', 'profit',
            'q1', 'q2', 'q3', 'q4', 'orders', 'reserve', 'quarterly', 'fiscal', 'budget'];
        if (tags.some(t => t.includes('finance') || t.includes('business') || t.includes('economy') || t.includes('indices') || t.includes('equities')) || matchesKeyword(financeKeywords)) {
            return 'finance';
        }

        // Geopolitics - check before politics (more specific international events)
        const geopoliticsKeywords = ['russia', 'ukraine', 'china', 'taiwan', 'israel', 'gaza', 'hamas',
            'iran', 'north korea', 'syria', 'yemen', 'nato', 'un ', 'united nations', 'sanctions',
            'invasion', 'ceasefire', 'war ', 'conflict', 'military', 'troops', 'missile',
            'hezbollah', 'middle east', 'eu ', 'european union', 'brexit', 'territorial'];
        if (tags.some(t => t.includes('geopolitic') || t.includes('international') || t.includes('global')) || matchesKeyword(geopoliticsKeywords)) {
            return 'geopolitics';
        }

        // Politics (US-focused)
        const politicsKeywords = ['trump', 'biden', 'election', 'president', 'congress', 'senate',
            'governor', 'democrat', 'republican', 'vote', 'ballot', 'primary', 'nominee',
            'cabinet', 'minister', 'parliament', 'impeach', 'scotus', 'supreme court'];
        if (tags.some(t => t.includes('politic') || t.includes('election')) || matchesKeyword(politicsKeywords)) {
            return 'politics';
        }

        // Crypto
        const cryptoKeywords = ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'solana', 'sol',
            'dogecoin', 'memecoin', 'token', 'blockchain', 'defi', 'nft', 'altcoin', 'xrp',
            'cardano', 'polygon', 'avalanche', 'binance', 'coinbase'];
        if (tags.some(t => t.includes('crypto')) || matchesKeyword(cryptoKeywords)) {
            return 'crypto';
        }

        return 'other';
    },

    /**
     * Get best price from markets
     */
    getBestPrice(event) {
        if (!event.markets || event.markets.length === 0) return null;

        let bestYes = 0;
        let bestNo = 0;

        for (const market of event.markets) {
            const outcomes = market.outcomes || ['Yes', 'No'];
            const prices = market.outcomePrices ? JSON.parse(market.outcomePrices) : [];

            if (prices.length >= 2) {
                const yesPrice = parseFloat(prices[0]);
                const noPrice = parseFloat(prices[1]);
                if (yesPrice > bestYes) bestYes = yesPrice;
                if (noPrice > bestNo) bestNo = noPrice;
            }
        }

        return { yes: bestYes, no: bestNo };
    },

    /**
     * Get total volume
     */
    getTotalVolume(event) {
        if (!event.markets) return parseFloat(event.volume || 0);

        return event.markets.reduce((sum, m) => {
            return sum + parseFloat(m.volume || 0);
        }, 0);
    }
};

// Export for use in other files
window.api = api;
window.utils = utils;
