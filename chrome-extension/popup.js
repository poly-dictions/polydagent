// Polymarket API
const API_URL = 'https://gamma-api.polymarket.com';
const LOCAL_API = 'http://localhost:8765';

// Store all events globally for filtering
let allEvents = [];

// Check if running in full page mode (not popup)
function isFullPageMode() {
  // Check if opened as a tab (chrome-extension:// URL in main window)
  const isTab = window.location.href.includes('chrome-extension://') &&
                (window.innerWidth > 450 || document.documentElement.clientWidth > 450);
  return isTab;
}

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
  // Detect full page mode
  if (isFullPageMode()) {
    document.body.classList.add('fullpage');
  }

  setupTabs();
  setupEventListeners();
  await loadData();

  // Re-check on resize
  window.addEventListener('resize', () => {
    if (window.innerWidth > 500) {
      document.body.classList.add('fullpage');
    } else {
      document.body.classList.remove('fullpage');
    }
  });
});

// Tab switching
function setupTabs() {
  const tabs = document.querySelectorAll('.nav-item');
  const contents = document.querySelectorAll('.content');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetTab = tab.dataset.tab;

      tabs.forEach(t => t.classList.remove('active'));
      contents.forEach(c => c.classList.remove('active'));

      tab.classList.add('active');
      document.getElementById(targetTab).classList.add('active');
    });
  });
}

// Event listeners
function setupEventListeners() {
  const refreshBtn = document.getElementById('refreshBtn');
  refreshBtn.addEventListener('click', async () => {
    refreshBtn.classList.add('loading');
    await loadData();
    setTimeout(() => refreshBtn.classList.remove('loading'), 500);
  });

  // Open in new tab
  const openFullBtn = document.getElementById('openFullBtn');
  if (openFullBtn) {
    openFullBtn.addEventListener('click', () => {
      chrome.tabs.create({ url: chrome.runtime.getURL('popup.html') });
    });
  }

  document.getElementById('clearWatchlist').addEventListener('click', clearWatchlist);
}

// Load data from Polymarket API via background script (avoids CORS)
async function loadData() {
  // Show loading state
  document.getElementById('totalEvents').textContent = '...';
  document.getElementById('totalVolume').textContent = '...';
  document.getElementById('activeMarkets').textContent = '...';

  let hotEvents = [];
  let recentEvents = [];

  // Try fetching via background script
  try {
    const hotResponse = await chrome.runtime.sendMessage({ action: 'fetchHotEvents' });
    console.log('Hot response:', hotResponse);
    if (hotResponse?.success && hotResponse.data) {
      hotEvents = hotResponse.data;
    }
  } catch (e) {
    console.error('fetchHotEvents failed:', e);
  }

  try {
    const recentResponse = await chrome.runtime.sendMessage({ action: 'fetchRecentEvents' });
    console.log('Recent response:', recentResponse);
    if (recentResponse?.success && recentResponse.data) {
      recentEvents = recentResponse.data;
    }
  } catch (e) {
    console.error('fetchRecentEvents failed:', e);
  }

  // If background failed, show reload message
  if (!hotEvents.length) {
    console.error('No events loaded - background script may need reload');
    showError('Reload extension');
    return;
  }

  console.log(`Loaded ${hotEvents.length} hot + ${recentEvents.length} recent events`);

  allEvents = hotEvents;

  updateOverviewStats(hotEvents);
  updateCategories(hotEvents);
  updateHotMarkets(hotEvents);
  updateNewMarkets(recentEvents);
  updateTrendingEvents(hotEvents);
  syncAndUpdateWatchlist();
}

// Show error state
function showError(msg = '') {
  document.getElementById('totalEvents').textContent = msg || '-';
  document.getElementById('totalVolume').textContent = '-';
  document.getElementById('activeMarkets').textContent = '-';
}

// Update overview statistics
function updateOverviewStats(events) {
  const totalEvents = events.length;
  const totalVolume = events.reduce((sum, e) => sum + parseFloat(e.volume || 0), 0);
  const totalMarkets = events.reduce((sum, e) => sum + (e.markets?.length || 0), 0);

  document.getElementById('totalEvents').textContent = totalEvents;
  document.getElementById('totalVolume').textContent = formatVolume(totalVolume);
  document.getElementById('activeMarkets').textContent = totalMarkets;
}

// Update categories with real counts
function updateCategories(events) {
  const categories = {
    'Crypto': { count: 0, volume: 0, keywords: /btc|bitcoin|eth|ethereum|crypto|sol|solana|xrp|doge|bnb|ada|dot/ },
    'Politics': { count: 0, volume: 0, keywords: /trump|biden|election|president|congress|senate|vote|governor|democrat|republican/ },
    'Sports': { count: 0, volume: 0, keywords: /nfl|nba|mlb|soccer|football|sports|game|championship|super bowl|ufc|boxing/ },
    'Tech': { count: 0, volume: 0, keywords: /ai|tech|tesla|apple|google|nvidia|openai|microsoft|meta|amazon/ },
    'Finance': { count: 0, volume: 0, keywords: /stock|fed|inflation|interest rate|gdp|recession|s&p|nasdaq|dow/ },
    'Other': { count: 0, volume: 0, keywords: null }
  };

  events.forEach(event => {
    const title = (event.title || event.question || '').toLowerCase();
    const vol = parseFloat(event.volume || 0);
    let matched = false;

    for (const [cat, data] of Object.entries(categories)) {
      if (data.keywords && title.match(data.keywords)) {
        data.count++;
        data.volume += vol;
        matched = true;
        break;
      }
    }

    if (!matched) {
      categories['Other'].count++;
      categories['Other'].volume += vol;
    }
  });

  const categoriesList = document.getElementById('categoriesList');
  categoriesList.innerHTML = '';

  // Sort by volume, show top 4
  const sorted = Object.entries(categories)
    .filter(([_, data]) => data.count > 0)
    .sort((a, b) => b[1].volume - a[1].volume)
    .slice(0, 4);

  sorted.forEach(([name, data]) => {
    const cat = document.createElement('a');
    cat.href = '#';
    cat.className = 'category';
    cat.dataset.category = name.toLowerCase();
    cat.innerHTML = `
      <span class="category-name">${name}</span>
      <span class="category-meta">${data.count} events - ${formatVolume(data.volume)}</span>
    `;
    cat.addEventListener('click', (e) => {
      e.preventDefault();
      filterByCategory(name, events);
    });
    categoriesList.appendChild(cat);
  });
}

// Filter events by category
function filterByCategory(categoryName, events) {
  const categories = {
    'Crypto': /btc|bitcoin|eth|ethereum|crypto|sol|solana|xrp|doge|bnb|ada|dot/,
    'Politics': /trump|biden|election|president|congress|senate|vote|governor|democrat|republican/,
    'Sports': /nfl|nba|mlb|soccer|football|sports|game|championship|super bowl|ufc|boxing/,
    'Tech': /ai|tech|tesla|apple|google|nvidia|openai|microsoft|meta|amazon/,
    'Finance': /stock|fed|inflation|interest rate|gdp|recession|s&p|nasdaq|dow/
  };

  const pattern = categories[categoryName];
  let filtered;

  if (pattern) {
    filtered = events.filter(e => {
      const title = (e.title || e.question || '').toLowerCase();
      return title.match(pattern);
    });
  } else {
    filtered = events.filter(e => {
      const title = (e.title || e.question || '').toLowerCase();
      return !Object.values(categories).some(p => title.match(p));
    });
  }

  // Switch to trending tab and show filtered
  document.querySelectorAll('.nav-item').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.content').forEach(c => c.classList.remove('active'));
  document.querySelector('[data-tab="trending"]').classList.add('active');
  document.getElementById('trending').classList.add('active');

  const trendingList = document.getElementById('trendingList');
  trendingList.innerHTML = '';

  const sorted = filtered
    .sort((a, b) => parseFloat(b.volume || 0) - parseFloat(a.volume || 0))
    .slice(0, 15);

  sorted.forEach(event => {
    const card = createEventCard(event);
    trendingList.appendChild(card);
  });
}

// Update hot markets (high volume)
function updateHotMarkets(events) {
  const activityList = document.getElementById('activityList');
  activityList.innerHTML = '';

  const hotEvents = [...events]
    .sort((a, b) => parseFloat(b.volume || 0) - parseFloat(a.volume || 0))
    .slice(0, 5);

  hotEvents.forEach((event, index) => {
    const item = createFeedItem(event, index === 0 ? 'hot' : (index < 3 ? 'new' : ''));
    activityList.appendChild(item);
  });
}

// Update new markets - show newest events from Polymarket API (sorted by createdAt)
function updateNewMarkets(recentEvents) {
  const newMarketsList = document.getElementById('newMarketsList');
  newMarketsList.innerHTML = '';

  if (recentEvents && recentEvents.length > 0) {
    // Show top 6 newest events
    recentEvents.slice(0, 6).forEach((event) => {
      const item = createNewMarketItem(event);
      newMarketsList.appendChild(item);
    });
  } else {
    newMarketsList.innerHTML = '<div class="empty-small">No new markets</div>';
  }
}

// Create item for bot-posted market (with time ago)
function createPostedMarketItem(event) {
  const item = document.createElement('div');
  item.className = 'feed-item';

  const title = event.title || 'Unknown';
  const postedAt = event.posted_at ? new Date(event.posted_at) : new Date();
  const timeAgo = getTimeAgo(postedAt);

  item.innerHTML = `
    <span class="feed-dot new"></span>
    <span class="feed-text">${title.length > 35 ? title.substring(0, 35) + '...' : title}</span>
    <span class="feed-time">${timeAgo}</span>
  `;

  item.style.cursor = 'pointer';
  item.addEventListener('click', () => {
    window.open(`https://polymarket.com/event/${event.slug}`, '_blank');
  });

  return item;
}

// Create feed item for markets from bot API (with posted_at timestamp)
function createNewMarketItemFromBot(event) {
  const item = document.createElement('div');
  item.className = 'feed-item';

  const title = event.title || 'Unknown';
  const postedAt = new Date(event.posted_at);
  const timeAgo = getTimeAgo(postedAt);

  item.innerHTML = `
    <span class="feed-dot new"></span>
    <span class="feed-text">${title.length > 35 ? title.substring(0, 35) + '...' : title}</span>
    <span class="feed-time">${timeAgo}</span>
  `;

  item.style.cursor = 'pointer';
  item.addEventListener('click', () => {
    window.open(`https://polymarket.com/event/${event.slug}`, '_blank');
  });

  return item;
}

// Create feed item for new markets (fallback - show volume)
function createNewMarketItem(event) {
  const item = document.createElement('div');
  item.className = 'feed-item';

  const title = event.title || event.question || 'Unknown';
  const volume = parseFloat(event.volume || 0);
  const volDisplay = formatVolume(volume);

  item.innerHTML = `
    <span class="feed-dot new"></span>
    <span class="feed-text">${title.length > 35 ? title.substring(0, 35) + '...' : title}</span>
    <span class="feed-time">${volDisplay}</span>
  `;

  item.style.cursor = 'pointer';
  item.addEventListener('click', () => {
    window.open(`https://polymarket.com/event/${event.slug}`, '_blank');
  });

  return item;
}

// Get human readable time ago
function getTimeAgo(date) {
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'now';
  if (diffMins < 60) return `${diffMins}m`;
  if (diffHours < 24) return `${diffHours}h`;
  if (diffDays < 7) return `${diffDays}d`;
  return `${Math.floor(diffDays / 7)}w`;
}

// Create feed item
function createFeedItem(event, dotClass) {
  const item = document.createElement('div');
  item.className = 'feed-item';

  const vol = parseFloat(event.volume || 0);
  const title = event.title || event.question || 'Unknown';

  item.innerHTML = `
    <span class="feed-dot ${dotClass}"></span>
    <span class="feed-text">${title.length > 35 ? title.substring(0, 35) + '...' : title}</span>
    <span class="feed-time">${formatVolume(vol)}</span>
  `;

  item.style.cursor = 'pointer';
  item.addEventListener('click', () => {
    window.open(`https://polymarket.com/event/${event.slug}`, '_blank');
  });

  return item;
}

// Update trending events
function updateTrendingEvents(events) {
  const trendingList = document.getElementById('trendingList');
  trendingList.innerHTML = '';

  const topEvents = [...events]
    .sort((a, b) => parseFloat(b.volume || 0) - parseFloat(a.volume || 0))
    .slice(0, 10);

  topEvents.forEach(event => {
    const card = createEventCard(event);
    trendingList.appendChild(card);
  });
}

// Create event card
function createEventCard(event) {
  const card = document.createElement('article');
  card.className = 'event';

  const markets = event.markets || [];
  let probability = 0;

  if (markets.length > 0) {
    const market = markets[0];
    try {
      const prices = typeof market.outcomePrices === 'string'
        ? JSON.parse(market.outcomePrices)
        : market.outcomePrices;
      probability = parseFloat(prices?.[0] || 0) * 100;
    } catch (e) {
      probability = 0;
    }
  }

  const watchlist = getWatchlist();
  const isWatched = watchlist.includes(event.slug || event.id);

  card.innerHTML = `
    <div class="event-top">
      <h3 class="event-title">${event.title || event.question}</h3>
      <button class="star-btn ${isWatched ? 'active' : ''}" data-event-slug="${event.slug}" title="Add to watchlist">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="${isWatched ? 'currentColor' : 'none'}" stroke="currentColor" stroke-width="2">
          <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
        </svg>
      </button>
    </div>
    <div class="event-data">
      <div class="data-item">
        <span class="data-value">${probability.toFixed(0)}%</span>
        <span class="data-label">Yes</span>
      </div>
      <div class="data-item">
        <span class="data-value">${formatVolume(event.volume)}</span>
        <span class="data-label">Vol</span>
      </div>
      <div class="data-item">
        <span class="data-value">${markets.length}</span>
        <span class="data-label">Markets</span>
      </div>
    </div>
    <div class="event-bottom">
      <span class="tag">${getCategory(event)}</span>
      <a href="https://polymarket.com/event/${event.slug}" target="_blank" class="event-link">Open</a>
    </div>
  `;

  // Watchlist button
  const starBtn = card.querySelector('.star-btn');
  starBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    toggleWatchlist(event.slug, event);
    starBtn.classList.toggle('active');

    const svg = starBtn.querySelector('svg');
    svg.setAttribute('fill', starBtn.classList.contains('active') ? 'currentColor' : 'none');

    updateWatchlistUI();
  });

  return card;
}

// Watchlist functions
function getWatchlist() {
  const watchlist = localStorage.getItem('watchlist');
  return watchlist ? JSON.parse(watchlist) : [];
}

function getWatchlistEvents() {
  const events = localStorage.getItem('watchlistEvents');
  return events ? JSON.parse(events) : {};
}

function toggleWatchlist(slug, event) {
  let watchlist = getWatchlist();
  let events = getWatchlistEvents();

  if (watchlist.includes(slug)) {
    watchlist = watchlist.filter(s => s !== slug);
    delete events[slug];
  } else {
    watchlist.push(slug);
    events[slug] = event;
  }

  localStorage.setItem('watchlist', JSON.stringify(watchlist));
  localStorage.setItem('watchlistEvents', JSON.stringify(events));

  // Try to sync with bot API
  syncWatchlistToBot(watchlist);
}

// Sync watchlist to bot API
async function syncWatchlistToBot(watchlist) {
  const userId = localStorage.getItem('telegramUserId');
  if (!userId) return;

  try {
    await fetch(`${LOCAL_API}/api/watchlist/${userId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ slugs: watchlist })
    });
  } catch (error) {
    console.log('Bot API not available for sync');
  }
}

// Sync watchlist from bot API
async function syncWatchlistFromBot() {
  const userId = localStorage.getItem('telegramUserId');
  if (!userId) return null;

  try {
    const response = await fetch(`${LOCAL_API}/api/watchlist/${userId}`);
    const data = await response.json();
    if (data.success && data.watchlist) {
      return data.watchlist;
    }
  } catch (error) {
    console.log('Bot API not available');
  }
  return null;
}

// Sync and update watchlist
async function syncAndUpdateWatchlist() {
  // Try to sync from bot
  const botWatchlist = await syncWatchlistFromBot();

  if (botWatchlist && botWatchlist.length > 0) {
    // Merge with local watchlist
    const localWatchlist = getWatchlist();
    const merged = [...new Set([...localWatchlist, ...botWatchlist])];
    localStorage.setItem('watchlist', JSON.stringify(merged));

    // Fetch event data for bot watchlist items
    for (const slug of botWatchlist) {
      const events = getWatchlistEvents();
      if (!events[slug]) {
        // Find event in allEvents
        const event = allEvents.find(e => e.slug === slug);
        if (event) {
          events[slug] = event;
          localStorage.setItem('watchlistEvents', JSON.stringify(events));
        }
      }
    }
  }

  updateWatchlistUI();
}

function updateWatchlistUI() {
  const watchlist = getWatchlist();
  const events = getWatchlistEvents();
  const watchlistList = document.getElementById('watchlistList');
  const emptyState = document.getElementById('emptyWatchlist');

  if (watchlist.length === 0) {
    watchlistList.classList.add('hidden');
    emptyState.classList.remove('hidden');
  } else {
    emptyState.classList.add('hidden');
    watchlistList.classList.remove('hidden');
    watchlistList.innerHTML = '';

    watchlist.forEach(slug => {
      let event = events[slug];

      // Try to find fresh data from allEvents
      const freshEvent = allEvents.find(e => e.slug === slug);
      if (freshEvent) {
        event = freshEvent;
        events[slug] = freshEvent;
      }

      if (event) {
        const card = createEventCard(event);
        watchlistList.appendChild(card);
      }
    });

    localStorage.setItem('watchlistEvents', JSON.stringify(events));
  }
}

function clearWatchlist() {
  localStorage.setItem('watchlist', JSON.stringify([]));
  localStorage.setItem('watchlistEvents', JSON.stringify({}));
  updateWatchlistUI();

  // Sync clear to bot
  syncWatchlistToBot([]);

  // Update star buttons
  document.querySelectorAll('.star-btn.active').forEach(btn => {
    btn.classList.remove('active');
    const svg = btn.querySelector('svg');
    svg.setAttribute('fill', 'none');
  });
}

// Helper functions
function formatVolume(volume) {
  const vol = parseFloat(volume || 0);
  if (vol >= 1000000000) {
    return `$${(vol / 1000000000).toFixed(2)}B`;
  } else if (vol >= 1000000) {
    return `$${(vol / 1000000).toFixed(1)}M`;
  } else if (vol >= 1000) {
    return `$${(vol / 1000).toFixed(0)}K`;
  } else {
    return `$${vol.toFixed(0)}`;
  }
}

function getCategory(event) {
  const title = (event.title || event.question || '').toLowerCase();

  if (title.match(/btc|bitcoin|eth|ethereum|crypto|sol|solana/)) return 'Crypto';
  if (title.match(/trump|biden|election|politics|president|congress/)) return 'Politics';
  if (title.match(/nfl|nba|mlb|soccer|football|sports|game/)) return 'Sports';
  if (title.match(/stock|market|fed|inflation|finance/)) return 'Finance';
  if (title.match(/ai|tech|tesla|apple|google|nvidia/)) return 'Tech';

  return 'Other';
}
