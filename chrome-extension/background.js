// Background service worker for Polydictions extension

const API_URL = 'https://gamma-api.polymarket.com';

// Check for new events periodically
const CHECK_INTERVAL = 5 * 60 * 1000; // 5 minutes

chrome.runtime.onInstalled.addListener(() => {
  console.log('polydictions extension installed');

  // Set up periodic checks for notifications
  chrome.alarms.create('checkNewEvents', {
    periodInMinutes: 5
  });
});

// Also handle startup
chrome.runtime.onStartup.addListener(() => {
  console.log('polydictions extension started');
});

// Handle messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('Background received message:', request.action);

  const handleRequest = async () => {
    try {
      if (request.action === 'fetchEvents') {
        const data = await fetchEvents(request.params);
        return { success: true, data };
      }
      if (request.action === 'fetchRecentEvents') {
        const data = await fetchRecentEvents();
        return { success: true, data };
      }
      if (request.action === 'fetchHotEvents') {
        const data = await fetchHotEvents();
        return { success: true, data };
      }
      if (request.action === 'fetchPostedEvents') {
        const data = await fetchPostedEvents();
        return { success: true, data };
      }
      return { success: false, error: 'Unknown action' };
    } catch (error) {
      console.error('Background error:', error);
      return { success: false, error: error.message };
    }
  };

  handleRequest().then(sendResponse);
  return true; // Keep channel open for async response
});

// Fetch events from Polymarket API (no CORS issues in background)
async function fetchEvents(params = {}) {
  const queryParams = new URLSearchParams();

  queryParams.set('limit', params.limit || 500);
  queryParams.set('active', params.active !== false ? 'true' : 'false');
  queryParams.set('closed', params.closed === true ? 'true' : 'false');

  if (params.order) {
    queryParams.set('order', params.order);
  }
  if (params.ascending !== undefined) {
    queryParams.set('ascending', params.ascending ? 'true' : 'false');
  }

  const response = await fetch(`${API_URL}/events?${queryParams}`);
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  return response.json();
}

// Fetch 100 most recent events (sorted by createdAt) - FAST
async function fetchRecentEvents() {
  console.log('fetchRecentEvents: Fetching 100 newest...');
  const response = await fetch(`${API_URL}/events?limit=100&active=true&closed=false&order=createdAt&ascending=false`);
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  const events = await response.json();
  console.log(`fetchRecentEvents: Got ${events.length} events`);
  return events;
}

// Fetch top 100 by volume (for Hot Markets and stats)
async function fetchHotEvents() {
  console.log('fetchHotEvents: Fetching top 100 by volume...');
  const response = await fetch(`${API_URL}/events?limit=100&active=true&closed=false&order=volume&ascending=false`);
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  const events = await response.json();
  console.log(`fetchHotEvents: Got ${events.length} events`);
  return events;
}

// Fetch events posted to Telegram channel (same as bot sends)
async function fetchPostedEvents() {
  console.log('fetchPostedEvents: Fetching from bot API...');
  const response = await fetch('http://localhost:8765/api/new-markets');
  if (!response.ok) {
    throw new Error(`Bot API error: ${response.status}`);
  }
  const data = await response.json();
  if (data.success) {
    console.log(`fetchPostedEvents: Got ${data.events?.length || 0} posted events`);
    return data.events || [];
  }
  return [];
}

// Listen for alarm
chrome.alarms?.onAlarm?.addListener((alarm) => {
  if (alarm && alarm.name === 'checkNewEvents') {
    checkNewEvents();
  }
});

// Check for new events
async function checkNewEvents() {
  try {
    const response = await fetch('https://gamma-api.polymarket.com/events?limit=10&active=true');
    const events = await response.json();

    // Get last seen event ID
    const result = await chrome.storage.local.get(['lastEventId']);
    const lastEventId = result.lastEventId;

    if (!lastEventId && events.length > 0) {
      // First run, just save the latest event
      await chrome.storage.local.set({ lastEventId: events[0].id });
      return;
    }

    // Check for new events
    const newEvents = [];
    for (const event of events) {
      if (event.id === lastEventId) break;
      newEvents.push(event);
    }

    if (newEvents.length > 0) {
      // Update last seen
      await chrome.storage.local.set({ lastEventId: newEvents[0].id });

      // Show notification
      chrome.notifications.create({
        type: 'basic',
        iconUrl: 'icons/icon128.png',
        title: 'New Polymarket Event',
        message: `${newEvents.length} new event(s) available!`,
        priority: 1
      });
    }
  } catch (error) {
    console.error('Error checking new events:', error);
  }
}

// Handle notification clicks - open Polymarket in new tab
chrome.notifications.onClicked.addListener(() => {
  chrome.tabs.create({ url: 'https://polymarket.com' });
});
