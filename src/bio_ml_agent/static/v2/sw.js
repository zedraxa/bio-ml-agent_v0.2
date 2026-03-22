const CACHE_NAME = 'agentos-v2';
const ASSETS = [
  './index.html',
  'https://cdn.tailwindcss.com',
  'https://unpkg.com/lucide@latest',
  'https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS))
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  
  // API Requests: Stale-while-revalidate
  const isApi = url.pathname.startsWith('/api/') || 
                url.pathname.startsWith('/missions/') || 
                url.pathname.startsWith('/projects/') ||
                url.pathname === '/notifications';

  if (isApi) {
    event.respondWith(
      caches.open('api-cache').then((cache) => {
        return cache.match(event.request).then((cachedResponse) => {
          const fetchPromise = fetch(event.request).then((networkResponse) => {
            if (networkResponse.ok) {
              cache.put(event.request, networkResponse.clone());
            }
            return networkResponse;
          }).catch(() => cachedResponse); // Fallback to cache on network failure
          return cachedResponse || fetchPromise;
        });
      })
    );
  } else {
    // Static Assets: Cache-first
    event.respondWith(
      caches.match(event.request).then((response) => {
        return response || fetch(event.request);
      })
    );
  }
});
