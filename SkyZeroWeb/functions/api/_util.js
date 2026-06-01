// Shared helpers for the leaderboard API. The leading underscore keeps this
// file from being treated as a route by Pages Functions; it is import-only.

// Known games + a loose sanity ceiling on each score. These are NOT
// tamper-proof anti-cheat (the games run client-side) — they only reject
// obviously-garbage values. See LEADERBOARD_SETUP.md for the threat model.
//   2048:  SUM_CAP=262144, MAX_TILE=131072 → real scores top out a few ×10^6.
//   dodge: +1..+N per dodged projectile over a survival run → ~10^4 realistic.
export const GAMES = {
  '2048':  { max: 100_000_000 },
  'dodge': { max: 10_000_000 },
};

export const NAME_MAX = 20;

export function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'content-type': 'application/json; charset=utf-8',
      // API responses must never be edge/browser cached (the site-wide
      // _headers rule caches /* for an hour).
      'cache-control': 'no-store',
    },
  });
}

export function err(message, status = 400) {
  return json({ error: message }, status);
}

// Validate + normalise a display nickname. Returns {name} or {error}.
export function cleanName(raw) {
  if (typeof raw !== 'string') return { error: 'name required' };
  // Drop ASCII control chars (0x00-0x1F and 0x7F), trim, collapse whitespace.
  const stripped = Array.from(raw)
    .filter((ch) => { const c = ch.charCodeAt(0); return c > 0x1f && c !== 0x7f; })
    .join('');
  const name = stripped.trim().replace(/\s+/g, ' ');
  if (name.length < 1) return { error: 'name empty' };
  if (name.length > NAME_MAX) return { error: 'name too long' };
  return { name };
}

// Validate a submitted score against the game's ceiling. Returns {score} or {error}.
export function cleanScore(raw, game) {
  const g = GAMES[game];
  if (!g) return { error: 'unknown game' };
  const n = Number(raw);
  if (!Number.isInteger(n) || n < 0) return { error: 'bad score' };
  if (n > g.max) return { error: 'score out of range' };
  return { score: n };
}
