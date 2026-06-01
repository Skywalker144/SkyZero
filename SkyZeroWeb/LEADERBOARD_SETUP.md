# Leaderboard setup (Cloudflare Pages + D1)

A nickname-only ("Plan A") user system + leaderboard for **2048** and
**Channel Dodge**. No login, no passwords: a player picks a nickname once, the
server mints a secret token the browser stores, and scores are posted with that
token. Gomoku is intentionally excluded (no continuous score).

## What's in the repo

| File | Role |
|------|------|
| `functions/api/claim.js` | `POST /api/claim` — claim a nickname, get a token |
| `functions/api/submit.js` | `POST /api/submit` — report a score (token-gated) |
| `functions/api/leaderboard.js` | `GET /api/leaderboard?game=…` — top N + your rank |
| `functions/api/_util.js` | shared validation / helpers (not a route) |
| `schema.sql` | D1 tables (`users`, `scores`) |
| `wrangler.toml` | Pages config + D1 binding (`DB`) |
| `leaderboard.js` | front-end: 🏆 button, ranking modal, name dialog, `SkzLB.submit()` |

The games call `window.SkzLB.submit('2048' | 'dodge', score)` at game-over.
2048 skips runs where the board was hand-edited via the Advanced panel.

## One-time setup

```bash
# wrangler comes via npx; no global install needed
cd /home/sky/RL/SkyZero/SkyZeroWeb

# 1. Create the D1 database, then paste the printed database_id into wrangler.toml
npx wrangler d1 create skyzero

# 2. Create the tables — local (for dev) and remote (production)
npx wrangler d1 execute skyzero --local  --file=./schema.sql
npx wrangler d1 execute skyzero --remote --file=./schema.sql
```

Then make sure the Cloudflare **Pages project's "build output directory" is the
repo root (`/`)** so it matches `pages_build_output_dir = "."` in `wrangler.toml`.
(Alternative to `wrangler.toml`: add the D1 binding named `DB` in the Pages
dashboard under Settings → Functions → D1 bindings.)

## Local development

`python3 -m http.server` will **not** run the API (the 🏆 modal shows
"排行榜暂不可用" — by design). To exercise the full stack locally:

```bash
npx wrangler pages dev .
# serves the static site + functions/ + the --local D1 database
```

## Deploy

`git push` — Cloudflare Pages auto-builds. The leaderboard goes live once the
D1 database exists and its `database_id` is in `wrangler.toml`. Until then the
site works exactly as before and the modal degrades gracefully.

## Threat model (honest)

The games run entirely client-side, so a determined user can forge a score with
devtools. Server-side defenses here are deliberately light: per-game sanity
ceilings (`functions/api/_util.js`) and token-gated writes (you can't post under
someone else's name). This is an honour-system board.

If abuse appears, the cheapest next step is **Cloudflare Turnstile** (free,
invisible): add a widget on the page, send its token with `/api/submit`, and
verify server-side via `https://challenges.cloudflare.com/turnstile/v0/siteverify`.
The only truly cheat-proof option is replaying the full input sequence
server-side, which means porting each game's logic into the Function — much more
work, not recommended for a hobby site.

## Resetting / inspecting data

```bash
npx wrangler d1 execute skyzero --remote --command "SELECT * FROM users;"
npx wrangler d1 execute skyzero --remote --command "DELETE FROM scores WHERE game='2048';"
```
