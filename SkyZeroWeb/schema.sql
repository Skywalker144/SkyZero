-- SkyZero leaderboard schema (D1 / SQLite).
-- Apply with:  npx wrangler d1 execute skyzero --remote --file=./schema.sql
--        (and  --local  for the local dev database)

-- One row per claimed nickname. `token` is a server-minted secret the client
-- stores in localStorage; submissions must present it, so nobody else can post
-- under your name (Plan A: nickname-only, honour system).
CREATE TABLE IF NOT EXISTS users (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  nickname       TEXT NOT NULL,
  nickname_lower TEXT NOT NULL UNIQUE,   -- case-insensitive uniqueness
  token          TEXT NOT NULL,
  created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One row per (user, game): the user's personal best for that game. The
-- leaderboard is just the top of this table per `game`. Keeping only the best
-- (instead of every play) keeps the table tiny and the query trivial.
CREATE TABLE IF NOT EXISTS scores (
  user_id    INTEGER NOT NULL,
  game       TEXT NOT NULL,             -- '2048' | 'dodge'
  best       INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (user_id, game),
  FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Leaderboard read path: top N by best for a game.
CREATE INDEX IF NOT EXISTS idx_scores_game_best ON scores(game, best DESC);
