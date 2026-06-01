// POST /api/submit  { token, game, score }  →  { name, best, rank }
// Records a score for the token's user. Only ever raises the per-(user,game)
// best, never lowers it. `rank` is 1 + the number of users with a strictly
// higher best for that game.
import { json, err, cleanScore } from './_util.js';

export async function onRequestPost({ request, env }) {
  let body;
  try { body = await request.json(); } catch (_) { return err('invalid json'); }

  const token = body && body.token;
  if (!token) return err('token required', 401);

  const game = body && body.game;
  const { score, error } = cleanScore(body && body.score, game);
  if (error) return err(error);

  const user = await env.DB.prepare(
    'SELECT id, nickname FROM users WHERE token = ?'
  ).bind(token).first();
  if (!user) return err('invalid token', 403);

  // Upsert: insert the first time, then keep the higher of old/new best.
  await env.DB.prepare(
    `INSERT INTO scores (user_id, game, best, updated_at)
       VALUES (?, ?, ?, datetime('now'))
     ON CONFLICT(user_id, game) DO UPDATE SET
       best = MAX(scores.best, excluded.best),
       updated_at = CASE WHEN excluded.best > scores.best
                         THEN excluded.updated_at ELSE scores.updated_at END`
  ).bind(user.id, game, score).run();

  const row = await env.DB.prepare(
    'SELECT best FROM scores WHERE user_id = ? AND game = ?'
  ).bind(user.id, game).first();
  const best = row ? row.best : score;

  const rankRow = await env.DB.prepare(
    'SELECT COUNT(*) AS higher FROM scores WHERE game = ? AND best > ?'
  ).bind(game, best).first();
  const rank = (rankRow ? rankRow.higher : 0) + 1;

  return json({ name: user.nickname, best, rank });
}
