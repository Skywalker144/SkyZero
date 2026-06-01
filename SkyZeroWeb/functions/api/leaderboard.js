// GET /api/leaderboard?game=2048|dodge[&limit=50][&name=<nick>]
//   → { game, top: [{ rank, name, best, at }], me?: { rank, name, best } }
// `top` is the highest scorers for the game. If `name` is supplied and that
// user sits outside the returned window, `me` carries their own rank/best.
import { json, err, GAMES } from './_util.js';

export async function onRequestGet({ request, env }) {
  const url = new URL(request.url);
  const game = url.searchParams.get('game');
  if (!GAMES[game]) return err('unknown game');

  let limit = parseInt(url.searchParams.get('limit') || '50', 10);
  if (!Number.isFinite(limit) || limit < 1) limit = 50;
  if (limit > 100) limit = 100;

  const top = await env.DB.prepare(
    `SELECT u.nickname AS name, s.best AS best, s.updated_at AS at
       FROM scores s JOIN users u ON u.id = s.user_id
      WHERE s.game = ?
      ORDER BY s.best DESC, s.updated_at ASC
      LIMIT ?`
  ).bind(game, limit).all();

  const rows = (top.results || []).map((r, i) => ({
    rank: i + 1, name: r.name, best: r.best, at: r.at,
  }));

  const resp = { game, top: rows };

  const name = url.searchParams.get('name');
  if (name) {
    const me = await env.DB.prepare(
      `SELECT s.best AS best FROM scores s
         JOIN users u ON u.id = s.user_id
        WHERE s.game = ? AND u.nickname_lower = ?`
    ).bind(game, name.toLowerCase()).first();
    if (me) {
      const rk = await env.DB.prepare(
        'SELECT COUNT(*) AS higher FROM scores WHERE game = ? AND best > ?'
      ).bind(game, me.best).first();
      resp.me = { name, best: me.best, rank: (rk ? rk.higher : 0) + 1 };
    }
  }

  return json(resp);
}
