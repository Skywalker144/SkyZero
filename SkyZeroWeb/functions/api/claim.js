// POST /api/claim  { name, token? }  →  { name, token }
// Claim a nickname (Plan A: nickname-only identity). On success returns a
// server-minted token the client stores and presents on every submit. If the
// name already exists, a matching token is treated as a no-op "login"; an
// absent/wrong token is a 409 (name taken).
import { json, err, cleanName } from './_util.js';

export async function onRequestPost({ request, env }) {
  let body;
  try { body = await request.json(); } catch (_) { return err('invalid json'); }

  const { name, error } = cleanName(body && body.name);
  if (error) return err(error);
  const lower = name.toLowerCase();

  const existing = await env.DB.prepare(
    'SELECT nickname, token FROM users WHERE nickname_lower = ?'
  ).bind(lower).first();

  if (existing) {
    if (body && body.token && body.token === existing.token) {
      return json({ name: existing.nickname, token: existing.token });
    }
    return err('name taken', 409);
  }

  const token = crypto.randomUUID();
  try {
    await env.DB.prepare(
      'INSERT INTO users (nickname, nickname_lower, token) VALUES (?, ?, ?)'
    ).bind(name, lower, token).run();
  } catch (_) {
    // UNIQUE violation from a race between the SELECT and the INSERT.
    return err('name taken', 409);
  }
  return json({ name, token });
}
