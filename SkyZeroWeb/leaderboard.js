// SkyZero leaderboard client (Plan A: nickname-only identity).
//
// Drop `<script defer src="/leaderboard.js"></script>` into any page. It
// injects a floating 🏆 entry (a right-edge side tab on desktop, a bottom-right
// FAB on mobile), a 🏆 button on any game-over #overlay, a ranking modal, and a
// name dialog, and exposes `window.SkzLB.submit(game, score)` for the games to
// report scores at game-over. Degrades silently if the /api/* backend is
// absent (e.g. a plain `python -m http.server` dev server).
(function () {
  'use strict';

  var NAME_KEY = 'skz_name';
  var TOKEN_KEY = 'skz_token';
  var GAMES = ['2048', 'dodge'];
  // A game page can pin the board to its own game by setting
  // `window.SKZ_LB_GAME = '2048' | 'dodge'` before this script runs. Then only
  // that game's scores show here (no cross-game tab), and first-time visitors
  // are prompted to set a nickname on arrival. The landing page leaves it
  // unset and keeps both games behind tabs.
  var ONLY_GAME = (function () {
    var g = (typeof window !== 'undefined') ? window.SKZ_LB_GAME : null;
    return GAMES.indexOf(g) >= 0 ? g : null;
  })();
  var VIEW_GAMES = ONLY_GAME ? [ONLY_GAME] : GAMES;

  // A pinned "house champion": SkyZero (the site's own AI) holds rank #1 on
  // every board, shown with the brand rainbow. It is synthetic — never stored in
  // D1 nor fetched — so it always leads and real players are renumbered below it.
  var CHAMPION = { name: 'SkyZero', score: { '2048': 175842, 'dodge': 31961 } };
  var CHAMP_LOWER = CHAMPION.name.toLowerCase();

  // ---------- i18n (reads the site-wide skz_lang) ----------
  var STR = {
    zh: {
      title: '排行榜', open: '排行榜', view: '查看排行榜', close: '关闭',
      tab_2048: '2048', tab_dodge: '躲避',
      you: '你', set_name: '设置昵称', change: '改名',
      name_title: '起个名字', name_ph: '昵称(最多 20 字)',
      name_hint: '昵称用于排行榜显示。成绩仅保存在当前浏览器,换浏览器、换设备或清除缓存后无法同步,需要重新设置。',
      save: '保存', cancel: '取消',
      empty: '还没有记录,快来抢第一!',
      loading: '加载中…', unavailable: '排行榜暂不可用',
      rank: '#', player: '玩家', score: '分数',
      taken: '这个名字被占用了,换一个吧', err: '出错了,稍后再试',
      uploaded: '已上传排行榜 ✓', your_rank: '你的排名',
      outside: '(未上榜)',
    },
    en: {
      title: 'Leaderboard', open: 'Leaderboard', view: 'Leaderboard', close: 'Close',
      tab_2048: '2048', tab_dodge: 'Dodge',
      you: 'You', set_name: 'Set name', change: 'Rename',
      name_title: 'Pick a name', name_ph: 'Nickname (max 20)',
      name_hint: 'Your nickname shows on the leaderboard. Scores live only in this browser — they will not sync to another browser or device, and you will need to set the name again after switching or clearing storage.',
      save: 'Save', cancel: 'Cancel',
      empty: 'No scores yet — be the first!',
      loading: 'Loading…', unavailable: 'Leaderboard unavailable',
      rank: '#', player: 'Player', score: 'Score',
      taken: 'That name is taken, try another', err: 'Something went wrong, try again',
      uploaded: 'Uploaded ✓', your_rank: 'Your rank',
      outside: '(unranked)',
    },
  };
  function lang() {
    try { var l = localStorage.getItem('skz_lang'); return (l === 'en') ? 'en' : 'zh'; }
    catch (_) { return 'zh'; }
  }
  function t(k) { var s = STR[lang()] || STR.zh; return s[k] != null ? s[k] : (STR.en[k] || k); }

  // ---------- identity ----------
  function getName() { try { return localStorage.getItem(NAME_KEY) || ''; } catch (_) { return ''; } }
  function getToken() { try { return localStorage.getItem(TOKEN_KEY) || ''; } catch (_) { return ''; } }
  function setIdentity(name, token) {
    try { localStorage.setItem(NAME_KEY, name); localStorage.setItem(TOKEN_KEY, token); } catch (_) {}
  }

  // Scores reported before a name was set (or while the network was down) are
  // held here and flushed once identity exists.
  var pending = {};            // game -> best score seen this session
  var sessionBest = {};        // game -> best score already POSTed this session

  // ---------- API ----------
  async function api(path, opts) {
    var res = await fetch('/api/' + path, opts);
    var data = null;
    try { data = await res.json(); } catch (_) {}
    if (!res.ok) { var e = new Error((data && data.error) || ('HTTP ' + res.status)); e.status = res.status; throw e; }
    return data;
  }
  function claim(name) {
    return api('claim', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ name: name, token: getToken() || undefined }),
    });
  }
  function postScore(game, score) {
    return api('submit', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ token: getToken(), game: game, score: score }),
    });
  }
  function fetchBoard(game) {
    var q = 'leaderboard?game=' + encodeURIComponent(game);
    var n = getName();
    if (n) q += '&name=' + encodeURIComponent(n);
    return api(q, { method: 'GET' });
  }

  // ---------- public: submit a score ----------
  // Games call this at game-over. Dedupes per session and only POSTs when a
  // name is set; otherwise stashes the score to flush after the name is set.
  function submit(game, score) {
    score = Math.floor(Number(score) || 0);
    if (GAMES.indexOf(game) < 0 || score <= 0) return;
    if (!(pending[game] >= score)) pending[game] = score;
    if (!getName()) { markPending(); return; }
    flush();
  }
  function flush() {
    if (!getName()) return;
    GAMES.forEach(function (game) {
      var score = pending[game];
      if (score == null) return;
      if (sessionBest[game] >= score) return;
      sessionBest[game] = score;
      postScore(game, score).then(function () {
        toast(t('uploaded'));
        refreshViews();   // reflect the new score in the live dock / open modal
      }).catch(function () { /* keep pending for a later retry */ });
    });
  }

  // ---------- DOM: styles ----------
  function injectStyles() {
    if (document.getElementById('skz-lb-style')) return;
    var css =
      // Floating entry. Mobile-first: a circular FAB pinned bottom-right (no
      // layout cost). On wide screens it becomes a vertical tab docked to the
      // right edge, using the empty side gutter.
      '.skz-lb-fab{position:fixed;right:16px;bottom:16px;z-index:55;box-sizing:border-box;' +
        'display:flex;flex-direction:column;align-items:center;justify-content:center;cursor:pointer;' +
        'width:52px;height:52px;padding:0;border-radius:50%;' +
        'background:var(--surface,#fff);border:1px solid var(--border,#d0d7de);' +
        'box-shadow:0 4px 14px rgba(0,0,0,.18);color:var(--fg,#1b1f24);font-size:22px;line-height:1}' +
      '.skz-lb-fab:hover{border-color:var(--border-strong,#afb8c1)}' +
      '.skz-lb-fab-label{display:none}' +
      '.skz-lb-fab .skz-lb-dot{position:absolute;top:-3px;right:-3px;width:10px;height:10px;border-radius:50%;' +
        'background:var(--accent,#3b82f6);border:2px solid var(--surface,#fff);display:none}' +
      '.skz-lb-fab.has-pending .skz-lb-dot{display:block}' +
      '@media (min-width:820px){' +
        '.skz-lb-fab{right:0;top:50%;bottom:auto;transform:translateY(-50%);' +
          'width:auto;height:auto;gap:6px;padding:14px 7px;border-right:none;' +
          'border-radius:12px 0 0 12px;font-size:20px;box-shadow:-2px 4px 16px rgba(0,0,0,.14)}' +
        // writing-mode default (mixed) keeps 排行榜 upright and rotates Latin
        // labels sideways — both read naturally in a vertical tab.
        '.skz-lb-fab-label{display:block;writing-mode:vertical-rl;letter-spacing:3px;font-size:13px;font-weight:700}' +
        '.skz-lb-fab .skz-lb-dot{top:6px;right:6px}' +
      '}' +
      '.skz-lb-overlay-btn{margin-top:2px}' +
      '.skz-lb-mask{position:fixed;inset:0;z-index:60;display:flex;align-items:center;justify-content:center;' +
        'padding:16px;background:color-mix(in srgb,var(--bg,#fff) 72%,transparent);backdrop-filter:blur(3px)}' +
      '.skz-lb-mask[hidden]{display:none}' +
      '.skz-lb-panel{width:100%;max-width:420px;max-height:80vh;display:flex;flex-direction:column;gap:14px;' +
        'padding:18px;background:var(--surface,#fff);border:1px solid var(--border-strong,#afb8c1);' +
        'border-radius:var(--radius-lg,16px);box-shadow:0 12px 40px rgba(0,0,0,.3)}' +
      '.skz-lb-head{display:flex;align-items:center;justify-content:space-between}' +
      '.skz-lb-h{font-size:18px;font-weight:700;color:var(--fg,#1b1f24)}' +
      '.skz-lb-x{cursor:pointer;background:none;border:none;font-size:22px;line-height:1;color:var(--fg-muted,#656d76)}' +
      '.skz-lb-id{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--fg-muted,#656d76)}' +
      '.skz-lb-id b{color:var(--fg,#1b1f24)}' +
      '.skz-lb-id .skz-lb-link{margin-left:auto;cursor:pointer;background:none;border:none;' +
        'color:var(--accent,#3b82f6);font-size:13px;font-weight:600}' +
      '.skz-lb-tabs{display:flex;gap:6px}' +
      '.skz-lb-tab{flex:1;cursor:pointer;padding:7px 10px;border-radius:var(--radius-sm,8px);' +
        'background:var(--surface-2,#f6f8fa);border:1px solid var(--border,#d0d7de);' +
        'color:var(--fg-muted,#656d76);font-size:14px;font-weight:600}' +
      '.skz-lb-tab[aria-selected="true"]{border-color:var(--accent,#3b82f6);color:var(--fg,#1b1f24)}' +
      '.skz-lb-view{display:flex;flex-direction:column;gap:14px;flex:1 1 auto;min-height:0}' +
      '.skz-lb-list{overflow-y:auto;min-height:0;flex:1 1 auto}' +
      // Desktop dock: always-on left panel. Hidden by default; shown only on
      // wide screens where it clears the centered game (≥1200px).
      '.skz-lb-dock{display:none}' +
      '@media (min-width:1200px){' +
        // position:absolute (not fixed) so the dock scrolls with the page and
        // stays anchored just below the topbar — it can never ride up over the
        // brand. positionDock() keeps `top` in document coordinates.
        '.skz-lb-dock{display:flex;flex-direction:column;gap:12px;position:absolute;left:20px;top:72px;' +
          'width:272px;max-height:calc(100vh - 96px);box-sizing:border-box;padding:16px;z-index:40;' +
          'background:var(--surface,#fff);border:1px solid var(--border,#d0d7de);' +
          'border-radius:var(--radius-lg,16px);box-shadow:0 6px 24px rgba(0,0,0,.06)}' +
        // Where a dock exists, the floating FAB and the overlay button are redundant.
        '.skz-lb-has-dock .skz-lb-fab{display:none}' +
        '.skz-lb-has-dock .skz-lb-overlay-btn{display:none}' +
      '}' +
      // Inline panel (e.g. 2048 mobile): in-flow below the board. Hidden while
      // empty (before the script fills it) and on desktop (the dock takes over).
      '.skz-lb-inline{display:flex;flex-direction:column;gap:12px;margin-top:16px;box-sizing:border-box;' +
        'padding:16px;background:var(--surface,#fff);border:1px solid var(--border,#d0d7de);' +
        'border-radius:var(--radius-lg,16px)}' +
      '.skz-lb-inline:empty{display:none}' +
      // Where an inline panel exists, the FAB + overlay button are redundant at
      // every width (inline covers narrow, dock covers wide).
      '.skz-lb-has-inline .skz-lb-fab{display:none}' +
      '.skz-lb-has-inline .skz-lb-overlay-btn{display:none}' +
      '@media (min-width:1200px){.skz-lb-inline{display:none}}' +
      '.skz-lb-row{display:grid;grid-template-columns:36px 1fr auto;align-items:center;gap:10px;' +
        'padding:8px 6px;border-bottom:1px solid var(--border,#eaeef2);font-size:14px;color:var(--fg,#1b1f24)}' +
      '.skz-lb-row:last-child{border-bottom:none}' +
      '.skz-lb-row.me{background:color-mix(in srgb,var(--accent,#3b82f6) 12%,transparent);border-radius:8px}' +
      '.skz-lb-rk{text-align:center;font-variant-numeric:tabular-nums;color:var(--fg-muted,#656d76)}' +
      '.skz-lb-nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}' +
      // Pinned champion wordmark: neutral "Sky" (inherits --fg) + rainbow "Zero",
      // mirroring the .bw-zero brand gradient in style.css (120deg = its slant).
      '.skz-lb-champ-nm{font-weight:700}' +
      '.skz-lb-zero{background:linear-gradient(120deg,' +
        '#fb3640 0%,#fb3640 15%,#fc5132 21.1%,#fe6f11 27.1%,#ff9200 33.2%,' +
        '#ffbb00 39.3%,#f1d525 45.4%,#a6de58 51.4%,#3ddc84 57.5%,#00d8be 63.6%,' +
        '#00cbf4 69.6%,#3ba9ff 75.7%,#5977ff 81.8%,#7554ff 87.9%,#9f4dff 93.9%,' +
        '#c14bff 100%);-webkit-background-clip:text;background-clip:text;' +
        '-webkit-text-fill-color:transparent;color:transparent}' +
      '.skz-lb-sc{font-variant-numeric:tabular-nums;font-weight:700}' +
      '.skz-lb-msg{padding:24px 8px;text-align:center;color:var(--fg-muted,#656d76);font-size:14px}' +
      '.skz-lb-me-out{border-top:1px dashed var(--border,#d0d7de);margin-top:4px;padding-top:8px}' +
      '.skz-lb-form{display:flex;flex-direction:column;gap:10px}' +
      '.skz-lb-form input{padding:9px 11px;border:1px solid var(--border,#d0d7de);border-radius:var(--radius-sm,8px);' +
        'background:var(--surface-sunken,#f6f8fa);color:var(--fg,#1b1f24);font-size:15px;font-family:inherit}' +
      '.skz-lb-form .skz-lb-hint{font-size:12px;color:var(--fg-muted,#656d76);line-height:1.5}' +
      '.skz-lb-form .skz-lb-error{font-size:13px;color:#e5484d;min-height:1em}' +
      '.skz-lb-acts{display:flex;gap:8px;justify-content:flex-end}' +
      '.skz-lb-acts .btn,.skz-lb-form .btn{cursor:pointer;padding:8px 14px;border-radius:var(--radius-sm,8px);' +
        'border:1px solid var(--border,#d0d7de);background:var(--surface,#fff);color:var(--fg,#1b1f24);' +
        'font-size:14px;font-weight:600;font-family:inherit}' +
      '.skz-lb-acts .btn.primary,.skz-lb-form .btn.primary{background:var(--accent,#3b82f6);border-color:var(--accent,#3b82f6);color:#fff}' +
      '.skz-lb-toast{position:fixed;left:50%;bottom:28px;transform:translateX(-50%);z-index:70;' +
        'padding:9px 16px;border-radius:999px;background:var(--fg,#1b1f24);color:var(--bg,#fff);' +
        'font-size:13px;font-weight:600;box-shadow:0 6px 20px rgba(0,0,0,.25);opacity:0;transition:opacity .2s ease}' +
      '.skz-lb-toast.show{opacity:1}';
    var el = document.createElement('style');
    el.id = 'skz-lb-style';
    el.textContent = css;
    document.head.appendChild(el);
  }

  // ---------- DOM: floating entry (side tab on desktop / FAB on mobile) ----------
  var btn;
  function injectButton() {
    if (document.getElementById('skz-lb-open')) return;
    btn = document.createElement('button');
    btn.id = 'skz-lb-open';
    btn.className = 'skz-lb-fab';
    btn.type = 'button';
    btn.innerHTML = '🏆<span class="skz-lb-fab-label"></span><span class="skz-lb-dot"></span>';
    relabelBtn();
    btn.addEventListener('click', openModal);
    document.body.appendChild(btn);
  }
  function relabelBtn() {
    if (!btn) return;
    btn.title = t('open');
    btn.setAttribute('aria-label', t('open'));
    var lab = btn.querySelector('.skz-lb-fab-label');
    if (lab) lab.textContent = t('open');
  }
  function markPending() {
    if (btn) btn.classList.add('has-pending');
  }

  // A second, contextual entry on the game-over overlay — the highest-intent
  // moment to check the board. Both games use #overlay and never rewrite its
  // child list, so appending here survives their state changes; the overlay is
  // hidden during play, so the button only shows when the overlay is up.
  function injectOverlayButton() {
    var ov = document.getElementById('overlay');
    if (!ov || ov.querySelector('.skz-lb-overlay-btn')) return;
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'btn skz-lb-overlay-btn';
    b.textContent = '🏆 ' + t('view');
    b.addEventListener('click', function (e) {
      e.preventDefault();
      e.stopPropagation();   // don't bubble into the game's own overlay/board handlers
      openModal();
    });
    ov.appendChild(b);
  }
  function relabelOverlayBtn() {
    var ob = document.querySelector('.skz-lb-overlay-btn');
    if (ob) ob.textContent = '🏆 ' + t('view');
  }

  // ---------- DOM: toast ----------
  var toastEl, toastTimer;
  function toast(msg) {
    if (!toastEl) {
      toastEl = document.createElement('div');
      toastEl.className = 'skz-lb-toast';
      document.body.appendChild(toastEl);
    }
    toastEl.textContent = msg;
    toastEl.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { toastEl.classList.remove('show'); }, 2200);
  }

  // ---------- leaderboard view (shared by the desktop dock + mobile modal) ----------
  // Each view owns its DOM (id row + tabs + list) and its own current tab, so
  // the always-on dock and the pop-up modal can coexist without fighting over
  // shared state. `isVisible()` lets refreshViews() skip off-screen instances.
  var views = [];
  function createView(isVisible) {
    var v = { root: null, idEl: null, listEl: null, tabEls: {}, game: VIEW_GAMES[0],
              isVisible: isVisible || function () { return true; } };
    v.root = document.createElement('div');
    v.root.className = 'skz-lb-view';

    v.idEl = document.createElement('div');
    v.idEl.className = 'skz-lb-id';

    var tabs = document.createElement('div');
    tabs.className = 'skz-lb-tabs';
    [['2048', 'tab_2048'], ['dodge', 'tab_dodge']].forEach(function (pair) {
      if (VIEW_GAMES.indexOf(pair[0]) < 0) return;   // pinned page: hide the other game
      var b = document.createElement('button');
      b.className = 'skz-lb-tab'; b.type = 'button';
      b.textContent = t(pair[1]);
      b.setAttribute('aria-selected', pair[0] === v.game ? 'true' : 'false');
      b.addEventListener('click', function () { selectTab(v, pair[0]); });
      v.tabEls[pair[0]] = b;
      tabs.appendChild(b);
    });

    v.listEl = document.createElement('div');
    v.listEl.className = 'skz-lb-list';

    v.root.appendChild(v.idEl);
    // A single-game page needs no tab strip — there's nothing to switch to.
    if (VIEW_GAMES.length > 1) v.root.appendChild(tabs);
    v.root.appendChild(v.listEl);
    renderId(v);
    views.push(v);
    return v;
  }
  function renderId(v) {
    v.idEl.innerHTML = '';
    var name = getName();
    var label = document.createElement('span');
    if (name) {
      label.innerHTML = t('you') + ': <b></b>';
      label.querySelector('b').textContent = name;
    } else {
      label.textContent = t('set_name');
    }
    var link = document.createElement('button');
    link.className = 'skz-lb-link'; link.type = 'button';
    link.textContent = name ? t('change') : t('set_name');
    link.addEventListener('click', openNameDialog);
    v.idEl.appendChild(label);
    v.idEl.appendChild(link);
  }
  function selectTab(v, game) {
    v.game = game;
    GAMES.forEach(function (g) {
      if (v.tabEls[g]) v.tabEls[g].setAttribute('aria-selected', g === game ? 'true' : 'false');
    });
    loadList(v);
  }
  function fmt(n) { try { return Number(n).toLocaleString(); } catch (_) { return String(n); } }
  function loadList(v) {
    v.listEl.innerHTML = '<div class="skz-lb-msg">' + t('loading') + '</div>';
    var game = v.game;
    fetchBoard(game).then(function (data) {
      if (game !== v.game) return;             // a newer tab click won the race
      var myName = getName().toLowerCase();
      // Drop any real row colliding with the reserved champion name so SkyZero
      // can never appear twice.
      var top = ((data && data.top) || []).filter(function (r) {
        return r.name.toLowerCase() !== CHAMP_LOWER;
      });
      v.listEl.innerHTML = '';
      // SkyZero is pinned at #1; real players are renumbered from #2 down.
      v.listEl.appendChild(champRow(game));
      top.forEach(function (r, i) {
        v.listEl.appendChild(row(i + 2, r.name, r.best, r.name.toLowerCase() === myName));
      });
      if (!top.length) {
        var msg = document.createElement('div');
        msg.className = 'skz-lb-msg';
        msg.textContent = t('empty');     // "be the first" now reads as: beat SkyZero
        v.listEl.appendChild(msg);
      } else if (data.me && !top.some(function (r) { return r.name.toLowerCase() === myName; })) {
        // Show the caller's own rank when they fell outside the top window.
        // +1 because the champion occupies a slot above every real player.
        var wrap = document.createElement('div');
        wrap.className = 'skz-lb-me-out';
        wrap.appendChild(row(data.me.rank + 1, data.me.name, data.me.best, true));
        v.listEl.appendChild(wrap);
      }
    }).catch(function () {
      if (game !== v.game) return;
      v.listEl.innerHTML = '<div class="skz-lb-msg">' + t('unavailable') + '</div>';
    });
  }
  function row(rank, name, score, isMe) {
    var medal = rank === 1 ? '🥇' : rank === 2 ? '🥈' : rank === 3 ? '🥉' : null;
    var el = document.createElement('div');
    el.className = 'skz-lb-row' + (isMe ? ' me' : '');
    var rk = document.createElement('div'); rk.className = 'skz-lb-rk'; rk.textContent = medal || rank;
    var nm = document.createElement('div'); nm.className = 'skz-lb-nm'; nm.textContent = name;
    var sc = document.createElement('div'); sc.className = 'skz-lb-sc'; sc.textContent = fmt(score);
    el.appendChild(rk); el.appendChild(nm); el.appendChild(sc);
    return el;
  }
  // The pinned champion, rendered through row() so it shares the grid layout,
  // then given the brand wordmark treatment: neutral "Sky" + rainbow "Zero".
  function champRow(game) {
    var el = row(1, CHAMPION.name, CHAMPION.score[game], false);
    var nm = el.querySelector('.skz-lb-nm');
    if (nm) {
      nm.textContent = '';                  // replace the plain "SkyZero" text
      nm.classList.add('skz-lb-champ-nm');
      var sky = document.createElement('span'); sky.textContent = 'Sky';
      var zero = document.createElement('span'); zero.className = 'skz-lb-zero'; zero.textContent = 'Zero';
      nm.appendChild(sky); nm.appendChild(zero);
    }
    return el;
  }
  // Reload every currently-visible view (after a name change or score upload),
  // keeping each on its own selected tab.
  function refreshViews() {
    views.forEach(function (v) {
      if (v.isVisible()) { renderId(v); loadList(v); }
    });
  }

  // ---------- DOM: desktop dock (always-on left panel, game pages only) ----------
  var dock, dockView;
  function buildDock() {
    // Only on pages with a game overlay; the landing page's content is too wide
    // to clear a left panel, so it keeps the FAB instead.
    if (dock || !document.getElementById('overlay')) return;
    dock = document.createElement('div');
    dock.className = 'skz-lb-dock';
    var head = document.createElement('div');
    head.className = 'skz-lb-head';
    var h = document.createElement('div'); h.className = 'skz-lb-h'; h.textContent = '🏆 ' + t('title');
    head.appendChild(h);
    dockView = createView(function () { return dock && getComputedStyle(dock).display !== 'none'; });
    dock.appendChild(head);
    dock.appendChild(dockView.root);
    document.body.appendChild(dock);
    // Tells the stylesheet it's safe to hide the FAB / overlay button on wide
    // screens (only when a dock actually exists on this page).
    document.documentElement.classList.add('skz-lb-has-dock');
    positionDock();
    window.addEventListener('resize', positionDock);
    selectTab(dockView, dockView.game);   // load now so it's ready when shown
  }
  function positionDock() {
    if (!dock) return;
    var tb = document.querySelector('.topbar');
    // Document-relative top (add scrollY) so the absolutely-positioned dock sits
    // just below the topbar regardless of the scroll offset when this runs.
    // getBoundingClientRect() alone is viewport-relative and, if measured while
    // the page is scrolled (resize, restored scroll position), would bake in a
    // wrong/negative top and leave the dock floating over the brand.
    dock.style.top = (tb ? Math.round(tb.getBoundingClientRect().bottom + window.scrollY) + 8 : 72) + 'px';
  }

  // ---------- DOM: inline panel (mounted by the page, e.g. 2048 below board) ----------
  function buildInline() {
    var mount = document.getElementById('skz-lb-inline');
    if (!mount || mount.firstChild) return;
    var head = document.createElement('div');
    head.className = 'skz-lb-head';
    var h = document.createElement('div'); h.className = 'skz-lb-h'; h.textContent = '🏆 ' + t('title');
    head.appendChild(h);
    var view = createView(function () { return getComputedStyle(mount).display !== 'none'; });
    mount.appendChild(head);
    mount.appendChild(view.root);
    // Lets the stylesheet drop the FAB / overlay button (inline covers narrow,
    // dock covers wide).
    document.documentElement.classList.add('skz-lb-has-inline');
    selectTab(view, view.game);
  }

  // ---------- DOM: mobile modal ----------
  var mask, modalView;
  function buildModal() {
    mask = document.createElement('div');
    mask.className = 'skz-lb-mask';
    mask.hidden = true;
    var panel = document.createElement('div');
    panel.className = 'skz-lb-panel';
    var head = document.createElement('div');
    head.className = 'skz-lb-head';
    var h = document.createElement('div'); h.className = 'skz-lb-h'; h.textContent = '🏆 ' + t('title');
    var x = document.createElement('button');
    x.className = 'skz-lb-x'; x.type = 'button'; x.innerHTML = '×';
    x.setAttribute('aria-label', t('close'));
    x.addEventListener('click', closeModal);
    head.appendChild(h); head.appendChild(x);
    modalView = createView(function () { return mask && !mask.hidden; });
    panel.appendChild(head);
    panel.appendChild(modalView.root);
    mask.appendChild(panel);
    mask.addEventListener('click', function (e) { if (e.target === mask) closeModal(); });
    document.body.appendChild(mask);
  }
  function openModal() {
    if (!mask) buildModal();
    renderId(modalView);
    mask.hidden = false;
    selectTab(modalView, modalView.game);
  }
  function closeModal() { if (mask) mask.hidden = true; }

  // ---------- DOM: name dialog ----------
  var nameMask;
  function openNameDialog() {
    if (!nameMask) buildNameDialog();
    // Re-localize on each open so an in-page language switch is reflected.
    nameMask.querySelector('.skz-lb-name-h').textContent = t('name_title');
    nameMask.querySelector('.skz-lb-hint').textContent = t('name_hint');
    var input = nameMask.querySelector('input');
    input.placeholder = t('name_ph');
    var btns = nameMask.querySelectorAll('.skz-lb-acts .btn');
    if (btns[0]) btns[0].textContent = t('cancel');
    if (btns[1]) btns[1].textContent = t('save');
    var errEl = nameMask.querySelector('.skz-lb-error');
    input.value = getName();
    errEl.textContent = '';
    nameMask.hidden = false;
    input.focus(); input.select();
  }
  function buildNameDialog() {
    nameMask = document.createElement('div');
    nameMask.className = 'skz-lb-mask';
    nameMask.hidden = true;
    var panel = document.createElement('div');
    panel.className = 'skz-lb-panel';
    panel.style.maxWidth = '360px';

    var h = document.createElement('div');
    h.className = 'skz-lb-h skz-lb-name-h'; h.textContent = t('name_title');

    var form = document.createElement('div');
    form.className = 'skz-lb-form';
    var input = document.createElement('input');
    input.type = 'text'; input.maxLength = 20; input.placeholder = t('name_ph');
    var hint = document.createElement('div');
    hint.className = 'skz-lb-hint'; hint.textContent = t('name_hint');
    var errEl = document.createElement('div');
    errEl.className = 'skz-lb-error';
    var acts = document.createElement('div');
    acts.className = 'skz-lb-acts';
    var cancel = document.createElement('button');
    cancel.className = 'btn'; cancel.type = 'button'; cancel.textContent = t('cancel');
    cancel.addEventListener('click', function () { nameMask.hidden = true; });
    var save = document.createElement('button');
    save.className = 'btn primary'; save.type = 'button'; save.textContent = t('save');

    function doSave() {
      var v = input.value.trim();
      if (!v) { return; }
      save.disabled = true; errEl.textContent = '';
      claim(v).then(function (data) {
        setIdentity(data.name, data.token);
        nameMask.hidden = true;
        if (btn) btn.classList.remove('has-pending');
        flush();            // upload anything stashed before the name existed
        refreshViews();     // update id row + lists in the dock / modal
      }).catch(function (e) {
        errEl.textContent = (e && e.status === 409) ? t('taken') : t('err');
      }).then(function () { save.disabled = false; });
    }
    save.addEventListener('click', doSave);
    input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') { e.preventDefault(); doSave(); }
      else if (e.key === 'Escape') { e.preventDefault(); nameMask.hidden = true; }
    });

    acts.appendChild(cancel); acts.appendChild(save);
    form.appendChild(input); form.appendChild(hint); form.appendChild(errEl); form.appendChild(acts);
    panel.appendChild(h); panel.appendChild(form);
    nameMask.appendChild(panel);
    nameMask.addEventListener('click', function (e) { if (e.target === nameMask) nameMask.hidden = true; });
    document.body.appendChild(nameMask);
  }

  // Re-localize every injected label after an in-page language switch.
  function relabelAll() {
    relabelBtn();
    relabelOverlayBtn();
    views.forEach(function (v) {
      if (v.tabEls['2048']) v.tabEls['2048'].textContent = t('tab_2048');
      if (v.tabEls['dodge']) v.tabEls['dodge'].textContent = t('tab_dodge');
      renderId(v);
    });
    // Board panel titles (dock + modal); the name dialog's own title is
    // re-localized when it opens, so exclude it here.
    document.querySelectorAll('.skz-lb-h:not(.skz-lb-name-h)').forEach(function (el) {
      el.textContent = '🏆 ' + t('title');
    });
  }

  // ---------- boot ----------
  function init() {
    injectStyles();
    injectButton();
    injectOverlayButton();
    buildDock();     // game pages only (checks for #overlay); loads immediately
    buildInline();   // pages with a #skz-lb-inline mount (e.g. 2048 mobile)
    // On a pinned game page (2048 / dodge), nudge first-time players to set a
    // nickname right away so their scores can land on the board. Only when one
    // isn't set yet — returning players aren't nagged.
    if (ONLY_GAME && !getName()) openNameDialog();
    // Keep labels in sync if the user toggles language in-page. The page's own
    // lang listener is registered first (inline script runs before this defer
    // script), so localStorage is already updated when these fire.
    document.querySelectorAll('#lang_seg .seg-btn[data-lang]').forEach(function (b) {
      b.addEventListener('click', relabelAll);
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.SkzLB = { submit: submit, open: openModal };
})();
