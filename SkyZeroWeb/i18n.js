// Lightweight i18n. Default language is Chinese; toggle in the topbar.
// Persisted via localStorage key 'skz_lang'. Public API: getLang, setLang, t,
// applyI18n, registerI18nCallback.

const I18N_STRINGS = {
    en: {
        page_title: "SkyZero Gomoku · ONNX Web",
        brand_title: "SkyZero Gomoku",
        brand_sub: "AlphaZero-style self-play · ONNX · static web",
        aria_theme: "Toggle color theme",
        aria_lang: "Switch language",
        status_idle: "idle",
        status_your_turn: "Your turn",
        status_ai_thinking: "AI thinking…",
        status_black_wins: "Black wins!",
        status_white_wins: "White wins!",
        status_draw: "Draw!",
        label_model: "Model",
        label_human_side: "Human side",
        side_black: "Black",
        side_white: "White",
        label_board_size: "Board size",
        label_rule: "Rule",
        rule_renju: "Renju",
        rule_standard: "Standard",
        rule_freestyle: "Freestyle",
        label_search: "Search",
        label_sims: "sims",
        label_value_estimates: "Value estimates",
        legend_root: "root",
        legend_nn: "nn",
        legend_axis: "W/L · −1…+1",
        wdl_dash: "—",
        btn_new_game: "New game",
        btn_undo: "Undo",
        heat_improved_policy: "Improved Policy",
        heat_visits_dist: "Visits Dist",
        heat_nn_policy: "NN Policy",
        heat_nn_opp_policy: "NN Opp Policy",
        heat_nn_futurepos_8: "NN Futurepos +8",
        heat_nn_futurepos_32: "NN Futurepos +32",
        heatmap_default_title: "Heatmap",
        aria_expand: "Expand",
        aria_close: "Close",
        chart_no_data: "no data",
        loading_initial: "Loading model…",
        loading_manifest: "Loading manifest…",
        loading_model: (label) => `Loading ${label}…`,
        err_manifest_load: (msg) => `manifest load failed: ${msg}`,
        err_manifest_empty: "manifest empty — add models",
        err_worker_failed: (msg, where) => `Worker failed: ${msg}${where}`,
        err_unknown: "unknown error",
        err_worker_msg: "Worker message error",
        err_prefix: (msg) => `Error: ${msg}`,
        theme_label_auto: "Auto",
        theme_label_light: "Light",
        theme_label_dark: "Dark",
    },
    zh: {
        page_title: "SkyZero 五子棋 · ONNX Web",
        brand_title: "SkyZero 五子棋",
        brand_sub: "AlphaZero 风格自对弈 · ONNX · 静态网页",
        aria_theme: "切换配色主题",
        aria_lang: "切换语言",
        status_idle: "空闲",
        status_your_turn: "轮到您",
        status_ai_thinking: "AI 思考中…",
        status_black_wins: "黑方获胜!",
        status_white_wins: "白方获胜!",
        status_draw: "和棋!",
        label_model: "模型",
        label_human_side: "我方执子",
        side_black: "黑棋",
        side_white: "白棋",
        label_board_size: "棋盘大小",
        label_rule: "规则",
        rule_renju: "连珠",
        rule_standard: "标准",
        rule_freestyle: "无禁",
        label_search: "搜索",
        label_sims: "模拟数",
        label_value_estimates: "胜率估计",
        legend_root: "根",
        legend_nn: "网络",
        legend_axis: "胜负 · −1…+1",
        wdl_dash: "—",
        btn_new_game: "新对局",
        btn_undo: "悔棋",
        heat_improved_policy: "改进策略",
        heat_visits_dist: "访问分布",
        heat_nn_policy: "网络策略",
        heat_nn_opp_policy: "对手策略",
        heat_nn_futurepos_8: "未来落点 +8",
        heat_nn_futurepos_32: "未来落点 +32",
        heatmap_default_title: "热力图",
        aria_expand: "展开",
        aria_close: "关闭",
        chart_no_data: "暂无数据",
        loading_initial: "正在加载模型…",
        loading_manifest: "正在加载清单…",
        loading_model: (label) => `正在加载 ${label}…`,
        err_manifest_load: (msg) => `清单加载失败:${msg}`,
        err_manifest_empty: "清单为空 — 请添加模型",
        err_worker_failed: (msg, where) => `Worker 失败:${msg}${where}`,
        err_unknown: "未知错误",
        err_worker_msg: "Worker 消息错误",
        err_prefix: (msg) => `错误:${msg}`,
        theme_label_auto: "自动",
        theme_label_light: "明亮",
        theme_label_dark: "暗黑",
    },
};

const I18N_DEFAULT = "zh";
const I18N_LANGS = ["zh", "en"];

function getLang() {
    try {
        const v = localStorage.getItem("skz_lang");
        if (v && I18N_STRINGS[v]) return v;
    } catch (_) {}
    return I18N_DEFAULT;
}

function setLang(lang) {
    if (!I18N_STRINGS[lang]) return;
    try { localStorage.setItem("skz_lang", lang); } catch (_) {}
    document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
    applyI18n();
    for (const cb of i18nCallbacks) {
        try { cb(); } catch (_) {}
    }
}

function t(key, ...args) {
    const dict = I18N_STRINGS[getLang()] || I18N_STRINGS[I18N_DEFAULT];
    const v = dict[key];
    if (v == null) return key;
    return typeof v === "function" ? v(...args) : v;
}

const i18nCallbacks = [];
function registerI18nCallback(fn) { i18nCallbacks.push(fn); }

function applyI18n() {
    document.title = t("page_title");
    for (const el of document.querySelectorAll("[data-i18n]")) {
        el.textContent = t(el.dataset.i18n);
    }
    for (const el of document.querySelectorAll("[data-i18n-title]")) {
        el.title = t(el.dataset.i18nTitle);
    }
    for (const el of document.querySelectorAll("[data-i18n-aria]")) {
        el.setAttribute("aria-label", t(el.dataset.i18nAria));
    }
    for (const el of document.querySelectorAll("[data-i18n-aria-group]")) {
        el.setAttribute("aria-label", t(el.dataset.i18nAriaGroup));
    }
    updateLangSegPressed();
}

function updateLangSegPressed() {
    const cur = getLang();
    const seg = document.getElementById("lang_seg");
    if (!seg) return;
    for (const b of seg.querySelectorAll(".seg-btn[data-lang]")) {
        b.setAttribute("aria-pressed", b.dataset.lang === cur ? "true" : "false");
    }
}

function initLangSeg() {
    const seg = document.getElementById("lang_seg");
    if (!seg) return;
    for (const b of seg.querySelectorAll(".seg-btn[data-lang]")) {
        b.addEventListener("click", () => {
            if (b.dataset.lang !== getLang()) setLang(b.dataset.lang);
        });
    }
}

document.addEventListener("DOMContentLoaded", () => {
    document.documentElement.lang = getLang() === "zh" ? "zh-CN" : "en";
    applyI18n();
    initLangSeg();
    for (const cb of i18nCallbacks) {
        try { cb(); } catch (_) {}
    }
});
