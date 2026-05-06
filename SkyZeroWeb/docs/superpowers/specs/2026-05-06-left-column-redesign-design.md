# 左栏重构 (Left Column Redesign) — Design

**Date:** 2026-05-06
**Status:** Awaiting user review

## Problem

左栏目前由 6 张独立卡片堆叠：状态 / 模型 / 我方执子 / 棋盘大小 / 规则 / 搜索。每张卡只装一项设置，视觉碎片化、行高不一致、上下间距不规律；卡间分隔线让设置区看起来像一摞便签而不是一个面板。

棋盘大小放开到 9–19 后还引入了一个**显式 bug**：11 个 segmented 按钮加上左侧 `棋盘大小` 标签塞不进 310px 的 `.side-col`，按钮被挤出右边界。

中栏（棋盘 + 新对局/悔棋）和右栏（胜率 + 6 张热力图）当前布局没有问题，本次重构不动。

## Non-goals

- 不动棋盘画布、悔棋/新对局按钮、棋子绘制逻辑。
- 不动右栏（胜率折线图、WDL 双条、6 张热力图）。
- 不动顶栏（品牌区、分析模式 toggle、语言、主题切换）。
- 不动 i18n 字典结构、`worker.js`、MCTS 管线、模型加载流程。
- 不引入新的 CSS 框架；继续用 `style.css` 内已有的设计 token（`--card-bg`、`--border` 等）。
- 不做移动端专门适配——`.main` 在 ≤1399px 已是单列堆叠，这次重构在桌面宽度下完成即可，移动端自然继承。

## User-facing design

### 整体结构

左栏从 6 张卡缩成 **2 张卡**：

1. **状态条**（顶上一条窄卡，仅承载 `status_pill`）。保留独立卡是因为状态变化频繁（"AI 思考中…" / "黑方获胜!"），独立 + 显眼有助识别。
2. **设置卡**——把模型 / 执子 / 规则 / 大小 / 搜索 全部装进这一张，内部用两个分组小标题切开。

### 设置卡内部布局

每行格式统一为：

```
[ label · 左对齐 ]   [ 控件 · 右对齐固定宽度 ]
```

- 行高一致（约 32–36px）；行与行之间用细分隔线（`1px var(--border)`）而非空白卡间隙。
- 控件右对齐占据右侧约 60% 宽度，保证视觉对齐。

分组用一个小标题（`小写字号 · 灰色 · 字距加宽`）：

```
对局
├── 模型     [LV3 · 进阶 · ELO +400 ▾]   ← <select>
├── 执子     [● 黑棋] [○ 白棋]            ← seg-row
├── 规则     [连珠]   [无禁]               ← seg-row
└── 大小     [—————●————] 15              ← <input type="range"> + 当前值

搜索
├── 开启     [toggle]                       ← <input type="checkbox" class="toggle">
└── 深度     [    32    ]                   ← <input type="number">
```

`深度` 行受现有 `.search-disabled` body class 控制，搜索关闭时整行隐藏（沿用现状）。

### 棋盘大小 widget

用 `<input type="range" min="9" max="19" step="1">` + 右侧当前值文字。理由：

- 9–19 是连续整数序列，slider 在语义上比下拉更贴。
- 11 项下拉点开要扫一遍才能选到目标；slider 一次拖拽即可。
- 占一行高度，跟周围对齐。

slider 旁的当前值文字用等宽字体显示两位数字（`var(--font-mono)`）以避免数字宽度跳动。

### 状态卡

保留为独立窄卡置顶，内部仍是 `status_pill`。视觉上跟设置卡有一道明显间距（设置卡圆角 + 自有边框就够，不必再加分隔线）。

## Architecture

### HTML 改动（`index.html`）

`<aside class="side-col" id="left_col">` 内部从 6 个 `<div class="card">` 改成 2 个：

```html
<aside class="side-col" id="left_col">
  <div class="card status-card">…status_pill 不变…</div>

  <div class="card settings-card">
    <div class="card-body">
      <div class="settings-group">
        <div class="settings-group-title" data-i18n="settings_group_game">对局</div>
        <div class="settings-row">…模型…</div>
        <div class="settings-row">…执子 seg-row…</div>
        <div class="settings-row">…规则 seg-row…</div>
        <div class="settings-row">…大小 slider…</div>
      </div>
      <div class="settings-group">
        <div class="settings-group-title" data-i18n="settings_group_search">搜索</div>
        <div class="settings-row">…搜索开关…</div>
        <div class="settings-row">…深度 input…</div>
      </div>
    </div>
  </div>
</aside>
```

复用：`#model_select`、`#side_black/white`、`#rule_renju/freestyle`、`#search_enable_input`、`#sims_input` 这些 ID 和 `data-rule` / `data-size` 属性**保持不变**——`main.js` 的事件绑定都是按这些 ID/属性 选择器查的，HTML 重排不动它们。

新增类：`.settings-card`、`.settings-group`、`.settings-group-title`、`.settings-row`。
保留类：`.card`、`.card-body`、`.seg-row`、`.seg-btn`、`.toggle`、`.num` 都继续用。

### 棋盘大小控件改造

旧：

```html
<div class="seg-row" id="size_seg" role="group">
  <button class="seg-btn" data-size="9" …>9</button>
  …11 个按钮…
</div>
```

新：

```html
<div class="size-control">
  <input type="range" id="size_input" min="9" max="19" step="1" value="15">
  <span class="size-value" id="size_value">15</span>
</div>
```

`main.js` 改动：

- 删除 `for (const b of document.querySelectorAll(".seg-btn[data-size]"))` 循环（约第 1145 行）。
- 新增：`size_input` 的 `input` 事件 → 调用现有 `setSize(parseInt(ev.target.value, 10))`；`setSize` 内部更新 `#size_value` 文本，不再 toggle `aria-pressed`。

`BOARD_SIZES` 常量保留（`setSize` 仍用 `includes` 校验），但 slider 的 `min/max/step` 已经把范围锁住了，校验主要是防御性的。

### CSS 改动（`style.css`）

新增大约 30–50 行规则：

- `.settings-card .card-body` — 调整 padding，行间分隔线靠 `.settings-row + .settings-row` 处理。
- `.settings-group + .settings-group` — 组之间多一些垂直间距（约 12–16px）。
- `.settings-group-title` — `font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted);`（如果没有 `--muted` 这个 token，复用 `--card-title-hint` 颜色或加一个）。
- `.settings-row` — `display: flex; align-items: center; justify-content: space-between; gap: 12px; min-height: 32px; padding: 6px 0;`；右侧控件 `flex: 0 0 auto` 或固定宽度。
- `.size-control` — flex 容器，slider 占大头，`.size-value` 等宽字体右对齐。
- 不重写 `.side-row`——旧规则在搜索那行（label + toggle）仍然好用；保留它，新规则只用于设置卡内部。

### 状态条

`status-card` 这个类已经存在，沿用即可。可能需要调一下垂直 padding，让它跟设置卡视觉权重平衡（细节实现时定）。

### i18n 改动（`i18n.js`）

新增两个 key（en + zh）：

- `settings_group_game` → "Game" / "对局"
- `settings_group_search` → "Search" / "搜索"

注意：旧的 `label_search` 是搜索行内 label，重构后 `搜索` 这个词会被分组标题占掉。可以让分组标题用 `settings_group_search`，行 label 改成 `label_search_enable`（"开启" / "Enable"），或直接复用 `label_search` 给行用。**决定**：分组标题用新 key，行 label 改成"开启"——避免重复。

### 测试

- 现有 `tests/test_gomoku.mjs`、`tests/test_mcts.mjs` 不动（纯逻辑测试，不涉及 DOM）。
- 手动验证清单（实现阶段执行）：
  - 9–19 全部尺寸切换都能创建新对局
  - 搜索 toggle 关 → 深度行隐藏；toggle 开 → 显示
  - 模型下拉切换、规则切换、执子切换都不退化
  - 分析模式开关 → 右栏 6 张热力图按现状显示/隐藏
  - 状态变化（空闲 → AI 思考中 → 黑方获胜）pill 文本和颜色变体都正确
  - 中文 / 英文切换：分组标题、控件 label 都本地化

## 风险 / 取舍

- **slider 不直观吗？** 11 个值的下拉点开扫一遍其实更慢；slider 拖一下立得。担心拖到错位的话，旁边的当前值文字会让用户立刻感知。
- **状态条独立 vs 合进设置卡**：合进去（顶部一行）也行，但状态颜色变化（绿/红/黄）跟设置卡的中性背景对比强，独立卡视觉更清楚。本设计选独立。
- **样式 token 一致性**：现有 `style.css` 内 `--card-bg` 等变量风格已经统一，新增的 `.settings-*` 类需要复用同一组变量；不要引入临时硬编码色值。
