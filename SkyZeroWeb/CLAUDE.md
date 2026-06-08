# SkyZeroWeb

聚合游戏平台（最初是一个 AI 五子棋页面，后来把多个自训练 AI 都塞了进来）。
所有 AI **全部在浏览器本地运行**，没有后端推理服务器。

## 当前包含的游戏

| 游戏 | 页面 | AI | 相关文件 |
|------|------|----|---------|
| 五子棋 | `/gomoku` | AlphaZero 风格神经网络 + MCTS | `gomoku.html` `main.js` `worker.js` `mcts.js` `gomoku.js` |
| 2048 | `/2048` | Stochastic Gumbel AlphaZero value net + 搜索 | `2048.html` `worker2048.js` `mcts2048.js` `ai2048.js` |
| 躲避 (channel-dodge) | `/channel-dodge` | PPO 策略网络（纯 JS 内联权重，无 ONNX/Worker） | `channel-dodge.html`（单文件自包含）`dodge-policy.js` |

躲避游戏现在有一个 **SAC 训练的自动驾驶 AI**（页面上「AI 托管」按钮，玩家可随时开关）。和五子棋/2048 不同，它的网络极小（连续 SAC actor，MLP `271→256→256→4`，relu；输出前 2 维=移动向量的高斯均值，贪心动作 = `tanh(mean)` 裁剪到单位圆），所以**没有用 onnxruntime-web**：actor 权重打包成 `dodge-policy.js`（base64 little-endian float32，`continuous:true`），在 `channel-dodge.html` 内用 ~30 行**纯 JS 同步前向**跑（无 CDN、无 WASM、无 Worker、无 async），保持页面单文件自包含。271 维观测（最近 32 个威胁的相对位置/速度/伤害/半径 + 3 条激光 + 3 颗炸弹 + 2 个血包 + 玩家血量/无敌/位置/四向墙距）在 JS 里逐字复刻训练环境 `../DodgeSAC/env_dodge.py` 的 `_obs_vector`；决策固定 **30 Hz**。**关键：游戏物理必须跑在固定 450×600 逻辑坐标系**（`SIM_W/SIM_H`，`resize()` 里只缩放渲染，不缩放物理）——否则手机等小屏上 obs 会被 `POS_NORM/VEL_NORM` 裁剪/失真，精确的连续 SAC 直接秒死（旧离散 argmax 策略粗糙反而扛得住，所以这坑曾被掩盖）。敌人弹幕仍是手写行为脚本（AI 只接管玩家移动）。

## 部署（重要）

- 通过 **Cloudflare Pages** 部署，**无构建步骤**：源码即产物，`git push` 自动上线。
- git 仓库根目录是**父文件夹** `/home/sky/RL/SkyZero`；Cloudflare Pages 的「根目录」设为本文件夹 `SkyZeroWeb`，`wrangler.toml` 里 `pages_build_output_dir = "."` 与之对应。
- `functions/api/*` 会被 Pages 自动编译成 Functions（排行榜后端）。
- 缓存策略在 `_headers`：`*.onnx` 永久 immutable，`manifest.json` 300s，其余 1h。

## 关键约定 / 易踩的坑

- **缓存击穿**：因为 `_headers` 给 JS 设了 `max-age=3600`，改了 `gomoku.js`/`mcts.js`/`ai2048.js` 等被 `importScripts` 加载的文件后，必须靠 URL 上的 `?v=<ts>` query 才能让浏览器和 Worker 拿到新版本（见 `worker.js` / `worker2048.js` 顶部、`main.js` 里的 cache-bust 逻辑）。换模型同理要给 `models/2048.onnx` 之类的 URL 加 cache-bust。
- onnxruntime-web 从 **CDN** 加载，强制 **单线程 WASM**（`ort.env.wasm.numThreads = 1`，规避 SharedArrayBuffer 跨域问题）。
- 模型来自**同级训练仓库**（`../SkyZero_V7.1/`、`../SkyZero_2048/`），用 `tools/export_onnx.py`（五子棋）和 `tools/export_onnx_2048.py`（2048）导出到 `models/`，再改 `models/manifest.json`（五子棋 5 档 ELO 目录）。导出步骤与所需 conda 环境见 `README.md`。
- **躲避走另一条路（不是 ONNX）**：模型来自 `../DodgeSAC`（同在 SkyZero 仓库内的 SAC 训练项目），用 `tools/export_dodge_sac.py --ckpt ../DodgeSAC/runs/<name>/best.pt --version <tag>` 把 SAC actor 权重导出成根目录的 `dodge-policy.js`（导出时带 torch-vs-打包自检，布局错了会直接报错）。重训后**重新导出**，并把 `channel-dodge.html` 里 `<script src="dodge-policy.js?v=…">` 的 `?v=` 改成新 `<tag>` 做 cache-bust（脚本会打印要粘贴的那行）。Python↔JS 一致性除导出自检外，还可用 `node` 无头跑整页游戏逻辑复核（曾用此定位手机坐标系 bug）。
- 浏览器版相对训练版做了简化（无 8 重对称集成、无并行 MCTS 等），细节见 `README.md` 末尾「Differences from V5」。

## 排行榜（Cloudflare D1）

仅昵称、无登录的「荣誉系统」榜单，覆盖 2048 与躲避（五子棋无连续分数，故排除）。
后端在 `functions/api/`，库表见 `schema.sql`，前端是 `leaderboard.js`（`window.SkzLB.submit(game, score)`）。
完整搭建/重置说明见 `LEADERBOARD_SETUP.md`。

## 本地开发与测试

```bash
# 纯静态预览（不含排行榜 API；file:// 不行，Worker 的 importScripts/fetch 需要 HTTP）
python3 -m http.server 8000

# 含 Functions + 本地 D1 的完整栈
npx wrangler pages dev .

# 单元测试（纯逻辑：gomoku / mcts / ai2048 / mcts2048；UI 手测）
export PATH=/home/sky/.nvm/versions/node/v24.15.0/bin:$PATH
npm test
```

更详细的导出模型 / 排行榜运维命令，分别见 `README.md` 与 `LEADERBOARD_SETUP.md`。
