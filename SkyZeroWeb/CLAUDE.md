# SkyZeroWeb

聚合游戏平台（最初是一个 AI 五子棋页面，后来把多个自训练 AI 都塞了进来）。
所有 AI **全部在浏览器本地运行**，没有后端推理服务器。

## 当前包含的游戏

| 游戏 | 页面 | AI | 相关文件 |
|------|------|----|---------|
| 五子棋 | `/gomoku` | AlphaZero 风格神经网络 + MCTS | `gomoku.html` `main.js` `worker.js` `mcts.js` `gomoku.js` |
| 2048 | `/2048` | Stochastic Gumbel AlphaZero value net + 搜索 | `2048.html` `worker2048.js` `mcts2048.js` `ai2048.js` |
| 躲避 (channel-dodge) | `/channel-dodge` | 暂无 AI（人玩，敌人是手调脚本） | `channel-dodge.html`（单文件自包含） |

躲避游戏的 AI 已经在用 PPO 训练，但**还没放上去**——目前线上的躲避是纯人玩、敌人弹幕为手写行为。不要去找躲避的模型文件。

## 部署（重要）

- 通过 **Cloudflare Pages** 部署，**无构建步骤**：源码即产物，`git push` 自动上线。
- git 仓库根目录是**父文件夹** `/home/sky/RL/SkyZero`；Cloudflare Pages 的「根目录」设为本文件夹 `SkyZeroWeb`，`wrangler.toml` 里 `pages_build_output_dir = "."` 与之对应。
- `functions/api/*` 会被 Pages 自动编译成 Functions（排行榜后端）。
- 缓存策略在 `_headers`：`*.onnx` 永久 immutable，`manifest.json` 300s，其余 1h。

## 关键约定 / 易踩的坑

- **缓存击穿**：因为 `_headers` 给 JS 设了 `max-age=3600`，改了 `gomoku.js`/`mcts.js`/`ai2048.js` 等被 `importScripts` 加载的文件后，必须靠 URL 上的 `?v=<ts>` query 才能让浏览器和 Worker 拿到新版本（见 `worker.js` / `worker2048.js` 顶部、`main.js` 里的 cache-bust 逻辑）。换模型同理要给 `models/2048.onnx` 之类的 URL 加 cache-bust。
- onnxruntime-web 从 **CDN** 加载，强制 **单线程 WASM**（`ort.env.wasm.numThreads = 1`，规避 SharedArrayBuffer 跨域问题）。
- 模型来自**同级训练仓库**（`../SkyZero_V7.1/`、`../SkyZero_2048/`），用 `tools/export_onnx.py`（五子棋）和 `tools/export_onnx_2048.py`（2048）导出到 `models/`，再改 `models/manifest.json`（五子棋 5 档 ELO 目录）。导出步骤与所需 conda 环境见 `README.md`。
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
