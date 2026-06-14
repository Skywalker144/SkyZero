# SkyZero_2048_V2 — Project Guide for Claude

**Stochastic Gumbel AlphaZero for 2048**, re-forked from the mainline SkyZero
framework **V7.8** (2026-06-14) and re-specialized for 2048. 2048 is a
single-agent *stochastic* MDP, so the two-player AlphaZero machinery
(deterministic transitions, WDL value, perspective flip, Elo) is replaced by an
**afterstate** formulation with **chance nodes** and a **scalar
expected-discounted-score** value.

> ⚠️ **MID-MIGRATION.** This tree is a work-in-progress re-fork. The Python /
> scripts / configs are the V7.8 base being adapted to 2048; the C++ is the 2048
> afterstate core being reorganized to V7.8's layout. **`docs/refactor_plan.md`
> is the source of truth** for what's done vs pending — read it before touching
> anything. The old, fully-working V7.1-vintage implementation is at
> `../SkyZero_2048/` (parity reference); the mainline base is `../SkyZero_V7.8/`.

---

## 不变量(2048 特有,改任何东西前对照)

- **Single-agent**:无 `to_play`、无翻转。Backup 是 `G = r + γ·G_child`。
- **Stochastic afterstate**:DECISION(选 4 动作)+ CHANCE(环境 spawn)双层节点;
  spawn 概率已知(90% tile-2 / 10% tile-4,空格均匀)→ 枚举,不学 dynamics。
- **D4 augment 必须重标号 4 个动作**(`augment.py`)。纯转棋盘平面是错的。
- **标量 value**:value head softplus≥0;返回值达数万 → MCTS 用 MuZero min-max norm 做 PUCT。
- **value 变换硬编码开**(单轨,`data2048`):value head **恒**训 `h(raw)/VALUE_SCALE`,h 是 MuZero
  可逆缩放 `h(x)=sign(x)(√(|x|+1)−1)+εx`(ε=1e-3,`value_transform.py` ↔ `infer_server_2048.h::inv_value_h`),
  search 用 h⁻¹ 解回 raw points。`VALUE_SCALE=30`(h 空间)。VALUE_TRANSFORM 开关 + 线性轨已删。
- **TD_STEPS**:value target = n-step TD bootstrap on MCTS 根值。
- **npz**:`state(N,16) / policy(N,4) / value(N,1)`;infer server 吃 `encode_state`(NUM_PLANES×16),
  绝不是 raw 16-cell board。
- Torch 在 `pytorch` conda 环境(`scripts/env_paths.cfg.local` → `PY`)。

---

## 框架机制(继承自 V7.8,见 ../SkyZero_V7.8/CLAUDE.md 详述)

- 三层配置:`configs/<exp>/{run,paths,play}.cfg` + `*.cfg.local` + `scripts/env_paths.cfg`;
  `CONFIG_DIR=configs/<exp> bash scripts/run.sh`。`run.cfg` 平赋值(C++ `parse_cfg()` 也读)。
- 一个 iter:schedule active 网络 → 镜像 `models/latest.pt` → gate 化 selfplay 产量
  (`compute_selfplay_target.py`,替代旧 token bucket)→ selfplay(C++)→ shuffle(power-law 窗口)
  → 每个 NETWORK train→export → probe → view_loss。
- 多网络共训(只 active 跑 selfplay);多卡:主卡跑循环,副卡 daemon 持续 selfplay 热重载 `models/latest.pt`。
- resume 取每 net `max(iter)` + catch-up。
- **已对齐 V7.8 的增量(2026-06-14,默认值见 run.cfg)**:
  - 训练增强 `train.py`:AMP(`ENABLE_AMP`)、SWA(`ENABLE_SWA`,导出取 SWA 权重)、in-loop Lookahead
    (`LOOKAHEAD_K`)、KataGomo 5 段 LR-warmup(`ENABLE_LR_WARMUP`);AdamW。
  - nbt 网络 `full_nets.Net2048NBT`(fixscale+RepVGG,标量头):`FULL_NET=1` 开,默认仍 `Net2048`(BN);
    `build_net` 分发,train/export/init/evaluate 统一调 `initialize()`/`set_norm_scales()`(BN 版 no-op)。
  - 推理期随机 D4 对称(`ENABLE_STOCHASTIC_TRANSFORM_ROOT/_CHILD`):**动作重标号**(`game2048::ACTION_PERM`,
    经 apply_move 等变性单测验证),全程在 MCTS 类内(root_encoded/select_leaf + undo)。
  - P3 小旋钮:chosen-move 温度 early/halflife 插值 + root policy temperature(均 root=puct,`interpolate_early`)
    + 完整 KataGo radius-factor LCB(eval-only)。

**2048 与主线的差异**:scalar value 代替 WDL;`evaluate.py`(avg score + tile reach)代替 `elo.py`;
afterstate/chance C++ 搜索;`value_transform` / `augment` 动作重标号 / TD 目标是 2048 专属。

---

## 入口

| 命令 | 作用 |
|---|---|
| `bash scripts/build.sh [--target X]` | cmake 编 C++ |
| `CONFIG_DIR=configs/baseline bash scripts/run.sh [max_iters]` | 主训练循环(同步 gate:每 iter 有界 selfplay→train) |
| `CONFIG_DIR=configs/baseline bash scripts/faster_run.sh [max_iters]` | **异步循环**:`selfplay2048_par --daemon` 常驻生产(消 2048 长尾)+ 消费者攒够新行才 train;节奏复用 `TARGET_REPLAY_RATIO`。单卡 daemon+trainer 共享一张卡 |
| `CONFIG_DIR=configs/baseline bash scripts/play_web.sh` | web demo |

C++ 单元测试:
```
cd cpp && g++ -std=c++17 -I . envs/game2048_test.cpp -o /tmp/t && /tmp/t
        g++ -std=c++17 -I . skyzero_2048_test.cpp  -o /tmp/m && /tmp/m
```

---

## 项目特有准则

1. **改 cfg = 改 `.local`**(机器路径 / 一次性实验参数不进 git 跟踪的 cfg)。
2. 改 C++ 或 cfg 后先 `bash scripts/build.sh` 单独编一次。
3. 改 `python/{full_nets,nets,model_config,train,shuffle,...}.py` 前想清楚对已有 `DATA_DIR` 的兼容性。
4. **重构期间**:每完成一项更新 `docs/refactor_plan.md`;不要在 V2 上跑会和 `../SkyZero_V7.8`
   抢 GPU 的任务(单卡 5090,实验期勿占卡)。
