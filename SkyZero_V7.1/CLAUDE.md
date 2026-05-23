# SkyZero — Project Guide for Claude

C++/Python AlphaZero-style 训练框架,棋类:Gomoku / Renju(连珠),最大棋盘 19×19。
C++(LibTorch)负责 selfplay + MCTS,Python(PyTorch)负责训练 / shuffle / Elo / 模型导出,
bash 做编排。环境/安装见 `SETUP.md`(本文不重复)。

---

## 一个 iter 在做什么(`scripts/run.sh` 主循环)

1. `schedule.py active` → 算出当前 **active 网络**(累计 selfplay 样本跨过 `SELFPLAY_SCHEDULE` 的阈值)
2. 把 `data/nets/<active>/latest.pt` 镜像到 `data/models/latest.pt`(C++ selfplay 永远只读这个稳定路径)
3. `selfplay.sh`(主 GPU)+ 多 GPU 时 `selfplay_daemon.sh`(副卡并行)→ npz 落到 `data/selfplay/`
4. `shuffle.sh` → KataGo 三参数 power-law 窗口采样 → `data/shuffled/current/`
5. `bucket.py` 决定 `train_steps`(token bucket;不足一个 epoch 时**跳过训练**,selfplay 继续累积)
6. 对 `NETWORKS` 列的**每个**网络:`train.sh`(共享同一份 shuffled 数据,串行)→ `export.sh`(TorchScript)
7. 再次镜像 active 网络的新权重 → `mcts_probe`(空棋盘 rootValue 探针)→ `view_loss.py` 画曲线

---

## 关键架构概念

- **多网络协同训练**:`NETWORKS="b5c128, b10c256, b15c384"` 同 iter 一起训,只有 active 那个跑 selfplay。
  网络命名 `b<blocks>c<channels>`,其余拓扑参数由 `python/model_config.py` 从 blocks/channels 自动派生(KataGo v15)。
- **Token bucket**(`python/bucket.py`):`bucket += new_rows * MAX_TRAIN_PER_DATA`(封顶 `MAX_TRAIN_BUCKET_SIZE`);
  `bucket >= TRAIN_SAMPLES_PER_EPOCH` 才训。replay 比例随数据生成速率自适应。
- **Selfplay 窗口**(`python/shuffle.py`):`W(N) = (N^E - M^E) / (E * M^(E-1)) * IWPR + M`,
  即 `MIN_ROWS(M) / TAPER_WINDOW_EXPONENT(E) / EXPAND_WINDOW_PER_ROW(IWPR)`。`N <= M` 时 shuffle 跳过。
- **多 GPU 模型**:run.sh 探测 GPU 数 → 主循环占 `MAIN_GPU=0`,副卡 `1..N-1` 由 `selfplay_daemon.sh` 持续 selfplay,
  **热重载** `data/models/latest.pt`(active 换了之后 daemon 下一局自动用新模型,无需重启)。
  所有 train/probe/export 都设 `CUDA_VISIBLE_DEVICES=$MAIN_GPU`,否则会撞 daemon。
- **续跑(resume)**:run.sh 读每个 net 的 `state.json`,取 `max(iter)` 作为下一 iter;
  落后的 net 在已有 `shuffled/current/` 上做一次 **catch-up 训练**后才进主循环(bucket/selfplay/shuffle 不能重算)。

---

## 配置体系(三层覆盖)

| 文件 | 内容 | git 跟踪 |
|---|---|---|
| `configs/<exp>/{run,paths,play,elo}.cfg` | 实验默认值 | ✓ |
| `configs/<exp>/{run,paths}.cfg.local` | 这台机器在这个实验下的覆盖 | ✗ |
| `scripts/env_paths.cfg` + `.local` | 机器级 `LIBTORCH / NVCC / PY` | 默认 ✓ / `.local` ✗ |

切实验:`CONFIG_DIR=configs/nsim_64 bash scripts/run.sh`(默认 `configs/baseline`)。

**绝不**直接改 git 跟踪的 cfg 来做"本机微调" —— 写 `*.local`。
同名变量优先级:`.local` > 调用前 export 的 env > cfg 默认值。

`run.cfg` 既被 bash source,也被 C++ `parse_cfg()` 读 → 一份配置两边用。
改了 `MAX_BOARD_SIZE` 会触发 cmake 重 configure 并重编 C++(`run.sh` 每次都跑 `cmake --build`,自动处理)。

---

## 入口速查

| 脚本 | 作用 |
|---|---|
| `scripts/build.sh [--target X]` | cmake 编 C++(首次 configure,之后增量) |
| `scripts/run.sh [max_iters]` | 主训练循环,Ctrl+C 安全 |
| `scripts/elo.sh` | Elo 评估,产 jsonl + BT-MLE 拟合 |
| `scripts/play_web.sh` | 人机对弈 web UI |
| `scripts/bench.sh` | MCTS 速度 bench |

---

## 代码地图

**Python**(`python/`):
- 训练流水线:`train.py` / `bucket.py` / `shuffle.py` / `schedule.py` / `warmup.py`
- 模型:`full_nets.py` `nets.py` `model_config.py`(KataGo v15 拓扑)
- 工具:`export_model.py`(TorchScript 导出)/ `init_model.py`(随机初始化)
- 评估 & 可视化:`elo.py`(BT-MLE)/ `view_loss.py`(loss 曲线)
- 服务:`play_web.py`(人机 web 对弈,~2k 行)

**C++**(`cpp/`):
- selfplay 主程序:`selfplay_main.cpp` + `selfplay_manager.h`
- MCTS 三种实现:`skyzero.h`(单线程)/ `skyzero_parallel.h`(leaf-parallel)/ `skyzero_tree_parallel.h`(tree-parallel)
- 其他入口:`gomoku_{play,elo,ab}_main.cpp` / `mcts_{probe,bench}_main.cpp`
- 辅助 header:`random_opening.h` `policy_init.h` `policy_surprise_weighting.h` `npz_writer.h` `game_initializer.h`

---

## 已知坑 & 不变量

- **C++ 编译用的 LibTorch 必须是 PyTorch wheel 里的那份**(`site-packages/torch/`),
  不要单独下 LibTorch tarball — ABI / CUDA runtime 会撞。`env_paths.cfg` 默认就指向 conda 的 torch。
- **`data/models/latest.pt` 是镜像,不是真源** — 真源在 `data/nets/<net>/`。
  run.sh 在切 active 和 export 后都会重镜像;**改训练流程时永远不要直接写这个路径**。
- **多网络续跑取 `max(iter)`,不是 `FIRST_NET` 的 iter**(已修复 2026-05-23):
  否则 Ctrl+C 在 iter N 中段后,落后的 net 会在 `train.tsv` 留 gap,那一份 bucket 消耗也白费。
  Catch-up 块在主循环前先训落后的 net,之后才正常推进。
- **`MAIN_RULE=renju` 必须配 `NUM_PLANES=5`**(平面含 `forbidden_black/white`);
  切 `standard / freestyle` 要同步改 `NUM_PLANES`,否则 C++ / Python 两边的张量形状会对不上。
- **`MAX_BOARD_SIZE` 是编译期常量**(cmake `-D` 注入),`MAIN_BOARD_SIZE` 是运行时;
  改前者会触发重编,改后者不会。
- **改了网络拓扑**(`full_nets.py` / `nets.py` / `model_config.py`):
  已有 checkpoint 大概率加载失败 → 换 `DATA_DIR` 或显式提示用户跑 `init_model.py` 重 init。
- **训练状态文件有"消耗一次"语义**:`bucket.json` / `state.json` / `schedule.tsv` /
  `selfplay.tsv` / `train.tsv` 必须保持相对一致 — 改 resume 逻辑前先把这五个的关系读懂。

---

## 项目特有的行为准则

通用编码守则(simplicity / surgical changes / 不引入未要求的抽象 / 不加多余错误处理)
已在 Claude 的系统提示里,这里只补**项目特有**的:

1. **改 cfg = 改 `.local`**。机器路径、一次性实验参数都不应进 git 跟踪的 cfg。
2. **改 C++ 或 cfg 后**,先 `bash scripts/build.sh` 单独编一次 — `run.sh` 自己也会编,
   但单跑能更早暴露编译错误,避免 selfplay 启动失败才发现。
3. **跑 `run.sh` 前**,如果改了任何 `python/{full_nets,nets,model_config,bucket,shuffle,schedule,train}.py`,
   想清楚是否会影响已有 `DATA_DIR` 的兼容性 —— 不兼容就明确告诉用户。
4. **环境/依赖问题**(CUDA / LibTorch / nvcc / libzip / pip 装的 PyTorch CPU vs CUDA)→ 直接转 `SETUP.md §9 常见坑`,
   不要在对话里自己重新推理一遍。
