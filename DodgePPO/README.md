# PPO on Channel-Dodge (bullet-hell survival)

用 **PPO**(Proximal Policy Optimization)训练 AI 玩 `SkyZero/SkyZeroWeb/channel-dodge.html` 的弹幕躲避游戏。和 `../DQN`(QR-DQN)是同一个游戏、同一套移植逻辑,但换成 **on-policy + 多并行环境**的 PPO——更稳、更适合这种实时长时生存任务,也是冲高分的主力。

## 为什么用 PPO(对比 DQN 那版)

QR-DQN 第一版能学(超随机、追平手写 flee 脚本),但 eval 得分在峰值后**剧烈震荡不收敛**(value-based + PER/n-step 的典型不稳)。PPO 针对性地改了三点:

1. **on-policy + clip 目标**:更新更稳,不容易像 Q-learning 那样高估发散。
2. **多并行环境**:躲避环境是纯 numpy、极快,PPO 一次 rollout 同时跑 N 个环境,吞吐高。
3. **更精细的感知 + 更长回合**(见下),直接对准之前定位到的瓶颈。

> 仍然**不用 AlphaZero**:实时、连续控制、单智能体、强随机的生存任务没有对手、无法对实时物理做 MCTS——model-free 才对路。

## 文件结构

```
PPO/
├── env_dodge.py      # ChannelDodgeEnv + SyncVectorDodgeEnv（游戏逻辑移植 + 向量化）
├── networks.py       # ActorCritic：MLP(向量obs)/CNN(栅格obs)，离散/连续策略头
├── config.py         # .cfg 解析 + Config
├── configs/dodge.cfg # PPO 超参
├── train_ppo.py      # PPO 主循环：rollout + GAE + clip 更新，定时 eval，CSV + 仪表盘
├── evaluate.py       # 加载 checkpoint 贪婪对局 / 观战
├── requirements.txt
└── checkpoints/      # dodge_best.pt / dodge_final.pt / *_eval.csv / *_progress.png
```

## 环境 `ChannelDodgeEnv`

- **观测**(`OBS_MODE`):
  - `vector`(默认):最近 K=10 个威胁的**相对位置/速度/伤害** + 血包 + 玩家状态(HP、到四壁距离)→ 定长 `(74,)` 向量,喂 MLP。比 DQN 那版 24px 的粗栅格**精度高得多**——之前 agent 卡在 ~20 秒,粗感知是主因。
  - `grid`:`(6,13,13)` 自我中心危险场图,喂 CNN(和 DQN 版对齐做对照)。
- **动作**(`ACTION_MODE`):
  - `discrete`(默认):`Discrete(9)`,不动 + 8 方向。
  - `continuous`:`Box(-1,1,(2,))` 原生 2D 移动向量(env 内裁剪到单位圆),高斯策略,steering 最细腻。
- **奖励**:`+存活 + 躲弹得分 − 受伤 + 拾血包 − 死亡`(每步,密集且对齐真实分数)。
- **回合**:HP=0 → terminated;`MAX_STEPS`(默认 4000≈133s)→ truncated(PPO 做了正确的**截断 bootstrap**:用真实终态价值续接,而非当作硬终止)。

## 训练

```bash
python train_ppo.py configs/dodge.cfg              # 标准配方(默认 30M 步,到 ~1300 均值)
python train_ppo.py configs/dodge.cfg --set num_workers=12   # 空核多就多给点
python train_ppo.py configs/dodge.cfg --print-config
```

## 改游戏配置后的重训流程

每次改了游戏参数(子弹速度、伤害、spawn、难度曲线…)都要**从零重训**(配置变了,旧策略失效,续训只会负迁移、不稳):

1. 改 `../SkyZeroWeb/channel-dodge.html`;
2. 把改动的玩法常量同步进 `env_dodge.py`(顶部 `TYPES / PLAYER_SPEED / 各 spawn 函数 / difficulty()` 等是 HTML 的忠实移植);
3. `./retrain.sh <名字>` —— 跑标准配方,完事自动做 50 局稳健评估。

## 并行加速(多进程向量环境)

瓶颈是**纯 Python 的环境步进**(网络才 8.7 万参数,GPU 基本闲着),所以用多进程把环境铺到多核:

- `NUM_WORKERS>1` 启用 `SubprocVectorDodgeEnv`,把 `NUM_ENVS` 个环境分到 N 个进程并行步进。
- **关键:保持 `BATCH = NUM_ENVS × NUM_STEPS ≈ 4096` 不变**(加 env 就按比例减 num_steps),这样梯度更新次数/样本效率和原始 16×256 一致,纯赚吞吐:

  | 配置(batch 均 4096) | 训练吞吐 | 加速 |
  |---|---|---|
  | 16env×256step / 1 worker(原) | ~6k sps | 1× |
  | **64env×64step / 8 worker(默认)** | ~15k sps | **~2.6×** |
  | 128env×32step / 12 worker | ~22k sps | ~3.7× |

  > 反例:只加 env 不减 num_steps(batch 变大)→ 更新次数变少 → **学得更慢**,得不偿失。

  实测:从零到 ~1300 均值,原配方 ~1h(独占)/~2h(陪 2048);默认并行配置约 **20–40 分钟**。GPU 占用 <1GB,和 2048 训练共存无冲突。

每 `EVAL_EVERY` 次更新贪婪评估一次,打印 in-game 得分 / 存活时长 / return,并把指标写入
`checkpoints/dodge_{train,eval}.csv`、仪表盘 `dodge_progress.png`(得分 / 存活 / return / 熵+KL)。
最佳模型存 `dodge_best.pt`,滚动存 `dodge_final.pt`。

**基线参考**(eval 同种子 20 局):随机 score≈9 / 存活≈13s;手写 flee score≈22 / 存活≈18s;
人类:作者 ~800,朋友 ~1500(人类能撑数分钟进入封顶弹幕)。

## 评估 / 观战

```bash
python evaluate.py checkpoints/dodge_best.pt --episodes 50
python evaluate.py checkpoints/dodge_best.pt --episodes 1 --render --sleep 0.03
```

## 部署回浏览器(后续)

训练好后把策略网导出 ONNX → 在 `channel-dodge.html` 加 “AI 托管” 模式,用 onnxruntime-web 推理驱动
`moveVector()`,即可在线上看 AI 真的在玩——和网站里五子棋 / 2048 的模型一个套路。
