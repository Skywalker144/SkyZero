# DodgeSAC

SAC（Soft Actor-Critic）自动驾驶 agent，玩弹幕躲避小游戏 **Channel-Dodge**
（与 `../SkyZeroWeb/channel-dodge.html` 同一套规则）。连续动作（2D 移动向量），
271 维观测，纯前馈 MLP `271→256→256`。线上版就是从这里导出的（见「部署」）。

> 历史：本目录前身是 `DodgePPO`（离散 PPO）。对比过 PPO / Rainbow / QR-DQN / TD3 后，
> 连续 **SAC 样本效率与最终效果都最好**，故定为唯一算法，其余已移除。

## 环境 `env_dodge.py` / `env_dodge_gpu.py`

- `ChannelDodgeEnv`：CPU 单环境（gym 风格），`obs_mode="vector"`（271 维）、
  `action_mode="continuous"`（`Box(-1,1,(2,))`，裁剪到单位圆）。**固定 450×600 竞技场**
  —— 浏览器端必须在同一逻辑坐标系跑物理，否则连续策略在不同屏幕上会因 obs 裁剪/失真而失效。
- `env_dodge_gpu.py` `VecDodgeGPU`：全张量、全程驻留 GPU 的批量并行环境（IsaacGym 风格），
  随机策略与 CPU 版对齐（score/surv 误差 <10%）。SAC 训练用这个，吞吐 ~数十万 sps。
- 奖励整形旋钮：`stationary_bonus` / `reverse_penalty` / `accel_penalty`(速度一阶导，控"顺")
  / `jerk_penalty`(二阶导，**会害躲命，慎用**) / `speed_penalty`(控"省力，安全时不乱动")
  / `center_weight`(靠中心)。

## 训练

```bash
conda activate pytorch
# 从头训（标准）
python train_sac_gpu.py --run-name sac_gpu --num-envs 512 --total-steps 12000000 \
    --accel-penalty 0.04 --speed-penalty 0.012 --center-weight 0.02
# 从已有 checkpoint 续训（恢复 actor/critic/alpha，建议降 lr）
python train_sac_gpu.py --resume runs/sac_gpu_smooth/final.pt --lr 1e-4 ...
```
`train_sac.py` 是 CPU 版（用 `config_rl.py` 的 `.cfg`），保留作参考；主力是 `train_sac_gpu.py`。

## 评估 / 录制

- `evaluate.py runs/<name>/best.pt --episodes 50` —— 贪心评分。
- `eval_behavior.py runs/A/best.pt runs/B/best.pt` —— **四轴对比**：分数、顺(jitter)、
  省力(安全 vs 危险时速度)、居中(到中心像素距离)。调系数主要看这个。
- `record_play.py` —— 导出一局回放给 `viewer.html` 本地看走位。
- `analyze_energy.py` —— 安全/危险时速度分布（"省力"指标）。

## 部署到网页

线上不跑后端推理：把 SAC actor 权重打包成 `../SkyZeroWeb/dodge-policy.js`（base64 float32），
`channel-dodge.html` 内纯 JS 同步前向。重训后：

```bash
cd ../SkyZeroWeb
python tools/export_dodge_sac.py --ckpt ../DodgeSAC/runs/<name>/best.pt --version <tag>
# 按脚本提示把 channel-dodge.html 的 <script src="dodge-policy.js?v=…"> 改成新 <tag> 做 cache-bust
```

## 已训模型（`runs/`，gitignore）

| run | 说明 |
|---|---|
| `sac_gpu_smooth` | 现网基线（eval ~7475；accel0.05+center0.02，顺/居中但不够省力） |
| `sac_t2a` | 当前最佳行为（7258；jitter 0.30 / safe_v 0.44 / cdist 63） |
| `sac_t2b` | 最省力（ratio 0.73） |
| `sac_t3cont` | t2a 续训实验（未超过 t2a） |
