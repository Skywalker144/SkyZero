# GumbelAlphaZero

​    **GumbelAlphaZero**(Danihelka et al., ICLR 2022《Policy improvement by planning with Gumbel》)是对 AlphaZero 搜索的一次重新设计。它要解决 AlphaZero 的一个软肋:**当模拟次数很少、根节点没法访问到所有动作时,AlphaZero 不能保证训练目标真的比神经网络本身更强**。GumbelAlphaZero 用「**无放回采样 + Sequential Halving(序贯减半)**」替换了根节点原本那套靠 Dirichlet 噪声和 PUCT 的启发式,从理论上**保证了策略提升**——哪怕只有 2 次模拟。SkyZero_V7.1 用的就是它。

> ​    本文假设你已经读过同目录的 `AlphaZero算法简介及伪代码.md`。GumbelAlphaZero 只改动**根节点**的动作选择与训练目标,**树内部(非根节点)的选择、扩展、反向传播与 AlphaZero 完全一样**(V7.1 内部用的是 KataGo 风格的 PUCT)。

## 为什么 AlphaZero 在少量模拟下会失灵

​    AlphaZero 的训练目标是「根节点各子节点的访问次数分布」,落子则按访问次数采样。当模拟次数 $n$ 远小于动作数 $k$ 时(例如 19×19 围棋 $k=362$,却只跑 $n=2$ 次模拟),根节点只能访问到极少数动作,访问次数分布几乎就是随机的,**未必比神经网络的原始策略 $\pi$ 更好**。

​    论文给了一个反例:设三个动作的真实价值 $q=(0,0,1)$,神经网络策略 $\pi=(0.5,0.3,0.2)$。网络本身的期望价值是 $\sum_a \pi(a)q(a)=0.2$。若用「取概率最高的 $n=2$ 个动作 $\{0,1\}$,再在其中选 $q$ 最大的」这种朴素启发式,选出的动作期望价值是 $0$ —— **比不搜索还差**。问题的根源在于:概率最高的动作不一定价值最高,而少量模拟又盖不住所有动作。GumbelAlphaZero 就是为了堵上这个洞。

***

## 1、Gumbel-Top-k:无放回采样

​    GumbelAlphaZero 不再加 Dirichlet 噪声,而是用 **Gumbel-Top-k trick** 从策略里**无放回**地抽出 $m$ 个候选动作。

- **Gumbel-Max trick**:给每个动作的 logits 加一个独立的 Gumbel 噪声 $g(a)\sim\text{Gumbel}(0)$,取 $\arg\max_a\big(g(a)+\text{logits}(a)\big)$,等价于从 $\text{softmax}(\text{logits})$ 中**采样一个**动作。
- **Gumbel-Top-k trick**:用**同一组** $g$,取 $g(a)+\text{logits}(a)$ 最大的 $m$ 个动作,等价于从策略里**无放回采样 $m$ 个**动作。记作 $\texttt{argtop}(g+\text{logits},\,m)$。

```python
# g[a] ~ Gumbel(0, 1),每个动作一个;非法动作 logits 为 -inf
g = sample_gumbel(num_actions)
m = min(num_simulations, GUMBEL_M)        # V7.1: GUMBEL_M = 16
candidates = argtop(g + logits, m)        # 取 g+logits 最大的 m 个合法动作
```

> ​    评估 / 对弈时把 Gumbel 噪声关掉($g\equiv 0$),搜索就变成确定性的——直接取 logits 最大的 $m$ 个。噪声只在**自对弈**时打开,用来制造探索。

## 2、Sequential Halving:把模拟预算花在刀刃上

​    拿到 $m$ 个候选后,如何用有限的 $n$ 次模拟从中选出最好的一个?AlphaZero 的 PUCT 是为「最小化累积遗憾」设计的,但根节点真正在乎的是「**最后选出的那个动作有多好**」,即**最小化简单遗憾(simple regret)**。论文改用 **Sequential Halving** 算法:

​    把 $n$ 次模拟分成 $\lceil\log_2 m\rceil$ 轮;每一轮把当轮预算**均匀**分给所有存活候选(每个动作访问同样多次),轮末按分数排序、**淘汰掉一半**,直到只剩 1 个。候选规模因此是 $m\to m/2\to\cdots\to 1$。

```python
phases = ceil(log2(m))
sims_left = num_simulations
for phase in range(phases):
    # 本轮预算平均分给每个存活候选
    sims_this_phase = sims_left // (phases - phase)
    sims_per_action = max(1, sims_this_phase // len(candidates))
    for _ in range(sims_per_action):
        for a in candidates:
            simulate(root.children[a])     # 一次模拟:从该候选子节点起,PUCT 下潜→NN→回传
    sims_left -= sims_per_action * len(candidates)

    # 轮末:按 g + logits + σ(q̂) 排序,保留前一半
    if len(candidates) > 1:
        candidates.sort(key=lambda a: g[a] + logits[a] + sigma(q_hat(a)), reverse=True)
        candidates = candidates[: max(1, len(candidates) // 2)]

best_action = candidates[0]                 # 唯一存活者即为最终落子
```

> ​    每一次 `simulate()` 的内部和 AlphaZero 没有区别:从该候选动作的子节点出发,用 **PUCT** 一路下潜到未扩展的叶子,神经网络展开叶子并给出价值,再把价值**逐层取反**回传到根。GumbelAlphaZero 只是接管了「根节点这一层挑哪个动作来模拟」的调度权。

## 3、σ 变换与「补全的 Q 值(completed-Q)」

​    淘汰和最终排序用的分数是 $g(a)+\text{logits}(a)+\sigma(\hat q(a))$,其中 $\hat q(a)$ 是候选动作 $a$ 子节点的经验平均价值,$\sigma$ 是一个单调递增变换。论文式(8)给的具体形式是:

$$
\sigma(\hat q(a)) = \big(c_{visit} + \max_b N(b)\big)\cdot c_{scale}\cdot \hat q(a)
$$

​    $\max_b N(b)$ 是当前访问次数最多的动作的访问数。这个系数随搜索深入而变大,效果是:**初期先验 logits 说了算,随着访问增多,经验 Q 值逐步压过先验**。V7.1 默认 $c_{visit}=50,\ c_{scale}=1.0$(论文同款,换模拟数也不用调)。$\hat q$ 在双人棋里先归一化到 $[0,1]$:$\hat q=\frac{(W-L)+1}{2}$。

```python
def sigma(q_norm):                          # q_norm ∈ [0,1]
    max_n = max(child.n for child in root.children)
    return (C_VISIT + max_n) * C_SCALE * q_norm

def q_hat(a):                               # 候选 a 的经验动作价值
    child = root.children[a]
    return normalize(child.value()) if child.n > 0 else normalize(v_mix())
```

​    **未被访问的动作没有经验 Q 怎么办?** 论文式(10)用 $v_{mix}$ 来「补全」:已访问动作用经验 $\hat q$,未访问动作统一用一个估计值 $v_{mix}$,即把神经网络给的根价值和已访问子节点的 Q 做先验加权混合:

$$
v_{mix} = \frac{1}{1+\sum_b N(b)}\left(\hat v_{root} + \sum_b N(b)\cdot\frac{\sum_{a:N(a)>0}\pi(a)\,\hat q(a)}{\sum_{a:N(a)>0}\pi(a)}\right)
$$

```python
def v_mix():
    visited = [a for a in actions if root.children[a].n > 0]
    if not visited:
        return root.nn_value                # 一个都没访问,就用网络的根价值
    sum_n = sum(root.children[a].n for a in visited)
    wq = (sum(prior[a] * root.children[a].value() for a in visited)
          / sum(prior[a] for a in visited))
    return (root.nn_value + sum_n * wq) / (1 + sum_n)
```

## 最终策略与训练目标

​    搜索结束后:

- **落子动作**:Sequential Halving 最后唯一存活的那个动作(自对弈、评估、对弈都用它),**不再按访问次数采样**。
- **策略训练目标**:不再是访问次数分布,而是用 completed-Q 构造的**提升策略 $\pi'$**(论文式(11)):

$$
\pi' = \text{softmax}\big(\text{logits} + \sigma(\text{completedQ})\big)
$$

​    它对**所有**动作都有定义(已访问的用经验 Q,未访问的用 $v_{mix}$),信息量比 one-hot 的访问分布大得多。神经网络策略头朝 $\pi'$ 拟合($\text{loss}=\text{KL}(\pi',\pi)$);价值头仍朝对局最终胜负拟合。

```python
def gumbel_search(state, to_play):
    root = Node(state, to_play)
    logits, value = NN(encode(state))           # 策略 logits(合法屏蔽) + 价值
    expand(root, logits)
    backpropagate(root, value)

    # 1) 无放回采样 m 个候选
    g = sample_gumbel(num_actions) if selfplay else zeros(num_actions)
    m = min(num_simulations, GUMBEL_M)
    candidates = argtop(g + logits, m)

    # 2) Sequential Halving 选出胜者
    phases = ceil(log2(m))
    sims_left = num_simulations
    for phase in range(phases):
        sims_per_action = max(1, (sims_left // (phases - phase)) // len(candidates))
        for _ in range(sims_per_action):
            for a in candidates:
                simulate(root.children[a])      # 内部即 AlphaZero 的一次模拟(PUCT 下潜)
        sims_left -= sims_per_action * len(candidates)
        if len(candidates) > 1:
            candidates.sort(key=lambda a: g[a] + logits[a] + sigma(q_hat(a)), reverse=True)
            candidates = candidates[: max(1, len(candidates) // 2)]

    # 3) 训练目标 = 提升策略 π';落子 = 唯一存活者
    improved_policy = softmax(logits + sigma(completed_q(root)))
    best_action = candidates[0]
    return improved_policy, best_action
```

## 与 AlphaZero 的对照

| 机制 | AlphaZero | GumbelAlphaZero |
|---|---|---|
| 根节点探索 | 策略加 Dirichlet 噪声 | Gumbel-Top-k 无放回采样 |
| 根节点搜索调度 | PUCT(最小化累积遗憾) | Sequential Halving(最小化简单遗憾) |
| 落子 | 按访问次数 $N^{1/\tau}$ 采样 | Sequential Halving 的胜出动作 |
| 策略训练目标 | 访问次数分布 $N(a)/\sum N$ | completed-Q 提升策略 $\pi'$ |
| 少量模拟下 | 不保证策略提升 | **保证策略提升** |
| 树内部(非根)选择 | PUCT | PUCT(不变) |

> ​    **关于非根节点**:论文还提出了「Full Gumbel」变体,用确定性规则 $\arg\max_a\big(\pi'(a)-\frac{N(a)}{1+\sum_b N(b)}\big)$(式(14))替换树内部的 PUCT。但默认的 Gumbel AlphaZero / MuZero **只在根节点用 Gumbel,内部仍保留 PUCT**——SkyZero_V7.1 即采用这种默认配置(内部是 KataGo 风格、带 FPU 与方差缩放 cPUCT 的 PUCT)。

## 要点

1. GumbelAlphaZero 的价值在于**模拟次数少**的场景:论文里 9×9 围棋仅 2 次模拟也能稳定学习,而 AlphaZero 在 16 次以下就学不动了。
2. 三个改动是配套的:Gumbel 采样保证根价值不会变差,Sequential Halving 高效定位最优,completed-Q 让训练目标覆盖所有动作。
3. 它只动根节点;把树内部继续交给 PUCT,所以可以**无痛叠加**在已有的 AlphaZero / KataGo 实现之上。
