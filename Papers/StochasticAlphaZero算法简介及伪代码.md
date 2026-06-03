# StochasticAlphaZero

​    **StochasticAlphaZero** 把 AlphaZero 从「确定性、双人零和」推广到「**随机、单智能体**」的环境。AlphaZero 默认走一步棋后局面是唯一确定的;但像 **2048** 这样的游戏,每滑动一步后,**环境会随机生成一个新方块**(90% 是 2、10% 是 4,位置在空格里均匀随机),下一个局面并不唯一。StochasticAlphaZero 借用 Stochastic MuZero(Antonoglou et al., ICLR 2022《Planning in Stochastic Environments with a Learned Model》)的 **afterstate(后状态)+ chance node(机会节点)** 结构,把「玩家决策」和「环境随机」拆成两层节点来搜索。SkyZero_2048 用的就是它(单智能体 + 标量折扣回报价值 + 继承自 V7.1 的 Gumbel 根)。

> ​    本文假设你已读过 `AlphaZero算法简介及伪代码.md` 与 `GumbelAlphaZero算法简介及伪代码.md`。这里只讲**随机环境**带来的改动:节点结构、价值语义、机会节点的处理。根节点的 Gumbel 选择沿用 GumbelAlphaZero,不再重复。

## 和 AlphaZero 的三个根本差异

| | AlphaZero(围棋/五子棋) | StochasticAlphaZero(2048) |
|---|---|---|
| 玩家 | 两人零和 | **单智能体** |
| 视角 | 有 `to_play`,价值随层翻转 | **无 `to_play`,不翻转** |
| 价值 | WDL(胜/和/负),相对视角 | **标量**:期望折扣回报 $\mathbb E[\sum \gamma^t r_t]$ |
| 环境转移 | $s,a\to s'$ 唯一确定 | $s,a\to\text{afterstate}\xrightarrow{\text{随机}} s'$ |
| 树节点 | 只有 decision 节点 | decision 与 **chance 节点交替** |
| 反向传播 | $v\leftarrow -v$ 逐层翻转 | $G\leftarrow r+\gamma G$,**无翻转** |

***

## 1、afterstate 与 chance node:把一步拆成两半

​    随机环境里,一步转移其实包含两个阶段:**玩家的确定性动作** + **环境的随机响应**。StochasticAlphaZero 用两类节点把它们分开:

```
decision(s) ──action a──▶ chance(afterstate, reward) ──spawn──▶ decision(s')
  玩家选 4 个方向          确定性滑动+合并(本步得分)         环境随机生成一个方块
```

- **decision 节点(决策节点)**:轮到**玩家**决策,从 4 个方向里选一个。根节点永远是 decision 节点。
- **afterstate(后状态)**:动作执行完、但环境还没放新方块时的**中间局面**。2048 里就是「滑动+合并之后、生成新方块之前」的棋盘;这一步是**确定性**的,并顺带产生本步奖励 `reward`(被合并方块的数值之和)。
- **chance 节点(机会节点)**:挂在 afterstate 上,代表**环境的随机分裂**。它的每条边是一个可能的后继(在某个空格生成 2 或 4),边上标着**已知的**发生概率。

​    decision 与 chance 沿树深度交替出现,每个 decision 节点的父亲都是一个 chance 节点。

> ​    **关键:为什么是 Stochastic _AlphaZero_ 而不是 _MuZero_?** Stochastic MuZero 面对的是**未知动态**,得用 VQ-VAE 去**学习**机会分布 $Pr(c\mid as)$、并在 latent 空间里搜索。而 2048 的生成规则是**已知**的(90%/10%、空格均匀),所以这里直接**枚举**真实的机会分布、在**真实棋盘**上搜索——这正是 AlphaZero「完美模型 + 真实状态」的做法。换句话说:**借 Stochastic MuZero 的树结构,用 AlphaZero 的模型假设**。

| | Stochastic MuZero | StochasticAlphaZero(2048) |
|---|---|---|
| 模型 | 学习的 latent 模型 | **真实模拟器(完美模型)** |
| 状态 | 抽象 latent | **真实棋盘** |
| 机会分布 | 用 VQ-VAE 学 $Pr(c\mid as)$ | **枚举已知 spawn 分布** |
| chance 下潜 | $\arg\max_c \frac{Pr(c\mid as)}{N(c)+1}$ | $\arg\max_i\big(p_i-\frac{N_i}{N}\big)$(同理:逼近已知分布) |

## 2、价值:标量、折扣、不翻转

​    单智能体没有「对手」,价值不再是 WDL、也不需要逐层取反。一个状态的价值是从它出发能拿到的**期望折扣回报**:

$$
V(s)=\mathbb E\Big[\textstyle\sum_{t\ge 0}\gamma^{t}\,r_{t}\Big]
$$

​    反向传播就是一路**折扣累加**($\gamma$ 默认 0.999):每经过一个 chance 节点,就把那一步的即时奖励加进来,$G\leftarrow r+\gamma G$,**全程没有翻转**。神经网络价值头输出一个非负标量(softplus),训练时回归 `目标 / value_scale`(2048 里 `value_scale=4000`,因为回报能到上万,需要缩放)。

> ​    2048 的回报能达到几万,直接拿来比较会压垮 PUCT 的探索项,所以 PUCT 里的 $Q$ 要做 **MuZero 式 min-max 归一化**:用本次搜索见过的 $Q$ 的最小/最大值把它压到 $[0,1]$。(另:`data2048_vt` 实验还可选 MuZero 的可逆变换 $h(x)=\text{sign}(x)(\sqrt{|x|+1}-1)+\varepsilon x$ 来压缩价值目标,推理时再用 $h^{-1}$ 还原回真实分数——属于实现细节。)

## 3、一次模拟:decision 用 PUCT,chance 按已知分布

​    节点定义:

```python
class DecisionNode:                 # 玩家决策节点
    state; terminal; expanded
    prior[4]                        # 神经网络策略(4 个方向)
    children[4]                     # 每个合法动作 → 一个 ChanceNode
    nn_value; n; w
    def value(): return w / n if n > 0 else nn_value

class ChanceNode:                   # 环境随机节点
    afterstate; reward              # 确定性滑动的结果 + 本步得分
    edges                           # [(prob, cell, exp, child_or_None), ...] 来自已知 spawn 分布
    n; w
    def q(): return w / n if n > 0 else 0.0
```

​    一次模拟从根(decision)出发,在 decision 节点用 PUCT 选动作、在 chance 节点按已知分布选后继,交替下潜,直到撞上未扩展的叶子或终局:

```python
def simulate(root):
    path, rewards, node = [root], [], root
    while True:
        if node.terminal:
            backup(path, rewards, leaf_value=0.0)        # 游戏结束,叶子价值为 0
            return
        if not node.expanded:
            logits, value = NN(encode(node.state))
            expand(node, logits, value)                  # 建 chance 子节点 + 枚举 spawn 边
            backup(path, rewards, leaf_value=value)
            return
        a = select_action(node)                          # decision 节点:PUCT
        chance = node.children[a]
        path.append(chance); rewards.append(chance.reward)
        node = descend_chance(chance)                    # chance 节点:按已知分布选后继
        path.append(node)
```

**decision 节点 —— PUCT(和 AlphaZero 同形,Q 改用折扣回报):**

```python
def select_action(node):
    best, best_a = -inf, -1
    for a in legal_actions(node.state):
        ch = node.children[a]
        if ch.n > 0:
            q = ch.q()
        else:
            q = ch.reward + GAMMA * node.value()         # 未访问:即时奖励 + 折扣后的节点价值
        score = norm(q) + C_PUCT * node.prior[a] * sqrt(node.n) / (1 + ch.n)
        if score > best:
            best, best_a = score, a
    return best_a
```

**chance 节点 —— 按已知分布做「确定性分层采样」:** 选那个**实际访问频率最低于其已知概率**的后继,让访问分布逐步逼近真实的生成分布。

```python
def descend_chance(chance):
    best_i, best_deficit = -1, -inf
    for i, (prob, cell, exp, child) in enumerate(chance.edges):
        n_child = child.n if child else 0
        frac = n_child / chance.n if chance.n > 0 else 0.0
        deficit = prob - frac                            # 「欠访问」程度
        if deficit > best_deficit:
            best_deficit, best_i = deficit, i
    prob, cell, exp, child = chance.edges[best_i]
    if child is None:                                    # 懒扩展:第一次访问才真正建出后继棋局
        next_state = chance.afterstate.copy()
        next_state[cell] = exp                           # 在该空格放上 2(exp=1) 或 4(exp=2)
        child = DecisionNode(next_state)
        chance.edges[best_i].child = child
    return child
```

## 4、扩展:枚举已知的机会分布

​    展开一个 decision 叶子:神经网络给出策略和价值;对每个合法动作,先做确定性滑动得到 afterstate 和 reward,再把该 afterstate 下**所有可能的生成结果**枚举成 chance 节点的边。

```python
def expand(node, logits, value):
    node.expanded = True
    node.nn_value = value
    node.prior = softmax(mask_illegal(logits))
    for a in legal_actions(node.state):
        afterstate, reward, _ = apply_move(node.state, a)        # 确定性滑动+合并
        chance = ChanceNode(afterstate, reward)
        # spawn_distribution:每个空格 → (该格, 2, 0.9/空格数) 和 (该格, 4, 0.1/空格数)
        chance.edges = [(prob, cell, exp, None)
                        for (cell, exp, prob) in spawn_distribution(afterstate)]
        node.children[a] = chance
```

## 5、反向传播:折扣累加,无翻转

​    `path` 形如 `[dec0, chance0, dec1, chance1, ..., decK]`。从叶子的价值出发往回走,每碰到一个 chance 节点就把那一步奖励折进来;decision 节点直接累加当前回报。**没有 `value *= -1`**。

```python
def backup(path, rewards, leaf_value):
    g = leaf_value
    for node in reversed(path):
        if isinstance(node, ChanceNode):
            g = rewards.pop() + GAMMA * g            # 经过 chance:加本步即时奖励,再折扣
        node.n += 1
        node.w += g
        minmax.update(g)                             # 更新 min-max 归一化范围(供 PUCT 的 norm)
```

​    这样每个 decision 节点的 `w/n` 就是「从该局面出发的期望折扣回报」,每个 chance 节点的 `w/n` 是「执行完动作(含本步得分)之后的期望折扣回报」,正好对应 $V(as)=Q(s,a)$。

## 根节点与训练目标

​    根节点的动作选择、$\sigma$ 变换、completed-Q 提升策略 $\pi'$ **完全沿用 GumbelAlphaZero**(2048 只有 4 个方向,直接拿全部合法动作当候选,$m$ 即合法动作数,无需像围棋那样设上限 `GUMBEL_M`):落子取 Sequential Halving 的胜出方向,策略头朝 $\pi'$ 拟合。价值头的训练目标是该状态实际取得的折扣回报(MC 或 TD)。

```python
def search(state):
    root = DecisionNode(state)
    logits, value = NN(encode(state))
    expand(root, logits, value)
    backup([root], [], leaf_value=value)

    # —— 以下与 GumbelAlphaZero 相同:Gumbel-Top-k + Sequential Halving ——
    candidates = gumbel_top_k(root, m=num_legal_actions)        # 2048 只有 ≤4 个方向
    for phase in range(ceil(log2(len(candidates)))):
        run_simulations_evenly(root, candidates)     # 每次 simulate(root) 即第 3 节的下潜
        candidates = halve_by_score(candidates)      # 按 g+logits+σ(q̂) 砍一半

    improved_policy = softmax(logits + sigma(completed_q(root)))   # 训练目标 π'
    best_action = candidates[0]                                    # 落子方向
    return improved_policy, best_action
```

## 要点

1. **拆两层**是核心:decision 节点管玩家决策(PUCT),chance 节点管环境随机(按已知分布下潜),二者交替。
2. **价值是标量折扣回报**,反向传播只做 $G=r+\gamma G$、不翻转——这是「单智能体」带来的简化。
3. **机会分布已知 → 枚举而非学习**,所以是 Stochastic _AlphaZero_;若动态未知、要学 latent 机会模型,那才是 Stochastic _MuZero_。
4. 根节点照搬 GumbelAlphaZero,所以 SkyZero_2048 实际上是 **afterstate + Stochastic + Gumbel + AlphaZero** 的合体。
