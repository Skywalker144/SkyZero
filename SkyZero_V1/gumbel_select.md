# Gumbel MuZero 非根节点 Select 规则

参考论文: Danihelka et al., *"Policy improvement by planning with Gumbel"*, ICLR 2022.

---

## 核心公式

每次要在非根节点选一个动作展开,选:

$$
a^* = \arg\max_a \Big(\pi_{\text{improved}}(a) - \frac{N(a)}{1 + \sum_b N(b)}\Big)
$$

其中

$$
\pi_{\text{improved}} = \mathrm{softmax}\big(\text{logits} + \sigma(q_{\text{completed}})\big)
$$

是用 completed Q 算出来的"改进策略"。

---

## 三个关键组件

### 1. completed Q(a)

每个动作都需要一个 Q,但搜索预算少时大部分子节点 $N=0$。论文做法:

$$
q_{\text{completed}}(a) =
\begin{cases}
Q(a), & N(a) > 0 \\
v_{\text{mix}}, & N(a) = 0
\end{cases}
$$

其中 $v_{\text{mix}}$ 是 NN value 和"已访问子节点的访问加权 Q"的混合:

$$
v_{\text{mix}} = \frac{1}{1+\sum_b N(b)}\Big(v_{\text{NN}} + \sum_b N(b)\cdot \bar{q}\Big)
$$

$\bar{q}$ 用先验 $\pi$ 在已访问子节点上的归一化加权:

$$
\bar{q} = \frac{\sum_a \pi(a)\,\mathbb{1}[N(a)>0]\,Q(a)}{\sum_a \pi(a)\,\mathbb{1}[N(a)>0]}
$$

**直觉**: 没访问过的动作,借用"基于当前已知信息的合理估计"作 Q,这样所有动作都能算出 $\pi_{\text{improved}}$,而不是只对访问过的子集有定义。

### 2. $\sigma$ 变换

$$
\sigma(q) = \big(c_{\text{visit}} + \max_b N(b)\big)\cdot c_{\text{scale}}\cdot q
$$

作用是把 Q 缩放到和 logits 同一量级。$\max N$ 在里面是关键:**搜索越深,$\sigma(q)$ 的权重越大**,Q 信息逐渐压过先验 logits。这相当于 PUCT 里"随着 N 增大,exploitation 项占主导"的等价机制。

典型超参 (论文): $c_{\text{visit}} = 50$,$c_{\text{scale}} = 1.0$。

### 3. 比例规则 (the matching rule)

$$
a^* = \arg\max_a \Big(\pi_{\text{improved}}(a) - \frac{N(a)}{1+\sum_b N(b)}\Big)
$$

可以这么理解:

- 目标: 让最终的访问分布 $N(a)/\sum N$ **逼近** $\pi_{\text{improved}}$
- 如果某动作 $a$ 的目标概率是 $\pi_{\text{improved}}(a) = 0.3$,但已访问占比到 $N(a)/\sum N = 0.5$,它就"超额"了,不需要再访问
- 反之访问占比远低于目标概率的动作就是"欠缺的",优先访问

**数学推导**: 我们希望分配下一次访问后, $\dfrac{N(a)+1}{\sum N+1}$ 尽量接近 $\pi_{\text{improved}}(a)$。等价于让

$$
\pi_{\text{improved}}(a)\cdot\Big(\textstyle\sum N + 1\Big) - N(a)
$$

最大化的那个 $a$ 优先访问。两边除以常数 $\sum N + 1$,就得到论文公式。

---

## 与 PUCT 的对比

| 项目             | PUCT                                                                   | Gumbel 非根 select                                          |
| ---------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------- |
| 公式             | $Q(a) + c_{\text{puct}}\,\pi(a)\dfrac{\sqrt{\sum N}}{1+N(a)}$           | $\pi_{\text{improved}}(a) - \dfrac{N(a)}{1+\sum N}$         |
| 用的先验         | NN 的原始 $\pi$                                                        | Q 校正过的 $\pi_{\text{improved}}$                          |
| 探索机制         | $c_{\text{puct}}$ 超参 + UCB 项                                        | 已隐含在 $\sigma(q)$ 的 $\max N$ 缩放中                     |
| Q 处理           | 只用访问过的                                                           | 全部动作都用 completed Q                                    |
| 是否保证策略改进 | 否 (经验上 work)                                                       | 是 (论文 Theorem 2 给出证明)                                |

---

## 为什么这样设计能带来理论保证

PUCT 用 $N(a)/\sum N$ 作为 policy target 时,有个隐含假设: visit 分布最终会收敛到某个"好"的策略。但实际上它收敛到什么没人证过——只是经验上 work。

Gumbel 非根规则**直接强制** visit 分布去匹配 $\pi_{\text{improved}}$。而 $\pi_{\text{improved}}$ 本身可以证明优于 NN 的原 $\pi$ (用 completed Q 信息修正过)。所以:

$$
\text{visit 分布} \;\approx\; \pi_{\text{improved}} \;\succeq\; \pi_{\text{NN}}
\quad\text{(策略改进)}
$$

整条链路有数学保证,这是论文最大的卖点——即使在 sim 预算极小 (比如 16 次) 的情况下,policy target 依然是"好的",不会被噪声毁掉。

---

## 算法伪代码

```python
def select_gumbel(node):
    # 1. 收集子节点统计
    for c in node.children:
        if c.n > 0:
            Q[c.action] = flip_perspective(c.wdl / c.n)
            N[c.action] = c.n

    # 2. v_mix
    sum_n = N.sum()
    if sum_n > 0:
        weighted_q = sum(pi_visited * Q) / sum(pi_visited)
        v_mix = (nn_value + sum_n * weighted_q) / (1 + sum_n)
    else:
        v_mix = nn_value

    # 3. completed Q
    q_completed = where(N > 0, Q, v_mix)
    q_scalar    = wdl_to_scalar(q_completed)            # → [0, 1]

    # 4. sigma 变换 + improved policy
    sigma_q         = (c_visit + N.max()) * c_scale * q_scalar
    improved_logits = logits + sigma_q
    improved_policy = softmax(improved_logits)

    # 5. 比例规则
    score = improved_policy - N / (1 + sum_n)
    return argmax(score over node.children)
```

---

## 与代码的对应 (`skyzero.py`)

`_select_gumbel(node)` 中:

| 论文概念                       | 代码变量                                          |
| ------------------------------ | ------------------------------------------------- |
| $Q(a)$ (访问过)                | `q_wdl[a]` (从子视角翻转到父视角)                 |
| $N(a)$                         | `n_values[a]`                                     |
| $v_{\text{mix}}$               | `v_mix_wdl`                                       |
| $q_{\text{completed}}$         | `completed_q_wdl` → `completed_q_scalar`          |
| $\sigma(q_{\text{completed}})$ | `sigma_q`                                         |
| $\pi_{\text{improved}}$        | `improved_policy = softmax(logits + sigma_q)`     |
| 比例规则 score                 | `improved_policy - n_values / (1 + sum_n)`        |

每个非根节点都需要 `nn_policy_logits` 和 `nn_value_probs`,所以 `Node` 类加了这两个字段,在 `_inference` 里写入。

---

## 超参开关

在 `args` 里设置:

```python
args["gumbel_non_root_selection"] = "puct"    # 默认,沿用 cPUCT
args["gumbel_non_root_selection"] = "gumbel"  # 论文规则
```

只影响 `gumbel_sequential_halving` 内部的非根 select,根节点的 Sequential Halving 逻辑不变。
