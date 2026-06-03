# AlphaZero

​    **AlphaZero**是DeepMind开发的人工智能棋类算法，它能在**不需要人类数据**(例如棋谱)的情况下**从零开始**通过**自对弈**学习在**任何棋类游戏**中达到**超越人类**的表现。

## AlphaZero搜索算法流程

​    AlphaZero在走一步棋的时候会进行上千次模拟，首先将当前棋局作为根节点，每次模拟都会进行 **选择**、**扩展**、**反向传播** 三个步骤：

```python
# 将当前棋局作为根节点。to_play代表轮到谁走子。1代表黑方(先手)，-1代表白方(后手)
root = Node(
	state,
	to_play
)
for _ in range(num_simulations):
	1. 选择
	2. 扩展
	3. 反向传播
```

​    接下来具体介绍每个步骤的内容。

***

### 1、选择

​    从根节点开始，选择**PUCT**值最大的子节点，直到选择到一个**未被扩展的**节点

```python
# ------- 1、选择 ---------
while node.is_expanded():
	node = select(node)
# -------------------------

def select(node):  # 按照PUCT公式选择PUCT值最大的子节点
	max_puct = -inf
	best_child = None
	for child in node.children:
		if get_puct(child) > max_puct:
			max_puct = get_puct(child)
			best_child = child
	return best_child

def get_puct(node, c_puct=2):
	q = -node.v / node.n if node.n > 0 else 0
	# 对于父节点来说，当前节点的收益需要取反；因为子节点的胜利即是父节点的失败
	u = c_puct * node.prior * sqrt(node.parent.n) / (1 + node.n)
	return q + u
```

**PUCT** (Predictive Upper Confidence Bound applied to Trees)公式如下：
$$
PUCT(i) = \frac{v}{n_i} + C_{puct} \cdot Prior \cdot \frac{\sqrt{N}}{1 + n_i}
$$
 $v$：累计价值(胜利$v+1$，失败$v-1$，或者加上神经网络给出的value)

 $n_i$：节点i的访问次数

 $Prior$：神经网路给出的先验概率 $Prior = NN_{Policy}[action]$ (NN:NeuralNetwork)

 $C_{puct}$：探索系数

 $N$：父节点的访问次数

> ​    PUCT 最核心的改进在于引入了**先验知识（Prior Probability）**，是神经网络输出的策略的对应动作的概率值。即使某个节点还没被访问过，算法也会根据“直觉”给予它不同的初始关注度。
>
> **PUCT 的动态平衡过程：**
>
> 1、**初期（高权重先验）：** 在搜索初始阶段，由于每个节点的访问次数 $n_i$ 都很小，**先验概率 $Prior$ 占据主导地位**。算法会优先搜索神经网络认为“最有希望”的路径，而不是盲目地在所有子节点中平摊搜索次数。这极大提高了搜索效率。
>
> 2、**中期：** 如果神经网络看好的路径在实际模拟中表现不佳（$v/n_i$ 值下降），或者原本不被看好的路径通过少量尝试展现了极高的潜力，**探索项**和**利用项**的合力会引导算法转向。
>
> 3、**后期：** 随着访问次数 $n_i$ 变得非常大，先验概率 $Prior$ 的影响会逐渐减弱。如果搜索次数趋于无穷大，算法最终会摆脱神经网络的偏见，收敛到**实际的最优解**。

### 2、扩展

​    如果当前节点为终局节点，则获取当前赢家并且转换为当前节点的相对胜负；

否则使用神经网络展开当前节点的子节点，并且获得神经网络给出的节点的价值

```python
# ------- 2、扩展 ---------
if node.state.is_terminal():
	value = node.state.get_winner() * node.to_play  # 转换成当前节点玩家的视角
else:
	value = expand(node)
# -------------------------

def expand(node):
	encoded_state = encode_state(node.state, node.to_play)  # 将棋局编码
	nn_policy, nn_value = NN(encoded_state)  # 神经网络输出策略和相对价值
	
	nn_policy = mask_policy(nn_policy)  # 屏蔽非法动作（已有的棋子以及例如五子棋的禁手）
	
	nn_policy = softmax(nn_policy)
	
	for action in range(len(nn_policy)):
		to_play = node.to_play
		if nn_policy[action] > 0:  # 只展开合法动作的子节点
			child = Node(
				state=node.state.get_next_state(action, to_play),
				to_play=-to_play,
				prior=nn_policy[action],
				parent=node,
				action_taken=action
			)
			node.children.append(child)
	return nn_value
```



### 3、反向传播

​    将赢家按照相对视角获得各节点的价值，一路往上从当前节点传到根节点。

```python
# ----- 3、反向传播 -------
backpropagate(node, value)
# -------------------------

def backpropagate(node, value):
	while node is not None:
		node.v += value
		node.n += 1
		value *= -1  # 转换视角，因为当前节点和其父节点和to_play相反，所以相对价值也相反
		node = node.parent
```



## 最终策略

​    选取根节点的访问次数最高的子节点，该子节点的action_taken即为最终动作。

同时，将根节点的**各个子节点的访问次数分布**作为神经网络的**策略训练目标**；**棋局最终输赢**作为神经网络的**价值训练目标**

```python
def search(state, to_play):
	root = Node(state, to_play)

	for _ in range(num_simulations):
		node = root

		# 1、选择
		while node.is_expanded():
			node = select(node)

		# 2、扩展
		if node.state.is_terminal():
			value = node.state.get_winner() * node.to_play  # 转换成当前节点的视角
		else:
			value = expand(node)
			
		# 3、反向传播
		backpropagate(node, value)
	
	# 获取训练目标
	mcts_policy = 全零数组
	for child in root.children:
		mcts_policy[child.action_taken] = child.n
	mcts_policy /= sum(mcts_policy)
	
	# 评估时选择访问次数最多的动作
	best_child = max(root.children, key=lambda c: c.n)
	
	return mcts_policy, best_child.action_taken
```



## AlphaZero学习理论

- **原始策略 ($p$)**：神经网络给出的初始“灵感”
- **提升策略 ($\pi$)**：通过 MCTS 进行数百次模拟
- **结论**：经过搜索后的访问次数分布 $\pi$ 代表了一个**比原始 $p$ 更强大、更明智的决策**。通过让神经网络去拟合 $\pi$，本质上是让网络学习“搜索后的智慧”。
