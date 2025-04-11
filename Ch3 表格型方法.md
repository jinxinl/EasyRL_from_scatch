# Ch3 表格型方法

[TOC]

寻找策略的最简单方法就是 *查找表 `look-up table`* ，即表格型策略。使用查找表的强化学习方法叫做 *表格型方法 `tabular method`* ，常见的 `tabular method` 有 `MC` 、`Q-learning` 和 `Sarsa` 

## 马尔可夫决策过程

马尔可夫决策过程是强化学习的经典框架，可以用五元组表示 $<S,A,P,R,\gamma>$ 

### 有模型

我们可以在与环境的交互中得到经验，从而估计出状态转移概率 $P$ 和奖励函数 $R$ ，当已知 $P$ 和 $R$ 时，可以称为 *环境已知* ，后续求解实际上是动态规划问题的求解，而不是强化学习

<img src="C:\Users\xinling\AppData\Roaming\Typora\typora-user-images\image-20250410221351181.png" alt="image-20250410221351181" style="zoom:70%;" />

### 免模型

现实中，状态空间可能很大，转移概率矩阵的估计很困难，因此大部分都是免模型的算法。智能体会在与环境的互动中学习哪一个动作的奖励更高，使用 $V(s)$ 表示，目标是最大化 $V(s)$ ，同时也会使用 $Q(s,a)$ 来表示在某个状态下应该采取什么样的行动 $a$ 。【$V(s)$ 表示状态的好坏，$Q(s,a)$ 表示 *状态-动作对* 的好坏】

<img src="C:\Users\xinling\AppData\Roaming\Typora\typora-user-images\image-20250410221407751.png" alt="image-20250410221407751" style="zoom:50%;" />

## Q表格

- 纵轴是状态，横轴是动作，$(s,a)$ 表示状态 $s$ 下采取行动 $a$ 的奖励

- 在 $t$ 时刻，需要执行能够最大化 *长期奖励的折扣总和* 的动作 $a_t$ 

  - 说明了强化学习的导向性很强，根据环境给出的奖励作为重要反馈，进行选择

  - 使用长期奖励的折扣总和 $G_t$ 是因为奖励具有延迟性，所以强化学习需要考虑到当下动作对未来的影响，学习未来奖励，当然为了防止考虑得太长远，需要给未来奖励乘上折扣因子 $\gamma$ 

    【考虑得太长远指：（1）若任务是持续式任务 `continuing task` ，比如股票，考虑10年后的大涨大停不太合理，所以需要折扣因子抵消时间的距离，降低未来很远的奖励的影响（2）有的状态转移是带环的，可能会陷入无穷奖励】

  - 【回顾】长期奖励的折扣总和 $G_t$ 的计算公式如下：
    $$
    G_t=r_{t+1}+\gamma r_{t+2}+...+\gamma^{T-t-1} r_T=r_{t+1}+\gamma G_{t+1}
    $$

- $Q$ 表格的更新就是强化学习的强化

  **强化** 是指用下一个状态的价值更新当前状态，也就是自举 `bootstrapping` 。RL中，没走一次就更新一次 $Q$ 表格，用下一个状态的 $Q$ 值来更新当前状态的 $Q$ 值。—— *时序差分法 `temporal difference, TD`*



## 免模型预测

在无法获取马尔可夫决策过程 $<S,A,P,R,\gamma>$ 时，使用免模型预测的方法，如蒙特卡洛 `Monte Carlo` 、时序差分法 `Temporal Difference` 

### 蒙特卡洛策略评估

- 蒙特卡洛方法

  - 思想：给定策略 $\pi$ ，采样 $N$ 条轨迹，计算所有轨迹回报 $G_t$ 的均值，即可得到该策略 $\pi$ 下某一状态的价值。使用了经验平均回报 `empirical mean value` 。公式如下：
    $$
    V_{\pi}(s)=\mathbb{E}_{r\sim\pi}(G_t|s_t=s)
    $$
    

  - 算法流程：

    - 在每个回合中，若是 $t$ 时刻中状态 $s$ 被访问了，那么
      - 状态 $s$ 的访问数 $N(s)$ 加 $1$ ：$N(s)\leftarrow N(s)+1$ 
      - 状态 $s$ 的总回报数 $S(s)$ 需要加上此时的回报 $G_t$ ：$S(s)\leftarrow S(s)+G_t$ 
    - 估计状态 $s$ 的价值：$V(s)\leftarrow S(s)/N(s)$ 

  - 数学原理：大数定律表明，当 $N(s)\rightarrow \infin$ 时，状态 $s$ 的估计价值 $V(s)\rightarrow V_{\pi}(s)$ 。

- 增量蒙特卡洛 `incremental MC` 方法

  - 思想：可以每次采样的轨迹当成是与时间有关的序列，使用上一时刻的价值更新当前价值

  - 推导：假设现在有 $t$ 时刻的均值 $\mu_t$ ，使用 $\mu_{t-1}$ 更新 $\mu_t$ 的推导如下
    $$
    \begin{align*}
    \mu_t &= \frac{1}{t}\sum_{i=0}^t x_i \\
    &= \frac{1}{t}(x_t+\sum_{i=0}^{t-1}x_i) \\
    &= \frac{1}{t}(x_t+(t-1)\mu_{t-1}) \\
    &= \frac{1}{t}x_t +\mu_{t-1}-\frac{1}{t}\mu_{t-1} \\
    &= \mu_{t-1}+\frac{1}{t}(x_t-\mu_{t-1})
    \end{align*}
    $$
    使用 $\mu_t=\mu_{t-1}+\frac{1}{t}(x_t-\mu_{t-1})$ 来作为 $t$ 时刻状态 $s$ 的价值 $V_{t}(s)$ 的更新依据，可以得到
    $$
    V(s_{t})=V(s_t)+\frac{1}{N(s_{t})}(G_t-V(s_{t}))
    $$
    因为 $N(s_t)$ 是人为设置的采样数，因此可以看成学习率，改写成如下形式：
    $$
    V(s_t)\leftarrow V(s_t)+\alpha(G_t-V_t)
    $$
     $G_t-V_t$ 可以看成残差，使用更新后的 $V(s_t)$ 来逼近真实的价值 $V_{\pi}(s)$ ，体现了 **强化** 的概念

- 时序差分

  - 思想：结合了自举 `bootstrapping` 和采样的方法，时序差分目的是对于某个给定的策略 $\pi$ ，计算出其价值 $V_{\pi}(s)$ 

  -  一步时序差分 `one-step TD` ：给定策略 $\pi$ 下，在该回合中每走一步，就做自举，使用当前得到的估计回报 $r_{t+1}+\gamma V(s_{t+1})$ 来更新上一时刻的价值 $V(s_t)$ 。其中，估计回报 $r_{t+1}+\gamma V({s_{t+1}})$ 也叫做时序差分的目标 `TD target` 。根据 `incremental MC` ，可以得到 `one-step TD` 的 **公式** ：
    $$
    V(s_t)\leftarrow V(s_t)+\alpha(r_{t+1}+\gamma V(s_{t+1})-V(s_t))
    $$

  - 目标

    - 走了某一步后的实际奖励 $r_{t+1}$ 
    - 使用自举的方法，通过自举来更新 $V(s_t)$ ，并且乘上了折扣因子

  -  `TD target` 是估计的原因：（1）时序差分的目标是求期望（2）时序差分方法使用当前的估计 $V$ 而不是真实的 $V_{\pi}$ 

  - 误差：类比 `incremental MC` ，在给定的回合中可以更新 $V(s_{t})$ 来逼近真实回报 $G_t$
    $$
    \delta=r_{t+1}+\gamma V(s_{t+1})-V(s_t)
    $$

  - 推广—— $n$ 步时序差分 `n-step TD` ：

    - $n=1$ ，$G^{(1)}_{t}=r_{t+1}+\gamma V(s_{t+1})$ 
    - $n=2$ ，$G^{(2)}_{t}=r_{t+1}+\gamma r_{t+2}+\gamma^2V(s_{t+2})$ 
    - $\cdots$ 
    - $n=\infin$ ，$G^{(\infin)}_t=r_{t+1}+\gamma r_{t+2}+\gamma^2r_{t+3}+\cdots+\gamma^{T-t-1}r_{T}$ ，此时相当于 `MC` ，到游戏结束计算真实回报

     更通用地来表示，$n$ 步时序差分可以写成
    $$
    G^{(n)}_{t}=r_{t+1}+\gamma r_{t+2}+\gamma^2r_{t+3}+\cdots+\gamma^{(n-1)}r_{t+n}+\gamma^{n}V(s_{t+n})
    $$
    得到 `TD target` 之后，就可以更新当前时刻状态的价值了
    $$
    V(s_t)\leftarrow V(s_t)+\alpha(G^{(n)}_{t}-V(s_t))
    $$
    **不断逼近 `TD target` **

- `MC` 、`TD` 、动态规划的联系：

  - `MC` v.s. `TD` :

    - `TD` 是在线学习 `online learning` ，每走一步就可以更新，效率高，`MC` 必须等到该回合结束时才能够更新
    - `TD` 是边走边更新，因此可以在不完整序列上学习，而 `MC` 必须在完整序列上学习，因为它要根据完整轨迹计算状态的真实回报 $G_t$ 
    - `TD` 可以在连续环境下学习（可以没有终止），`MC` 只能在有终止情况下学习
    - `TD` 利用了马尔可夫性质，当前状态只与上一个状态有关，能够在马尔科夫环境下有更高的学习效率，`MC` 没有假设环境具有马尔可夫性质，只是通过采样求均值来更新，因此在马尔可夫环境下不会更加有效
    - 与 `MC` 相比，`TD` 的优势是低方差、能够在线学习、能够从不完整序列中学习、能够适应无终止序列的情况

    时序差分：路上堵车，会实时更新到达后面每个地点的最新预计时间

    蒙特卡洛：路上堵车，到达目的地之后才更新路上到达某些中间位置的时间

  | 方面     | 蒙特卡洛                                                     | 时序差分                                                     | 动态规划                                                     |
  | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | 数学原理 | 经验均值<br /><br />$V(s_t)\leftarrow V(s_t)+\frac{1}{N(s_t)}(G_t-V(s_t))$ ，使用一个回合的经验平均回报进行更新<br />大数定律 | 经验均值<br />$V(s_t)\leftarrow r(s_{t+1})+\gamma V(s_{t+1})-V(s_t)$ | 贝尔曼期望方程备份<br />$V_{i}(s)\leftarrow \sum_{a\in A}\pi(a|s)[R(s,a)+\gamma\sum_{s'\in S}p(s'|s,a)V_{i-1}(s')]$<br />两层加和，计算两次期望 |
  | 更新方法 | 采样一条轨迹，对该轨迹上的状态进行更新。使用的是真实回报 $G_t$ ，因为此时轨迹已经结束 | 使用当前这一步得到的估计回报来更新当前状态的价值 $V(s_t)$ ，使用的是估计，因为游戏还在进行中，不知道真实回报 | 一次更新所有状态                                             |
  | 特点     | 有模型，对于状态空间很大的问题，效率很低                     | 可以处理没有终止的任务                                       | 免模型，适用于环境未知，只更新一条轨迹的状态，但只能用于有终止的 `MDP` |
  | 思想     | 采样                                                         | 采样+自举                                                    | 自举                                                         |

  <img src="C:\Users\xinling\AppData\Roaming\Typora\typora-user-images\image-20250411135450270.png" alt="image-20250411135450270" style="zoom:55%;" />

  <img src="C:\Users\xinling\AppData\Roaming\Typora\typora-user-images\image-20250411132216187.png" alt="image-20250411132216187" style="zoom:33%;" />

  <img src="C:\Users\xinling\AppData\Roaming\Typora\typora-user-images\image-20250411132236440.png" alt="image-20250411132236440" style="zoom:33%;" />

### `MC` 、`TD` 和 `DP` 中的采样和自举

目的都是计算给定策略 $\pi$ 下每个状态的价值 $V_{\pi}(s)$ 

- 采样：更新时通过采样得到一个期望

- 蒙特卡洛

  - 纯采样

  - 在给定策略 $\pi$ 下，选择一条完整轨迹，计算状态 $s$ 的真实回报 $G$ ，并使用 $G$ 更新 $s$ 的价值 $V(s)$ 。大数定律表明当采样的轨迹数量足够多时，$V(s)\rightarrow V_{\pi}(t)$  ，用全局的、真实的信息更新

    <img src="C:\Users\xinling\AppData\Roaming\Typora\typora-user-images\image-20250411143313949.png" alt="image-20250411143313949" style="zoom:50%;" />

- 时序差分

  - 采样+自举

  - 选择一条轨迹，每走一步就使用自举的方法，用估计回报更新价值，用局部信息更新

    <img src="C:\Users\xinling\AppData\Roaming\Typora\typora-user-images\image-20250411143339908.png" alt="image-20250411143339908" style="zoom:50%;" />

- 动态规划

  - 纯自举

  - 使用贝尔曼期望方程，对所有状态和动作进行加和，每一次更新需要进行两次加和（即求两次期望）

    <img src="C:\Users\xinling\AppData\Roaming\Typora\typora-user-images\image-20250411143252773.png" alt="image-20250411143252773" style="zoom:50%;" />

所以如果需要广度的更新，就选择动态规划，需要深度的更新，就选择蒙特卡洛。穷举既需要广度的更新，又需要深度的更新

<img src="C:\Users\xinling\AppData\Roaming\Typora\typora-user-images\image-20250411143359811.png" alt="image-20250411143359811" style="zoom:50%;" />

## 免模型控制

在不知道马尔可夫决策过程的情况下，优化价值函数并得到最佳策略。

使用能够兼容蒙特卡洛和时序差分的方法——广义策略迭代 `generalized policy iteration, GPI` 

- 广义迭代策略

  在第二章中介绍了马尔可夫决策过程的控制问题，输入是 $<S,A,P,R,\gamma>$ ，需要寻找最佳价值函数和最佳策略，策略迭代是解决 `MDP` 控制问题的方法之一，通过 *策略评估-策略改进* 的反复迭代，最终能够使状态价值函数 $V(s)$ 收敛，再利用贪心策略从 $Q$ 函数中抽取出 $\pi^*$ 函数。

  【从 $V_{\pi}(s)$ 得到 $Q_{\pi}(s,a)$ 
  $$
  Q_{\pi}(s,a)=R(s,a)+\gamma\sum_{s'\in S}p(s'|s,a)V_{\pi}(s')
  $$
  】

  然而，在免模型控制中，输入 $<S,A,P,R,\gamma>$ 是未知的，$P$ 和 $R$ 未知，因此无法根据状态价值函数计算出 $Q$ 函数，在策略评估部分遇到困难。

  广义策略迭代中，使用蒙特卡罗方法代替动态规划的方法估计 $Q$ 函数，使用蒙特卡洛采样的轨迹价值均值作为 $Q$ 函数的估计值，即 $Q=Q_{\pi}$ ，使用当前回合采样、计算得到的 $Q$ 值填充 $Q$ 表格，然后再利用贪心策略得到改进后的策略
  $$
  \pi_{i+1}=\arg\max_{a}Q(s,a)
  $$

  - 保证策略迭代收敛的假设：假设回合有 *探索性开始* 

    - 探索性开始：所有的动作和状态在经过无限步的执行之后，都能够被采样到

  - 算法：采样轨迹 $\rightarrow$ 计算每个状态的回报 $G_t$ $\rightarrow$ 使用每个状态的价值均值作为 $Q$ 值，填充 $Q$ 表格 $\rightarrow$ 贪心策略抽取改进后的 $\pi$  

    <img src="C:\Users\xinling\AppData\Roaming\Typora\typora-user-images\image-20250411152026696.png" alt="image-20250411152026696" style="zoom:67%;" />

  - $\epsilon-greedy$ 探索：为了保证蒙特卡洛方法能够有充分的探索，在改进 $\pi$ 函数时，智能体有 $1-\epsilon$ 的概率选择最大 $Q$ 值对应的动作，有 $\epsilon$ 的概率随机选取未探索的动作，$\epsilon$ 一般是一个很小的数，比如 $0.1$， 因为刚开始时探索到的动作比较少，不知道什么动作更好，但是随着探索次数增多，智能体对哪些动作比较好已经有了基本全面的概念，所以 $\epsilon$ 随着训练的进行会逐渐变小

    - $\epsilon-greedy$ 中 $V_{\pi}(s)$ 仍然能够收敛，仍具有单调性。正式表述如下：

      对于任何 $\epsilon-greedy$ 策略 $\pi$ ，关于 $Q_{\pi}$ 的 $\epsilon-greedy$ 策略 $\pi'$ 都是对 $\pi$ 的改进， 即 $V_{\pi}(s)\le V_{\pi'}(s)$ 

      <img src="C:\Users\xinling\AppData\Roaming\Typora\typora-user-images\image-20250411153405390.png" alt="image-20250411153405390" style="zoom:67%;" />

      【这是基于 $\epsilon-greedy$ 的蒙特卡洛方法（非增量式，因为是回合结束更新的），使用上一时刻的 $Q$ 值更新当前时刻的 $Q$ 值，蒙特卡洛方法是使用回报均值作为 $Q$ 值，前者是逐步更新（在回合中每往前走一步就更新），后者是批量更新（回合结束才更新）

      之所以上述算法流程中非增量形式的 `MC` 要写成增量的公式，是因为这样可以（1）避免存储历史 $Q$ 值，只需要维护 $N(s_t,a_t)$ 和 $Q(s_t,a_t)$ 即可，实现便捷，（2）并且考虑到后续封装接口时，能够与增量 `MC` 形式统一，代码简洁，容易切换模式。通过拆括号化简就能够还原成求均值的样子。】

  - 也可以把 `TD` 放到控制循环 `control loop` 中，估计 `Q` 表格，再利用 $\epsilon-greedy$ 得到改进后的策略，这样可以在回合未结束时更新已经采集到的状态价值

- 偏差与方差

  - 偏差：预测值期望与真实值期望的距离
    $$
    \|\mathbb{E}_{pred}-\mathbb{E}_{real}\|
    $$
    偏差越大，说明预测越偏离真实情况

  - 方差：预测的各个值距离预测期望的距离
    $$
    \mathbb{E}[\hat{y}-\mathbb{E}(\hat{y})]
    $$
    方差越大，说明预测值之间越分散

    <img src="C:\Users\xinling\AppData\Roaming\Typora\typora-user-images\image-20250411154844245.png" alt="image-20250411154844245" style="zoom:43%;" />

## 时序差分控制

- 同策略 `on-policy` 时序差分控制

  - 在给定策略 $\pi$ 下，使用 $\epsilon-greedy$ 的方法，寻找最优策略。使用同一种策略进行动作选择和更新

  - 例子：`Sarsa` 算法

    - 输入：$<S,A,R,S',A'>$ ，这也是 `Sarsa` 算法名字的来源

    - 思想：根据当前时刻的状态 $s_t$ 、行动 $a_t$ 得到的奖励 $r_t$ 以及下一时刻的 $s_{t+1}$ $a_{t+1}$ ，直接对 $Q$ 函数进行更新
      $$
      Q(s_t,a_t) \leftarrow Q(s_t,a_t)+\alpha(r_{t+1}+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t))
      $$
      也就是原本的时序差分法对 $V(s)$ 进行更新，这里的 `Sarsa` 直接对 $Q$ 函数进行更新、直接估计 $Q$ 表格

    - 目标值：$r_{t+1}+\gamma Q(s_{t+1},a_{t+1})$ 

      为了使当前的 $Q(s_t,a_t)$ 逼近这个目标值，`Sarsa` 和 `TD` 一样，使用学习率乘上残差的方法对当前 $Q(s_t,a_t)$ 进行 **软更新** ，每次只更新一点点，由上述公式可以得到 `Sarsa` 算法的更新公式
      $$
      Q(S,A)\leftarrow Q(S,A)+\alpha[R+\gamma Q(S',A')-Q(S,A)]
      $$
      

    - $n$ 步 $Sarsa$ ：与 `TD` 类似，把 $V(s)$ 改成 $Q(s,a)$ 

      - $n=1$ ，$Q^{(1)}_{t}=r_{t+1}+\gamma Q(s_{t+1},a_{t+1})$ 
      - $n=2$ ，$Q^{(2)}_t=r_{t+1}+\gamma r_{t+2}+\gamma^2Q(s_{t+2},a_{t+2})$ 
      - $\cdots$ 
      - $n=\infin$ ，$Q^{\infin}_t=r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\cdots +\gamma^{T-t-1}r_{T}$ 

      更通用的公式：$n$ 步 $Sarsa$ 
      $$
      Q^{(n)}_t=r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\cdots+\gamma^{n-1}r_{t+n}+\gamma^{n} Q(s_{t+n},a_{t+n})
      $$

    - $Sarsa(\lambda)$ 的 $Q$ 回报：给 $Q^{n}_t$ 加上资格迹衰减参数 `decay-rate parameter for eligibility traces` $\lambda$ 并进行求和，能够得到 $Q^{\lambda}_t$ 
      $$
      Q^{\lambda}_{t}=(1-\lambda)\sum_{n=1}^{\infin}\lambda^{n-1}Q^{n}_t
      $$
      $n$ 步 $Sarsa(\lambda)$ 的更新策略是
      $$
      Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha(Q^{\lambda}_t-Q(s_t,a_t))
      $$

    - 

     $Sarsa$ 与 $Sarsa(\lambda)$ 的区别主要体现在更新策略上

    - 算法流程：
      - 当前状态已知，从当前的 $Q$ 表格中选择下一步动作 $\rightarrow$ $<S,A,R,S',A'>$ 均已知
      - 更新 $Q$ 表格

- 异策略 `off-policy` 时序差分控制

  - 在学习过程中有两种不同的策略：目标策略 `taget policy` 与行为策略 `behaviour policy` 。

    - 目标策略 $\pi$ 负责选择下一步动作，使用贪心策略，选择使当前 $s$ 的 $Q$ 值最大的动作——”军师“
    - 行为策略 $\mu$ 负责与环境交互，得到许多条轨迹，从轨迹中学习经验，并把经验告诉目标策略，（经验中不用包含 $a_{t+1}$ ，因为目标策略会用贪心策略自己选择）——”战士“

  - 例子： $Q$ 学习 `Q-learning` 

    - 目标策略直接在 $Q$ 表格上使用贪心策略
      $$
      \pi(s_{t+1})=\arg\max_{a'}Q(s_{t+1},a')
      $$

    - 行为策略是一个随机的策略，但是一般使用 $\epsilon-greedy$ 策略。

    - 目标：因为 $Q$ 学习的下一步是用贪心策略选出来的，所以：
      $$
      \begin{align*}
      r_{t+1}+\gamma Q(s_{t+1},A') &= r_{t+1}+\gamma Q(s_{t+1},\arg\max_a'Q(s_{t+1},a')) \\
      &= r_{t+1}+\gamma\max_{a'}Q(s_{t+1},a')
      \end{align*}
      $$

    - 更新策略：
      $$
      Q(S,A)\leftarrow Q(S,A)+\alpha[R+\gamma\max_{a}Q(S',a)-Q(S,A)]
      $$
      

- 区分 `Sarsa` 和 `Q-learning` 

  - `Sarsa` 
    $$
    Q(S,A)\leftarrow Q(S,A)+\alpha[R+\gamma Q(S',A')-Q(S,A)]
    $$
    `Q-learning` 
    $$
    Q(S,A)\leftarrow Q(S,A)+\alpha[R+\gamma\max_{a}Q(S',a)-Q(S,A)]
    $$

  - 区别：

    - 更新公式中的目标不同。

      - `Sarsa` 自己与环境的交互中产生轨迹 $<s,a,r,s',a'>$ ，用 $Q(S',A')$ 更新 $Q(S,A)$ ，使用 $\epsilon-greedy$ 策略
      - `Q-learning` 并不需要直到 $a'$ ，目标策略会根据贪心策略在吧 $Q$ 表格中选择 $a'$ ，目标策略选择 $a'$ 是贪心策略，行为策略是随机的，自由探索，不过一般是 $\epsilon-greedy$ 策略

      `Sarsa` 在选择时有一定概率不会选择最大化 $Q$ 值的动作，较为保守，会选择相对安全的路线，而 `Q-learning` 的每一步都最大化 $Q$ 值，是一种激进的方法。

    - `Sarsa` 兼顾探索，而 `Q-learning` 的行为策略是随机的，将探索得到的经验告诉目标策略，目标策略会利用探索到的数据采用贪心策略，不需要兼顾探索

    - `Sarsa` 是同策略，使用一种 $\pi$ 学习、与环境交互，在探索时若是发现了不好的动作，会尽可能不选择，尽量呆在安全区内，并且因为 $\epsilon$ 不断变小，因此策略并不稳定；`Q-learning` 无需兼顾探索，解释见上一条

      <img src="C:\Users\xinling\AppData\Roaming\Typora\typora-user-images\image-20250411175619283.png" alt="image-20250411175619283" style="zoom:80%;" />