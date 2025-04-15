---
layout: course_page
title: Reinforcement Learning - Learning from Interaction
---

# Reinforcement Learning: Learning from Interaction

This lecture introduces reinforcement learning (rl). Reinforcement learning addresses the problem of making sequential decisions under uncertainty. An agent learns through interaction with an environment, aiming to maximize a cumulative reward signal. This contrasts with supervised learning, which learns from labeled data, and unsupervised learning, which identifies patterns in unlabeled data. Reinforcement learning is relevant to robotics, game playing, and the alignment of large artificial intelligence models.

## 1. The Reinforcement Learning Problem

Reinforcement learning involves two primary entities: the agent and the environment. These interact over a sequence of discrete time steps. At each time step $t$, the agent observes the environment's state $s_t \in S$. Based on this state, the agent selects an action $a_t \in A$. The environment responds by providing a reward $r_{t+1} \in \mathbb{R}$ and transitioning to a new state $s_{t+1} \in S$. The agent's objective is to learn a behavior that maximizes the total reward accumulated over this sequence of interactions. This interaction loop formalizes the concept of learning through trial and error.

![Figure 1: Agent-Environment Loop](figures/agent_env_loop.png)
*figure 1: the basic reinforcement learning interaction loop between agent and environment. [murphy, fig 1.1: 'a small agent interacting with a big external world.']*


## 2. Formalizing the Environment: Markov Decision Processes (MDPs)

We often model the interaction between the agent and environment using a Markov Decision Process (MDP). An MDP provides a mathematical framework for modeling sequential decision making in situations where outcomes are partly random and partly under the control of the agent. The MDP is formally defined by a tuple $(S, A, P, R, \gamma)$.

The first component, $S$, is the set of all possible states the environment can be in. The second component, $A$, is the set of all possible actions the agent can take. The third component, $P$, is the state transition probability function. It defines the dynamics of the environment, specifying the probability of transitioning to state $s'$ given that the agent takes action $a$ in state $s$: $P(s'\|s,a) = Pr(s\_{t+1}=s' \| s\_t=s, a\_t=a)$. The fourth component, $R$, is the reward function. It specifies the expected reward received after transitioning from state $s$ to state $s'$ by taking action $a$. We denote the reward received at time $t+1$ as $r\_{t+1}$, and its expectation can be written as $R(s,a,s') = E[ r\_{t+1} \| s\_t=s, a\_t=a, s\_{t+1}=s'] $. The final component, $\gamma$, is the discount factor, which we will discuss later. MDPs provide a standard way to frame problems for reinforcement learning.

A key assumption in MDPs is the Markov property. This property states that the future is independent of the past, given the present state. Mathematically, the probability of the next state $s\_{t+1}$ depends only on the current state $s\_t$ and action $a\_t$, not on the entire history of states and actions that came before: $Pr(s\_{t+1}\|s\_t, a\_t, s\_{t-1}, a\_{t-1}, ..., s\_0, a\_0) = Pr(s\_{t+1}\|s\_t, a\_t)$. This means the current state $s\_t$ encapsulates all information from the history relevant for predicting the future. This assumption significantly simplifies the decision-making process.

The agent's behavior within an MDP is defined by its policy, $\pi$. A policy specifies which action the agent will take in a given state. A stochastic policy, denoted $\pi(a\|s)$, gives the probability of taking action $a$ when in state $s$: $\pi(a\|s) = Pr(a_t=a \| s_t=s)$. A deterministic policy is a special case where the policy outputs a single action for each state. The goal in reinforcement learning is typically to find an optimal policy, denoted $\pi^*$, which maximizes the expected cumulative reward.

![Figure 2: MDP Example](figures/mdp_fsm.png)
*figure 2: illustration of a markov decision process (mdp) as a finite state machine. circles are states, arrows represent transitions based on actions (not explicitly labeled here), with associated probabilities and rewards. [murphy, fig 1.3: 'illustration of an mdp as a finite state machine... numbers on the black edges represent state transition probabilities... numbers on the yellow wiggly edges represent expected rewards...']*


## 3. Goals and Objectives

The primary goal for a reinforcement learning agent is to maximize the cumulative reward it receives over time. We formalize this goal using the concept of the return.

The return at time step $t$, denoted $G_t$, is the sum of discounted rewards received from that time step forward. It is defined as:

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots = \sum_{k=0}^\infty \gamma^k r_{t+k+1}
$$

This equation sums the rewards the agent receives in the future, starting from the reward $r\_{t+1}$ obtained after taking action $a\_t$ in state $s\_t$.

The discount factor $\gamma$ is a value between 0 and 1 (i.e., $\gamma \in [0, 1)$). It serves two main purposes. First, for tasks that could continue indefinitely (infinite horizons), the discount factor ensures that the infinite sum defining the return $G_t$ converges, provided the rewards are bounded. Second, $\gamma$ determines the present value of future rewards. A reward received $k$ time steps in the future is only worth $\gamma^k$ times what it would be worth if received immediately. If $\gamma$ is close to 0, the agent prioritizes immediate rewards, acting "myopically". If $\gamma$ is close to 1, the agent values future rewards more strongly, acting "far-sightedly". The discount factor can also be interpreted as the probability of the interaction continuing to the next step in tasks with uncertain duration (Ref: Murphy Ch 1.1.3).

The overall objective for the agent in an MDP is to select actions according to a policy $\pi$ that maximizes the expected return from the initial state(s) $s\_0$. The agent aims to find a policy $\pi^* $ such that $ E\_{\pi^* }[G_0 \| s_0]$ is maximized over all possible policies.


## 4. Value Functions and Bellman Equations

To find the optimal policy, reinforcement learning algorithms often rely on estimating the "value" of being in a particular state or taking a particular action in a state. These estimates are captured by value functions.

The state-value function for a policy $\pi$, denoted $V^\pi(s)$, quantifies the expected return when starting in state $s$ and subsequently following policy $\pi$. Formally:

$$
V^\pi(s) = E_\pi[G_t | s_t=s] = E_\pi\left[\sum_{k=0}^\infty \gamma^k r_{t+k+1} | s_t=s\right]
$$

This function tells us how good it is for the agent to be in state $s$ while operating under policy $\pi$.

The action-value function for a policy $\pi$, denoted $Q^\pi(s,a)$, gives the expected return after taking action $a$ in state $s$ and then following policy $\pi$ thereafter. Formally:

$$
Q^\pi(s,a) = E_\pi[G_t | s_t=s, a_t=a] = E_\pi\left[\sum_{k=0}^\infty \gamma^k r_{t+k+1} | s_t=s, a_t=a\right]
$$

This function tells us how good it is to take a specific action $a$ from state $s$, assuming the agent follows policy $\pi$ afterwards. $Q^\pi(s,a)$ is often called the Q-function.

Value functions for a policy $\pi$ satisfy recursive relationships known as the Bellman expectation equations. These equations express the value of a state or state-action pair in terms of the expected values of successor states or state-action pairs. The relationship between $V^\pi$ and $Q^\pi$ is:

$$
\begin{aligned}
V^\pi(s) &= \sum_a \pi(a|s) Q^\pi(s,a)\\ 
Q^\pi(s,a) &= \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')]
\end{aligned}
$$

Substituting these into each other yields recursive equations solely in terms of $V^\pi$ or $Q^\pi$:

$$
\begin{aligned}
V^\pi(s) &= \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')]\\
Q^\pi(s,a) &= \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')]
\end{aligned}
$$

These equations form the basis for many policy evaluation algorithms.

The goal of reinforcement learning is often to find an optimal policy $\pi^* $ that achieves the highest possible expected return from all states. The corresponding optimal value functions are denoted $V^* (s) $ and $Q^* (s,a) $. They are defined as $V^* (s) = \max_\pi V^\pi(s)$ and $Q^* (s,a) = \max_\pi Q^\pi(s,a)$. These represent the best possible performance in the MDP.

The optimal value functions satisfy a special set of Bellman equations called the Bellman optimality equations. These express the fact that the value of a state under an optimal policy must equal the expected return for the best action from that state:

$$
\begin{aligned}
    V^*(s) &= \max_a Q^*(s,a)\\
    Q^*(s,a) &= \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')]
\end{aligned}
$$

Combining these yields the recursive forms:

$$
\begin{aligned}
V^*(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')]\\
Q^*(s,a) = \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma \max_{a'} Q^*(s',a')]
\end{aligned}
$$

These equations relate the optimal value of a state or action to the optimal values of its successors.

![Figure 3: Planning Backups](figures/planning_backups.png)
*figure 3: backup diagrams illustrating the difference between policy evaluation (for a fixed policy π, using expectation over actions) and value iteration (finding the optimal value v* by maximizing over actions). [murphy, fig 2.2: 'policy iteration vs value iteration represented as backup diagrams...']*

If we know the optimal action-value function $Q^* $, we can determine an optimal policy $\pi^* $. The optimal policy simply selects the action that maximizes $Q^* (s,a) $ in each state $s$:

$$
\pi^* (s) = \operatorname{argmax}_a Q^* (s,a)
$$

If multiple actions achieve the maximum value, any of these can be chosen. If only the optimal state-value function $V^* $ is known, determining the optimal action requires knowledge of the environment's dynamics ($P$ and $R$) to perform a one-step lookahead calculation for each action. Finding $Q^* $ or $V^* $ is central to solving many RL problems.

![Figure 4: Grid World Q-Values](figures/grid_world_qvals.png)
*figure 4: optimal q-values (q*) for a simple 1d grid world task under different discount factors (γ). shows how γ affects the agent's focus on immediate vs future rewards. [murphy, fig 2.1: 'left: illustration of a simple mdp... right: optimal q-functions for different values of γ.']*


## 5. Planning vs. Learning

There are two main settings for solving MDPs, differing in whether the model of the environment is known.

In the planning setting, the agent is assumed to have full knowledge of the MDP dynamics $P(s'\|s,a)$ and reward function $R(s,a,s')$. With a known model, algorithms like value iteration or policy iteration can be used to compute the optimal value function ($V^* $ or $Q^* $) and thus the optimal policy $\pi^* $ directly, often by iteratively applying the Bellman equations. These methods essentially involve computation without further interaction with the environment (Ref: Murphy Ch 2.2).

In the reinforcement learning setting, the agent does not know the MDP dynamics $P$ or the reward function $R$ beforehand. Instead, the agent must learn how to behave by interacting with the environment. It observes sequences of states, actions, and rewards, $(s\_t, a\_t, r\_{t+1}, s\_{t+1})$, generated by following some policy. The challenge is to use this sampled experience to find a good or optimal policy (Ref: Murphy Ch 2.3). This lecture focuses on the reinforcement learning setting where the model is unknown.




## 6. Introduction to RL Algorithms

The central challenge in reinforcement learning is learning good policies using only sampled experience. The agent observes transitions $(s\_t, a\_t, r\_{t+1}, s\_{t+1})$ but does not have access to the underlying transition probabilities $P(s'\|s,a)$ or the reward function $R(s,a,s')$. We introduce the main families of algorithms designed to solve this problem.

![Figure 5: Learning Backups](figures/learning_backups.png)
*figure 5: backup diagrams comparing monte carlo (mc), temporal difference (td), and dynamic programming (dp) updates for estimating state values. mc uses full returns from complete episodes, td bootstraps using the estimated value of the next state, dp uses the full model. [murphy, fig 2.3: 'backup diagrams of v(st) for monte carlo, temporal difference, and dynamic programming updates...']*

**6.1 Value-Based RL: Q-Learning**

Value-based reinforcement learning methods focus on estimating the optimal value functions, typically the optimal action-value function $Q^*(s,a)$. Once a good estimate of $Q^*$ is obtained, the policy can be derived by selecting actions greedily with respect to the estimated Q-values. A prominent algorithm in this family is Q-learning.

Q-learning learns the Q-function iteratively. After observing a transition $(s\_t, a\_t, r\_{t+1}, s\_{t+1})$, the Q-value estimate $Q(s\_t, a\_t)$ is updated using the following rule:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]
$$

Here, $\alpha$ is the learning rate, which controls the step size of the update. The term $[r\_{t+1} + \gamma \max\_{a'} Q(s\_{t+1}, a')]$ represents the target value, based on the observed reward $r\_{t+1}$ and the current estimate of the maximum Q-value achievable from the next state $s\_{t+1}$. The difference between the target and the current estimate $Q(s\_t, a\_t)$ is the temporal difference (TD) error. This update rule aims to make the Q-function estimates satisfy the Bellman optimality equation over time.

A key property of Q-learning is that it is an off-policy algorithm. This means it can learn the optimal value function $Q^*$ even if the data $(s\_t, a\_t, r\_{t+1}, s\_{t+1})$ is generated by following a policy different from the one derived from the current Q-function estimate (e.g., an exploratory policy like epsilon-greedy). This allows Q-learning to learn from past experience or demonstrations.

![Figure 6: Q-learning Example Trace](figures/qlearning_trace.png)
*figure 6: example q-learning updates over one trajectory in the 1d grid world. shows how q-values propagate based on experienced transitions and rewards using the td update rule. [murphy, fig 2.5: 'illustration of q learning for one random trajectory in the 1d grid world...']*

**6.2 Policy-Based RL: Policy Gradients**

Policy-based reinforcement learning methods learn a parameterized policy $\pi_\theta(a\|s)$ directly. The goal is to adjust the policy parameters $\theta$ to maximize the expected return, $J(\theta) = E_{\tau \sim \pi\_\theta}[G\_0]$, where $G\_0$ is the return from the start of an episode.

Policy gradient methods use gradient ascent to optimize $J(\theta)$. The core idea is to calculate the gradient $\nabla_\theta J(\theta)$ and update the parameters in that direction: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$. The policy gradient theorem provides an expression for this gradient. Let $P(\tau\|\theta)$ be the probability of a trajectory $\tau=(s\_0, a\_0, r\_1, s\_1, ...)$ under policy $\pi\_\theta$, and let $R(\tau) = \sum\_{t=0}^{T-1} \gamma^t r\_{t+1}$ be the total discounted reward for that trajectory (or $G\_0$). The objective is $J(\theta) = \sum\_\tau P(\tau\|\theta) R(\tau)$. The gradient is:

$$
\begin{aligned}
\nabla_\theta J(\theta) &= \nabla_\theta \sum_\tau P(\tau|\theta) R(\tau) \\
&= \sum_\tau (\nabla_\theta P(\tau|\theta)) R(\tau) \\
&= \sum_\tau P(\tau|\theta) (\frac{\nabla_\theta P(\tau|\theta)}{P(\tau|\theta)}) R(\tau) \\
&= \sum_\tau P(\tau|\theta) (\nabla_\theta \log P(\tau|\theta)) R(\tau)
\end{aligned}
$$

Here we used the identity $\nabla\_x \log f(x) = \frac{\nabla\_x f(x)}{f(x)}$. Since 

$$P(\tau|\theta) = p(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t) P(s_{t+1} |s_t, a_t)$$

its logarithm is 

$$\log P(\tau|\theta) = \log p(s_0) + \sum_{t=0}^{T-1} (\log \pi_\theta(a_t|s_t) + \log P(s_{t+1}|s_t, a_t))$$

The gradient of the log probability of the trajectory with respect to $\theta$ is therefore:

$$
\nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

Substituting this back gives:

$$
\nabla_\theta J(\theta) = \sum_\tau P(\tau |\theta) \left(\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)\right) R(\tau)
$$

This can be written as an expectation:

$$
\nabla_\theta J(\theta) = E_{\tau \sim \pi_\theta} \left[ \left(\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)\right) R(\tau) \right]
$$

The intuition here is to make trajectories $\tau$ with high total reward $R(\tau)$ more likely, by increasing the probabilities $\pi_\theta(a_t|s_t)$ of the actions taken along those trajectories.

The REINFORCE algorithm [Wil92] refines this by noting that the action $a_t$ taken at time $t$ can only influence rewards from $r_{t+1}$ onwards. It therefore replaces the total trajectory reward $R(\tau)$ with the return-from-step-t, $G_t = \sum\_{k=t}^{T-1} \gamma^{k-t} r\_{k+1}$:

$$
\nabla_\theta J(\theta) = E_{\pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) G_t \right]
$$

The REINFORCE update increases the log-probability of each action $a_t$ proportionally to the total future discounted reward $G_t$ obtained after taking it. A major drawback of REINFORCE is that $G_t$ is based on a potentially long sequence of rewards, making its estimate from a single trajectory very noisy (high variance).

To reduce this variance, we can subtract a baseline $b(s_t)$ that depends only on the state $s_t$, not the action $a_t$. The expected value of the subtracted term is zero:

$$
\begin{aligned}
E_{\pi_\theta} [\nabla_\theta \log \pi_\theta(a_t|s_t) b(s_t)] &= \sum_{s} d^\pi(s) \sum_a \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s) b(s) \\
&= \sum_{s} d^\pi(s) b(s) \sum_a \nabla_\theta \pi_\theta(a|s) \\
&= \sum_{s} d^\pi(s) b(s) \nabla_\theta \left(\sum_a \pi_\theta(a|s)\right) \\
&= \sum_{s} d^\pi(s) b(s) \nabla_\theta(1) \\
&= 0
\end{aligned}
$$

where $d^\pi(s)$ is the state distribution under $\pi$. Therefore, subtracting a baseline does not change the expected gradient:

$$
\nabla_\theta J(\theta) = E_{\pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) (G_t - b(s_t)) \right]
$$

The variance is reduced because if we choose $b(s\_t)$ close to the expected return $E[G\_t\|s\_t] = V^\pi(s\_t)$, the term $(G\_t - b(s\_t))$ will be centered around zero and have smaller magnitude than $G\_t$. The state-value function $V^\pi(s\_t)$ is a common and effective choice for the baseline.

**6.3 Actor-Critic Methods**

Actor-Critic (AC) methods implement the baseline idea by explicitly learning an estimate of the value function (the critic) to reduce the variance of the policy gradient estimate (used by the actor).

The critic learns an estimate of the state-value function, $V\_w(s) \approx V^\pi(s)$, typically using TD learning as described earlier. The actor is the policy $\pi\_\theta(a\|s)$. The advantage function $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ measures the relative value of taking action $a$ compared to the average action under $\pi$ in state $s$. The policy gradient can be expressed using the advantage function:

$$
\nabla_\theta J(\theta) = E_{\pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) A^\pi(s_t, a_t) \right]
$$

This follows directly from the baseline result by setting $b(s\_t) = V^\pi(s\_t)$ and noting that $E[G\_t \| s\_t, a\_t] = Q^\pi(s\_t, a\_t)$.

In actor-critic methods, we need to estimate the advantage $A^\pi(s\_t, a\_t)$ using the learned critic $V\_w$. We know $A^\pi(s\_t, a\_t) = E[r\_{t+1} + \gamma V^\pi(s\_{t+1}) \| s\_t, a\_t] - V^\pi(s\_t)$. We can approximate this expectation using a single sample $(r\_{t+1}, s\_{t+1})$ and replacing the true $V^\pi$ with the critic's estimate $V\_w$:

$$
A^\pi(s_t, a_t) \approx (r_{t+1} + \gamma V_w(s_{t+1})) - V_w(s_t) = \delta_t
$$

This shows that the TD error $\delta_t$ serves as a (biased but lower-variance) estimate of the advantage function.

This leads to the common online actor-critic update rule, performed at each step $t$ after observing $(s\_t, a\_t, r\_{t+1}, s\_{t+1})$:
1.  Compute the TD error (advantage estimate): $\delta\_t = r\_{t+1} + \gamma V\_w(s\_{t+1}) - V\_w(s\_t)$ (where $V\_w(s\_{terminal}) = 0$).
2.  Update the critic parameters $w$: $w \leftarrow w + \alpha\_w \delta\_t \nabla\_w V\_w(s\_t)$.
3.  Update the actor parameters $\theta$: $\theta \leftarrow \theta + \alpha\_\theta \delta\_t \nabla\_\theta \log \pi\_\theta(a\_t\|s\_t)$.

This is a stochastic gradient ascent update. The actor update modifies policy parameters based on the estimated advantage $\delta_t$ of the action $a_t$ actually taken. Unlike REINFORCE which waits until the end of an episode and sums over all steps using the high-variance $G_t$, this update happens at each step using the lower-variance TD error $\delta_t$.

## 7. Policy Optimization Algorithms: Stability and Direct Preference Methods

The policy gradient methods discussed in Section 6 provide a way to update policy parameters $\theta$ to increase the expected return $J(\theta)$. The fundamental update relies on estimating the gradient:

$$
\nabla_\theta J(\theta) = E_{\pi_\theta}\left[\sum_{t=0}^\infty \nabla_\theta \log \pi_\theta(a_t|s_t) \Psi_t\right]
$$

However, directly applying this gradient, especially with function approximators like neural networks and noisy estimates of $\Psi\_t$ (like the Monte Carlo return $G\_t$ or TD error $\hat{A}\_t$), can lead to problems. Large estimated gradients can cause large updates to $\theta$, drastically changing the policy $\pi_\theta$. This might move the policy into regions of poor performance or cause the learning process to become numerically unstable and diverge. Furthermore, the high variance associated with estimators like $G\_t$ can make learning slow and require many samples.

This motivates the development of algorithms that ensure more stable and reliable policy improvement. These methods often aim to control the size or effect of the policy update at each step, effectively keeping the new policy "close" to the policy used to collect the data. This concept is analogous to trust region or proximal point methods in general optimization, which constrain parameter updates to ensure stability. The algorithms below implement related ideas within the context of policy learning, often using statistical divergences as a measure of policy change.

### 7.1 Proximal Policy Optimization (PPO)

Proximal Policy Optimization (PPO) is an algorithm designed to achieve stable policy updates using relatively simple modifications to the policy gradient objective [Sch+17].

PPO typically operates on data collected by a previous version of the policy, $\pi\_{\theta\_{old}}$. To estimate the objective under the current policy $\pi\_\theta$, it uses importance sampling. We define the probability ratio:

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

This ratio re-weights the estimated advantage $\hat{A}\_t$ (calculated using data from $\pi\_{\theta\_{old}}$) to estimate the performance of $\pi\_\theta$. The basic importance-sampled objective is $L^{IS}(\theta) = \hat{E}\_t [ \rho\_t(\theta) \hat{A}\_t ]$. Directly maximizing $L^{IS}(\theta)$ via gradient ascent can suffer from high variance and instability. If $\rho\_t(\theta)$ becomes very large (i.e., $\pi\_\theta$ drastically increases the probability of action $a\_t$ compared to $\pi\_{\theta\_{old}}$), it can overly amplify the corresponding advantage estimate $\hat{A}\_t$ (which might itself be noisy). This leads to large variance in the gradient estimate $\nabla\_\theta L\^{IS}(\theta)$, causing potentially damaging parameter updates.

PPO mitigates this by using a *clipped surrogate objective*:

$$
L^{CLIP}(\theta) = \hat{E}_t [ \min( \rho_t(\theta) \hat{A}_t , \operatorname{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t ) ]
$$

Here, $\operatorname{clip}(x, l, u)$ restricts $x$ to be within the range $[l, u]$, and $\epsilon$ is a small hyperparameter (e.g., 0.2). The $\min$ operator selects the smaller of the unclipped importance-sampled advantage and a version where the ratio $\rho\_t(\theta)$ is clipped.

The clipping mechanism works as follows:
1.  If $\hat{A}\_t > 0$: The objective increases the probability of action $a\_t$. Clipping $\rho\_t(\theta)$ at $1+\epsilon$ prevents this increase from being excessively large if the action becomes much more likely under $\pi\_\theta$.
2.  If $\hat{A}\_t < 0$: The objective decreases the probability of action $a\_t$. Clipping $\rho\_t(\theta)$ at $1-\epsilon$ prevents this decrease from being excessively large if the action becomes much less likely under $\pi\_\theta$.

By taking the minimum, PPO ensures that the update derived from maximizing $L\^{CLIP}(\theta)$ does not benefit from excessively large probability ratios, effectively keeping the policy change within a bounded region around $\pi\_{\theta\_{old}}$. This improves stability compared to the unclipped objective. PPO typically maximizes this objective using stochastic gradient ascent over multiple epochs on the same batch of data.

### 7.2 Group Relative Policy Optimization (GRPO)

Group Relative Policy Optimization (GRPO) is an algorithm often used in settings where multiple completions are generated for the same prompt, common in language model alignment [Sha+24, Dee25]. It adapts the policy gradient framework by modifying the advantage estimation.

Instead of learning a state-value function $V\_w(s)$ as a baseline, GRPO uses a baseline derived from the rewards of other completions within the same group (i.e., generated from the same prompt $s$ in the current batch). For a group of $G$ completions $\{y\_1, ..., y\_G\}$ with corresponding rewards $\{r\_1, ..., r\_G\}$, the group relative advantage for completion $y\_i$ is estimated as:

$$
\hat{A}_i = r_i - \bar{r} \quad \text{where} \quad \bar{r} = \frac{1}{G} \sum_{j=1}^G r_j
$$

This advantage measures how completion $i$ performed relative to the average performance within its group for that specific prompt. The intuition is that $\bar{r}$ serves as an empirical, prompt-specific baseline. Recall the standard advantage $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$. Here, $r\_i$ acts as a Monte Carlo estimate of the value of completion $y\_i$ (i.e., $Q(s, y\_i)$), and the group average $\bar{r}$ acts as an empirical estimate of the expected value of the prompt $s$ under the current policy ($V(s)$), based on the samples generated in the batch. Thus, $\hat{A}\_i = r\_i - \bar{r}$ provides a sample-based, local estimate of the advantage of completion $y\_i$ relative to the average completion sampled for prompt $s$. This avoids needing a separate value network, potentially simplifying training for large models.

The GRPO objective combines this relative advantage with a PPO-style update mechanism, often including both clipping and an explicit KL penalty. The objective for a single group is often implemented as a sum over tokens $t$ within each completion $i$:

$$
L^{GRPO}(\theta) = \hat{E}_{s, \{y_i, r_i\}} \left[ \frac{1}{G}\sum_{i=1}^G \sum_t \min( \rho_{i,t}(\theta) \hat{A}_i , \operatorname{clip}(\rho_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i ) \right]
$$

Note that $\hat{A}\_i$ is constant across all tokens $t$ for a given completion $i$. This objective, maximized via gradient ascent, encourages actions within completions that performed better than the group average, while limiting large policy changes via clipping. The explicit KL penalty, if used, adds further regularization:

$$
L^{GRPO\_KL}(\theta) = L^{GRPO}(\theta) - \beta \hat{E}_{s, \{y_i\}} [ D_{KL}(\pi_\theta(\cdot|s) || \pi_{ref}(\cdot|s)) ]
$$

### 7.3 Direct Preference Optimization (DPO)

Direct Preference Optimization (DPO) offers a different approach that learns directly from preference data, without an explicit reward modeling step or a standard RL optimization loop [Raf+23]. It uses a dataset $D = \{(x, y\_w, y\_l)\}$, where $y\_w$ is preferred over $y\_l$ for prompt $x$.

The derivation starts by defining the standard objective for policy optimization in the context of alignment, often referred to as the RLHF objective. This objective seeks to maximize the expected reward according to the (unknown) true preference-based reward function $r^*(x,y)$, while penalizing deviation from a reference policy $\pi\_{ref}$ using KL divergence:

$$
\max_\pi E_{x \sim D, y \sim \pi(y|x)}[r^*(x,y)] - \beta D_{KL}(\pi(\cdot|x) || \pi_{ref}(\cdot|x))
$$

Here $\beta$ is a parameter controlling the strength of the KL penalty. It can be shown that the optimal policy $\pi^*$ solving this objective has a specific form related to the reward function and the reference policy:

$$
\pi^*(y|x) \propto \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r^*(x,y)\right)
$$

Rearranging this gives an expression for the implicit reward function:
$$
r^*(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \text{constant}(x)
$$

DPO connects this to preference data by assuming the probability of preferring $y_w$ over $y_l$ follows the Bradley-Terry model:

$$
P(y_w \succ y_l | x) = \sigma(r^*(x, y_w) - r^*(x, y_l))
$$

where $\sigma$ is the sigmoid function. Substituting the expression for $r^* $ into this model (and noting the constant term cancels), we get the probability of the preference in terms of the optimal policy $\pi^* $ and the reference policy $\pi\_{ref}$:

$$
P(y_w \succ y_l | x) = \sigma\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)}\right)
$$

DPO learns a policy $\pi\_\theta$ by maximizing the likelihood of the observed preference data $D$ under this model, replacing the unknown $\pi^*$ with the current parameterized policy $\pi\_\theta$. This yields the negative log-likelihood loss:

$$
L_{DPO}(\theta; \pi_{ref}) = -E_{(x, y_w, y_l) \sim D} \left[\log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right)\right]
$$

Minimizing this loss increases the likelihood of the preferred completions $y\_w$ and decreases the likelihood of the dispreferred completions $y\_l$, relative to the reference policy. The temperature parameter $\beta$ controls the strength of this update relative to the implicit KL penalty against $\pi\_{ref}$. The gradient effectively performs weighted supervised updates on the log-probabilities of the chosen and rejected sequences.


## 8. Conclusion

This lecture introduced reinforcement learning as a framework for sequential decision making to maximize cumulative reward. We defined Markov Decision Processes (MDPs) and the Bellman equations that characterize value functions ($V^\pi, Q^\pi, V^* , Q^* $). We differentiated planning (model known) from reinforcement learning (model unknown, learning from $(s\_t, a\_t, r\_{t+1}, s\_{t+1})$ samples). Key RL algorithm families were presented: value-based methods like Q-learning estimate optimal values; policy-based methods like REINFORCE optimize policy parameters $\theta$ using policy gradients; actor-critic methods use a learned value function (critic) to improve policy (actor) updates, often via advantage estimation $A\_t$. We introduced algorithms like PPO, GRPO, and DPO which modify policy gradient approaches for stability or learn directly from preferences, relevant for aligning large models.