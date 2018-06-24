# rl-sonic
Reinforcement Learning methods to play Sonic.

## Training Start
![Training Start GIF](https://i.imgur.com/GRyEVXc.gif)

Agent's behavior at the beginning of training. Moves are selected randomly as the
agent explores its environment.

## Training End
![Training End GIF](https://i.imgur.com/5OIGxnl.gifv)

After playing 1000 games, the agent is able to win the level with ease. The agent in this
gif is supported by a Dueling Double DQN. The full runthrough is available on YouTube at:

https://www.youtube.com/watch?v=BO5VcUd2RGQ
## Implementation
I designed and trained the following models to play Sonic. The agent made decisions
solely through the observation of the pixel values in each frame of the game over time.
While the following models represent a diverse collection of reinforcment learning
algorithms, each was trained with the same basic process of iteratively observing the 
environement, taking an action, and learning from the resulting reward, until ultimately
reaching a policy which dictates the optimal action to take in evey situation.

### Deep Q-Network (DQN)
Deep Q-Networks (DQNs) approximate the function **Q(s,a)**, which represents the value of taking 
action **a** in state **s**. In Q learning, the metric used to evaluate an action **a** is the 
expected sum of discounted future rewards acting optimally from state **s**. This idea is formalized by the *Bellman Equation*:

**Q(s,a) = r + y(max(Q(s’, a’))**

Where **r** is the reward observed after taking action **a** in state **s** and **y** is a discount factor.

As an agent explores its environment, it takes actions and observes the resulting rewards. These experiences (initial state, selected action, reward, next state) are then used to update the value associated with the initial state and selected action. Provided sufficient exploration and training, the q values will converge to an approximation of the optimal function mapping state action pairs to the optimal action overtime.

At their most basic, Q-networks simply iteratively update their approximation of the Q-function as each observation is made. Standard DQNs have proven to be very strong solutions to many difficult reinforcement
learning problems. However, the basic DQN can be improved dramatically through various methods which enhance convergence reliability and speed.

The following models are discrete extensions of the basic DQN model. The combination of all extensions into a single agent forms what Google’s *Deepmind* calls a "Rainbow DQN". While each extension generally improves the training efficiency and accuracy of the agent, the Rainbow DQN has been shown to outperform all sub-combinations of the following DQN additions in many Arcade Learning Environments.

#### Double DQN
Q values are updated by minimizing the squared error between Q(s,a) and r + y(max(Q(s’,a’)), the two quantities related by the Bellman equation. However, until the q values converge to an optimal policy, they do not accurately reflect the true value of (state, action) pairs. This poses a serious issue, as the discounted sum of expected rewards used to update the q function is inherently dependent on the current approximation of the q function. The reliance of the q function approximation on itself to perform iterative updates can frequently lead to divergence from the optimal q function approximation.

One method for mitigating this circular dependence issue is to learn two separate networks (hence the name **Double** DQN). The “main” network is used to evaluate state-action pairs **s,a)** and predict the optimal next action **a’** to take from the state **s’** (reached by taking action **a** from state **s**). A second “target” network is used to determine the value of taking the predicted action **a’** based on a “frozen” approximation of the q function. The “main” network is iteratively updated by computing Q(s,a)  and predicting **a’** for observed experiences (s, a, r, s’) with Q(s’,a’) computed by the frozen “target” network. After a set number of training steps, the “target” network is updated to reflect the current “main” network. This two network structure dramatically reduces the circular dependence of the q function on itself, leading to a much more stable training process.

 

The simplest DQN I trained was a Double DQN.

 



This agent can be found in [dqn_agent.py]() and trained with [train.py](). Ensure the correct parameters are set in [parameters.py]().


#### Dueling DQN

Dueling DQNs attempt to gain a deeper understanding of the environment by breaking down the q function **Q(s,a)** which represents the value of taking action **a** in state **s** into two separate functions: **V(s)** - the value of being in state s and **A(s,a)** - the advantage of taking action **a** over taking all other possible actions. Intuitively, the dueling architecture can now separate the value of simply existing in a state V(s) from the value associated with acting in the state A(s,a).


Dueling DQNs learn V(s) and A(s,a) within inner layers, then sum the output of the two layers to yield the q values based on the relation: 

V(s) + A(s,a) = Q(s,a).
 

Remarkably, this structure provides the agent with a much better understanding of the environment than a basic Double DQN, as the dueling agent was able to learn to complete the level within 1000 episodes of training, while the Double DQN was not.

Victory after 1000 Episodes of training

https://www.youtube.com/watch?v=BO5VcUd2RGQ

#### Prioritized Experience Replay (PER)
Another common method improving the stability of DQN training is to separate the exploration from updating the q function. This is accomplished by storing experiences in memory, then periodically training the DQN with a batch of stochastically sampled experiences.


Prioritized Experience Replay takes this process a step further by storing experiences with a *priority* which reflects its potential influence on the q function. The priorities are used to store experiences in a priority queue so that the probability of sampling an experience is based on its influence.

 
#### Noisy DQN
One of the biggest challenges in all reinforcement learning is balancing exploration (acting randomly) with acting according to the current policy. Exploration is necessary for the agent to understand the potential value of taking all actions in a given state. However, once the agent has a correct understanding of the environment randomness is no longer desirable.

 
The general solution to this issue is to define a quantity *epsilon* which dictates how often the agent chooses a random action. *epsilon* decays over a set number of training steps so that randomness is incrementally discouraged as training progresses.


An alternative approach to encouraging randomness in the correct situations is to use a “NoisyNet”. NoisyNets are a weighted noise layer that is applied to the output of a network to potentially add randomness to output values. The weights on the noise provide the agent with far more flexibility than a simple **epsilon** value, as the agent can learn to minimize noise when the correct action is clear without discouraging randomness in new situations.

#### N-step DQN
This extension improves the temporal awareness of the agent by storing experiences as 
**<st, a, R, st+n, done>** tuples, where **st** is the state at timestep **t** and **a** is the action selected (the same values stored in experience/ prioritized experience replay), **R** is the 
discounted sum of rewards over the next **n** states (hence the name n-step), and **st+n** is the nth state following **st**.


#### Distributional DQN

The Distributional DQN attempts to learn a distribution of future rewards associated with each action in a given state **Z(s,a)** instead of a single *expected* value **Q(s,a)**. The argument on behalf of this approach is that the expected value of a (state, action) pair is never realizable, as it is the average of many possible outcomes. A distribution on the other hand provides a broader view of the potential future rewards.


Fortunately, the Bellman equation still holds true for distributions (called the Distributional Bellman) and thus Distributional DQNs can be iteratively updated in the same was as q value based DQNs.

#### Rainbow DQN

#### Resources
[Rainbow DQN arvix paper](https://arxiv.org/abs/1710.02298)


### Advantage Actor Critic (A2C)

### Just Enough Retained Knowledge (JERK)

### Proximal Proximity Opperator (PPO)

