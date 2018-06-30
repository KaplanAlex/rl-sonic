# rl-sonic
Python, Keras implementation of a "Rainbow" DQN used to train an agent to play Sonic.

## Training Start
![Training Start GIF](https://i.imgur.com/GRyEVXc.gif)

Agent's behavior at the beginning of training. Moves are selected randomly as the
agent explores its environment.

## Training End
![Training End GIF](https://i.imgur.com/iN9KqpS.gif)

After playing 1000 games, the agent is able to win the level with ease. The agent in this
gif is supported by a Dueling Double DQN. The full run-through can be found on YouTube at: https://www.youtube.com/watch?v=BO5VcUd2RGQ

## Project Summary
I designed a Deep Q Network (DQN) in Python with Keras backed by Tensorflow, then
implemented the six extensions included in Google *Deepmind*'s "Rainbow" DQN. The
"Rainbow" DQN is built on the following DQN enhancements:

- Double DQN
- Dueling DQN
- Prioritized Experience Replay
- Noisy DQN
- N-step DQN
- Distributional DQN

After implementing all the pieces which make up the Rainbow DQN, I trained a series of agents 
backed by different combinations of these extensions to play a level of *Sonic The Hedgehog 2*.

The agents supported by these models made decisions solely through the observation of the 
pixel values in an input frame. While the following models represent a diverse collection 
of DQN improvements, each was trained with the same basic process of  iteratively observing 
the  environement, taking an action, and learning from the resulting reward. All models were 
trained over 1000 Episodes of gameplay, where each episode ended after:
1. 4500 timesteps elapsed (With a timestep taking 1/15th of a second and the observation of 4 frames at 60Hz).
2. The agent lost the game.
3. All lives were lost.

Various combinations of extensions can be trained by running [train_dqn.py](https://github.com/KaplanAlex/rl-sonic/blob/master/dqn/train_dqn.py) or [train_n-step_dqn.py](https://github.com/KaplanAlex/rl-sonic/blob/master/dqn/train_n-step_dqn.py) and setting values in [parameters.py](https://github.com/KaplanAlex/rl-sonic/blob/master/dqn/parameters.py) to reflect the inclusion of the desired improvements. These training scripts implement one of [dqn_agent.py](https://github.com/KaplanAlex/rl-sonic/blob/master/dqn/dqn_agent.py), [dqn_PER_agent.py](https://github.com/KaplanAlex/rl-sonic/blob/master/dqn/dqn_PER_agent.py), and [dqn_distributional_agent.py](https://github.com/KaplanAlex/rl-sonic/blob/master/dqn/dqn_distributional_agent.py) backed by one network from [networks.py](https://github.com/KaplanAlex/rl-sonic/blob/master/dqn/networks.py), as specified in [parameters.py](https://github.com/KaplanAlex/rl-sonic/blob/master/dqn/parameters.py). The complete "Rainbow" agent is located in [rainbow_agent.py](https://github.com/KaplanAlex/rl-sonic/blob/master/dqn/rainbow_agent.py) and can be trained simply by running the script [train_trainbow.py](https://github.com/KaplanAlex/rl-sonic/blob/master/dqn/train_rainbow.py).

## Implementation and Theory
Explantion of the theory supporting the models I implemented and trained.

### Network Design
All of the DQNs I created had the same base structure, which included three convolutional layers
with ReLu activation functions separated by two MaxPooling layers. The final convolutional layer
was flattened and fed into a series of fully connected (dense) ReLu activated layers,
which terminated in a Dense layer with Linear activation and 8 output nodes (one for each action).


The Dueling, Noisy, and Distributional extensions required modifying this structure to include
isolated network paths, weighted noise, and multi-dimensional output respectively. All of these 
extensions are touched on in the following section and are explained within the [networks.py](https://github.com/KaplanAlex/rl-sonic/blob/master/dqn/networks.py) file. 

### Deep Q-Network (DQN): Building the Rainbow
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

One method for mitigating this circular dependence is to learn two separate networks (hence the name **Double** DQN). The “main” network is used to evaluate state-action pairs **s,a)** and predict the optimal next action **a’** to take from the state **s’** (reached by taking action **a** from state **s**). A second “target” network is used to determine the value of taking the predicted action **a’** based on a “frozen” approximation of the q function. The “main” network is iteratively updated by computing Q(s,a) and predicting **a’** for observed experiences (s, a, r, s’) with Q(s’,a’) computed by the frozen “target” network. After a set number of training steps, the “target” network is updated to reflect the current “main” network. This two network structure dramatically reduces the circular dependence of the q function, leading to a much more stable training process.


The simplest DQN I trained was a Double DQN, thus all extensions were applied to the Double DQN.
This agent can be found in [dqn_agent.py](https://github.com/KaplanAlex/rl-sonic/blob/master/dqn/dqn_agent.py) and trained with [train.py](https://github.com/KaplanAlex/rl-sonic/blob/master/dqn/train_dqn.py) after ensuring the correct parameters are set in [parameters.py](https://github.com/KaplanAlex/rl-sonic/blob/master/dqn/parameters.py).


#### Dueling DQN
Dueling DQNs attempt to gain a deeper understanding of the environment by breaking down the q function **Q(s,a)** which represents the value of taking action **a** in state **s** into two separate functions: **V(s)** - the value of being in state s and **A(s,a)** - the advantage of taking action **a** over taking all other possible actions. Intuitively, the dueling architecture can  separate the value of simply existing in a 
state V(s) from the value associated with acting in the state A(s,a).

Dueling DQNs learn V(s) and A(s,a) within their inner layers, then sum the output of the two layers to 
yield the q values based on the relation: 

V(s) + A(s,a) = Q(s,a).

Thus, the dueling DQN builds on the Double DQN structure by adding two isolated paths through the network
which connect at the output layer. The dueling DQN can be trained simply by setting the network backing
the standard DQN agent ([dqn_agent.py](https://github.com/KaplanAlex/rl-sonic/blob/master/dqn/dqn_agent.py))
to "dueling_dqn" which can be found in [networks.py](https://github.com/KaplanAlex/rl-sonic/blob/master/dqn/networks.py).

Remarkably, this structure provides the agent with a much better understanding of the environment than a basic Double DQN, as the dueling agent was able to learn to complete the level within 1000 episodes of training, while the Double DQN was not.

Victory after 1000 Episodes of training: https://www.youtube.com/watch?v=BO5VcUd2RGQ


#### Prioritized Experience Replay (PER)
Another common method improving the stability of DQN training is to separate the exploration from updating the q function. This is accomplished by storing experiences in memory, then periodically training the DQN with a batch of stochastically sampled experiences.

Prioritized Experience Replay takes this process a step further by storing experiences with a *priority* which reflects its potential influence on the q function. The priorities are used to store experiences in a priority queue so that the probability of sampling an experience is based on its influence. At a high level, the potential power of this improvement is very apparent, as it encourages the use of significant memories in updates, expediting the training process.

To keep these six improvements isolated I created a second dqn agent [dqn_PER_agent.py](https://github.com/KaplanAlex/rl-sonic/blob/master/dqn/dqn_PER_agent.py) with PER, as experience replay is directly incorporated into the agent.
 
#### Noisy DQN
One of the biggest challenges in all reinforcement learning is balancing exploration (acting randomly) with acting according to the current policy. Exploration is necessary for the agent to understand the potential value of taking all actions in a given state. However, once the agent has a correct understanding of the environment, randomness is no longer desirable.

The general solution to this issue is to define a quantity *epsilon* which dictates how often the agent chooses a random action. *epsilon* decays over a set number of training steps so that randomness is incrementally discouraged as training progresses.

An alternative approach is to encourage randomness in the correct situations through the use of a “NoisyNet”. NoisyNets are a weighted noise layer that is applied to the output of a network to potentially add randomness to output values. The weights on the noise provide the agent with far more flexibility than a simple **epsilon** value, as the agent can learn to minimize noise when the correct action is clear without discouraging randomness in new situations.

I created versions of the standard DQN and dueling DQN which included a weighted noise layer preceeding the output, which can both be found in [networks.py](https://github.com/KaplanAlex/rl-sonic/blob/master/dqn/networks.py). I then used the standard agent [dqn_agent.py](https://github.com/KaplanAlex/rl-sonic/blob/master/dqn/dqn_agent.py) to test the effectiveness of the NoisyNet, as shown in the results section.

#### N-step DQN
N-step DQN aims to improve the temporal awareness of the agent by storing experiences as 
**<st, a, R, st+n, done>** tuples, where **st** is the state at timestep **t** and **a** is the action selected (the same values stored in experience/ prioritized experience replay), **R** is the 
discounted sum of rewards over the next **n** states (hence the name n-step), and **st+n** is the nth state following **st**.

Intuitively, this change provides the DQN with a broader understanding of its environment as a whole and specifically the impact of its actions. The goal of N-step DQN is to learn sequences of actions which ultimately lead to a positive reward despite potential negative rewards along the way.

The implementation of N-step DQN required modifying when observed experiences are stored, but did not require changing how experiences are used. Thus, the incorporation of N-step DQN into the agent occurred completely in the training process. The N-step DQN training logic is located in [train_n-step_dqn.py](https://github.com/KaplanAlex/rl-sonic/blob/master/dqn/train_n-step_dqn.py)


#### Distributional DQN

The Distributional DQN attempts to learn a distribution of future rewards associated with each action in a given state **Z(s,a)** instead of a single *expected* value **Q(s,a)**. The argument on behalf of this approach is that the expected value of a (state, action) pair is never realizable, as it is the average of many possible outcomes. A distribution on the other hand provides a broader view of the potential future rewards.

Fortunately, the Bellman equation still holds true for distributions (called the Distributional Bellman) and thus Distributional DQNs can be iteratively updated in the same was as q value based DQNs.

[dqn_distributional_agent.py](https://github.com/KaplanAlex/rl-sonic/blob/master/dqn/dqn_distributional_agent.py) includes the modifications necessary to train and act from a set of distributions rather than discrete values.

## Resources
### OpenAi Retro Contest
- [Contest](https://contest.openai.com/)
- [Environment Overview](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/retro-contest/gotta_learn_fast_report.pdf)

### Rainbow DQN
- [Rainbow DQN arvix paper](https://arxiv.org/abs/1710.02298)
- [ALE Deep Learning Paper](https://arxiv.org/pdf/1312.5602.pdf)
- [Double DQN Paper](https://arxiv.org/abs/1509.06461.pdf)
- [Dueling DQN Paper](https://arxiv.org/abs/1511.06581.pdf) 
- [Prioritized Experience Replay Paper](https://arxiv.org/abs/1511.05952.pdf)
- [NoisyNet Paper](https://arxiv.org/abs/1706.10295.pdf)
- [N-step DQN Paper](https://arxiv.org/pdf/1801.01968.pdf)
- [Distributional DQN Paper](https://arxiv.org/pdf/1707.06887.pdf)