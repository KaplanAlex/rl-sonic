# rl-sonic
Reinforcement Learning methods to play Sonic.

## Training Start
![Training Start GIF](https://i.imgur.com/GRyEVXc.gif)

Agent's behavior at the beginning of training.

## Models
I designed and trained the following models to play Sonic. The agent makes decisions
solely through the observation of the pixel values in each frame as the game progresses.
While the following models represent a diverse collection of reinforcment learning
algorithms, each was trained with the same basic process of iteratively observing the 
environement, taking an action, and learning from the resulting reward until ultimately
reaching a policy which dictates the optimal action is take in evey situation.

### Deep Q-Network (DQN)
Deep Q-Networks (abbreviated as DQNs) approximate the function, Q(s,a), which maps 
(state, action) pairs to the expected value of taking the given action in the given state.


The following models are extensions are the basic DQN model. The combination
of all discrete extentions into one agent forms a "Rainbow DQN". While each extension
generally (and in many cases substantially) improves the training efficiency and accuracy
of the agent, the Rainbow DQN has been shown to outperform all subcombinations of
the following DQN additions. 


##### Rainbow DQN
The Rainbow DQN is a collection of discrete extensions to the basic DQN model . 

#### Double DQN

#### Dueling DQN

#### Prioritized Experience Replay (PER)

#### Noisy DQN

#### N-step DQN
This extension improves the temporal awareness of the agent by storing experiences as 
**<st, a, R, st+n, done>**, where **st** is the state at timestep **t** and **a** is the action selected 
(the same values stored in experience/ prioritized experience replay), **R** is the discounted sum of 
rewards over the next **n** states (hence the name n-step), and **st+n** is the nth state following **st**.


#### Distributional DQN

#### Resources
[Rainbow DQN arvix paper](https://arxiv.org/abs/1710.02298)


### A2C

### JERK

### PPO