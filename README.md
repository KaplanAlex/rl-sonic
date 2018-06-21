# rl-sonic
Reinforcement Learning methods to play Sonic.

## Training Start
![Training Start GIF](https://i.imgur.com/GRyEVXc.gif)

Agent's behavior at the beginning of training.

## Models

### Basic DQN

### Rainbow DQN
The Rainbow DQN is a collection of discrete extensions to the basic DQN model which 
dramatically improves training efficiency and accuracy. 

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