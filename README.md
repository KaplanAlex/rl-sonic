# rl-sonic
Reinforcement Learning methods to play Sonic.

## Training Start
![Training Start GIF](https://i.imgur.com/GRyEVXc.gif)

Agent's behavior at the beginning of training. Moves are selected randomly as the
agent explores its environment.

## Training End


After playing 1000 games, the agent is able to win the level with easy.  

## Implementation
I designed and trained the following models to play Sonic. The agent made decisions
solely through the observation of the pixel values in each frame of the game over time.
While the following models represent a diverse collection of reinforcment learning
algorithms, each was trained with the same basic process of iteratively observing the 
environement, taking an action, and learning from the resulting reward, until ultimately
reaching a policy which dictates the optimal action to take in evey situation.

### Deep Q-Network (DQN)
Deep Q-Networks (abbreviated as DQNs) approximate the function, **Q(s,a)**, which 
arepresents "how good" it is to take action **a** in state **s**. In Q learning, the metric 
used to evaluate action **a** is the expected sum of discounted future rewards acting 
optimally from state **s**. 

As an agent explores its environment, taking actions and observing the resulting reward, 
experiences (initial state, selected action, reward, next state) are used to update the 
value associated with the initial state and selected action. Provided sufficient exploration,
the q values converge to an approximation of the *optimal* function mapping state action pairs 
to the *optimal* action overtime.


At their most basic, Q-networks simply iteratively update their approximation of the 
Q-function as each experience occurs. Standard DQNs have proven to be very strong
solutions to many difficult reinforcement. However, the basic DQN can be improved
dramatically through various methods which improve convergence reliability and speed. 

The following models are discrete extensions of the basic DQN model. The combination
of all extentions forms a "Rainbow DQN", as title by Google's Deepmind. While each extension
generally improves the training efficiency and accuracy of the agent, the Rainbow DQN has 
been shown to outperform all subcombinations of the following DQN additions in many Arcade
Learning Environments. 

#### Double DQN

#### Dueling DQN

Victory after 1000 Episodes of training

https://www.youtube.com/watch?v=BO5VcUd2RGQ

#### Prioritized Experience Replay (PER)

#### Noisy DQN

#### N-step DQN
This extension improves the temporal awareness of the agent by storing experiences as 
**<st, a, R, st+n, done>**, where **st** is the state at timestep **t** and **a** is the action selected 
(the same values stored in experience/ prioritized experience replay), **R** is the discounted sum of 
rewards over the next **n** states (hence the name n-step), and **st+n** is the nth state following **st**.


#### Distributional DQN

##### Rainbow DQN

#### Resources
[Rainbow DQN arvix paper](https://arxiv.org/abs/1710.02298)


### Advantage Actor Critic (A2C)

### Just Enough Retained Knowledge (JERK)

### Proximal Proximity Opperator (PPO)