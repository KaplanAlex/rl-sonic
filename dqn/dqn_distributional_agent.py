"""
Distributional Double DQN Implementation.

Distributional DQNs model the distribution of expected rewards
for all potential (state, action) pairs.
"""

from collections import deque
from keras.models import load_model
import math
import numpy as np
import random


class DQN_Agent:
    """
    Implementation of a Double Distributional DQN. Specifically, the distributional
    DQN is a C51 network, which outputs a distribution with 51 values for each
    action. 

    Depending on the selected models, this agent can also implement
    one or more of the following extensions:
        - Dueling DQN
        - Noisy DQN
        - n-step DQN
    """
    def __init__(self, input_size, action_size):
        """
        Arguments:
            - input_size: The dimension of input observations.
            - action_size: The dimension of the available actions.           
        """
        # Model structure parameters
        self.input_size = input_size
        self.action_size = action_size

        # Learning Parameters
        self.gamma = 0.95
        self.main_lr = 0.0001
        self.target_lr = 0.0001
        
        # Intial observation timesteps - epsilon is not changed and 
        # training does not occur during initial observation. 
        self.observation_timesteps = 5000

        # Exploration rate.
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.01
        
        # Timesteps between initial and final epsilon following 
        # initial observation phase
        self.exploration_timesteps = 4000000
        self.frame_per_action = 4
        self.update_target_freq = 10000 
        
        # Boolean flag signifying that randomness is built into the 
        # model network
        self.noisy = False

        # Number of timesteps between training intervals.
        self.timestep_per_train = 2000 
        
        # Number of experiences used simultaneously in training.
        self.batch_size = 64
        
        # Memory for experience replay
        self.memory = deque(maxlen=50000)
        
        # Number of future states used to form memories in n-step dqn 
        self.n_step = 3

        #The support for the value distribution. Set to 51 for C51
        self.num_atoms = 51
        
        # Break the range of rewards into 51 uniformly spaced values (support)
        self.v_max = 5 * self.n_step # Rewards are clipped to -20, 20
        self.v_min = -5 * self.n_step
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]


        # Main model used to predict q-values.
        self.main_model = None 

        #Second model with the same capabilities as the main model 
        # with fixed weights - improves convergence.
        self.target_model = None

        # A dictionary mapping action indicies to executable actions 
        # (button combinations).
        self.action_switch = self.initialize_action_switch()
    
    
    def act(self, input_state):
        """
        Return an action given an input observation
        based on the current DQN model, acting randomly
        self.epsilon portion of the time (encourages exploration).

        Arguments:
            - input_state: Information about the world used to predict ]
              the optimal aciton.
        Return:
            - act_idx: The index of the optimal action action as 
              determined by the current model.
            - new_action: The executable action (button combination)
              represented by act_idx.
        """

        # Act randomly if NoisyNet is not applied.
        if (np.random.rand() <= self.epsilon) and (not self.noisy):
            action_idx = random.randrange(self.action_size)
        else:
            # Predict the value associated with each action given the 
            # input state. Greedily choose the best action.
            action_distributions = self.main_model.predict(input_state)
            # Determine the index of the optimal action by computing the 
            # average value of each predicted distribution.
            # Multiply each distribution by the value ranges corresponding to 
            # each probablity.
            weighted_dists = np.multiply(np.vstack(action_distributions), np.array(self.z))
        
            # Sum the weighted values from each distribution and find the one with 
            # the largest expected value.
            avg_dist_values = np.sum(weighted_dists, axis=1)
            act_idx = np.argmax(avg_dist_values)
        
        new_action = self.action_switch[action_idx]

        return action_idx, new_action
    
    def save_memory(self, state, act_idx, reward, next_state, done):
        """
        Save an experience <state, action, reward, next_state, done> tuple
        to memory to be used in training. Memories are saved to separate
        the exploration/ information gathering process from learning.

        Arguments:
            - state: A (1, 128, 128, 4) np.array containing 4 preprocessed
              frames from the sonic game.
            - act_idx: The index of the action prediction made by the agent.
            - reward: The reward received for taking the action
            - next_state: A (1, 128, 128, 4) np.array containing 3 of the
              prevous frames and the new frame resulting from taking the action.
            - done: Boolean representing the end of the episode.
        """
        self.memory.append((state, act_idx, reward, next_state, done))

    def update_target_model(self):
        """
        Update the target model with the weights of updated model
        "main_model". 
        
        The target model is used to produce consistent value approximations 
        during the training process, as DQNs are trained relative to their 
        own predictions which can easily lead to divergence.
        """
        self.target_model.set_weights(self.main_model.get_weights())
        print("Target Model Updated")

    def compute_targets(self, mini_batch):
        """
        Computes the target values and prediction error for all experiences 
        in a minibatch. Implentation is based off of the Deepmind Distributional
        DQN arvix paper linked in the README.

        Arguments:
            - mini_batch: A collection of 1+ experiences sampled from memory.
        
        Returns:
            - update_input: Collection of input states parsed from sampled 
              experiences. Formatted for network training.
            - prediction: New predicted distribution for the action taken
              in each experience within the mini_batch.
            - error: The difference between the previous and updated q value 
              for the changed action for each sampled experience.
        """
        batch_size = len(mini_batch)

        # Batch size experiences - (batch_size, img_rows, img_cols, 4)
        update_input = np.zeros(((batch_size,) + self.input_size)) 
        update_target = np.zeros(((batch_size,) + self.input_size)) 
        actions, rewards, done = [], [], []

        # Probability distributions for each action for each input in the batch. Used
        # to update the current distribution model through cross-entropy loss. 
        m_prob = [np.zeros((batch_size, self.num_atoms)) for i in range(self.action_size)]

        # Extract information from the sampled memories.
        for i in range(batch_size):
            # Append each state (1, 128, 128, 4) to update_input
            update_input[i,:,:,:] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            # Append Next State (1, 128, 128, 4) to update_target
            update_target[i,:,:,:] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        # Predict the values associated with acting optimally from next_state.
        # (action_size- # of outputs -, batch_size, num_atoms)
        next_state_dist = self.main_model.predict(update_target)
        ref_target_dist = self.target_model.predict(update_target)

        # Stack all of the distributions from all actions and all inputs 
        # (batch_size * action_size, num_atoms)
        stacked_dist = np.vstack(next_state_dist)

        # Multiply all distributions by value ranges, then sum each distribution,
        # leaving a (batch_size, num_actions) matrix for each input in the batch.
        q_values = np.sum(np.multiply(stacked_dist, np.array(self.z)), axis = 1)
        q_values = q_values.reshape((batch_size, self.action_size), order='F')

        # The optimal action in the prediction distribution from the main model for 
        # each input.
        next_act_idx = np.argmax(q_values, axis=1)

        error = []
        # Project the next_state value distribution to the current distribution
        for batch_idx in range(batch_size):
            # Predicted q value before updating.
            #  
            # (distributions for selected action -> correct input ->
            # summed to evluated distribution value)
            prev_q = np.sum(next_state_dist[actions[batch_idx]][batch_idx])
            
            # If the state is terminal, set the action equal to the reward.
            if done[batch_idx]:
                # No next state exists, so the distribution can only be shifted.
                # Ensure rewards are clipped.
                Tz = min(self.v_max, max(self.v_min, rewards[batch_idx]))
                # Determine which segments the reward belongs to
                bj = math.floor((Tz - self.v_min) / self.delta_z) 
                segment_l = math.floor(bj)
                segment_u = math.ceil(bj)
                # Convert the rough segment value to indicies for upper and lower.
                # Add the portion of the range (0-1) which belongs to each segment. 
                m_prob[actions[batch_idx]][batch_idx][int(segment_l)] += (segment_u - bj)
                m_prob[actions[batch_idx]][batch_idx][int(segment_u)] += (bj - segment_l)
            else:
                # Scale, shift, and project the new distribution to the old predicted distribuiton.
                for atom in range(self.num_atoms):
                    # Ensure rewards are clipped.
                    Tz = min(self.v_max, max(self.v_min, rewards[batch_idx] + self.gamma * self.z[atom])) 
                   # Determine which segments the reward belongs to
                    bj = math.floor((Tz - self.v_min) / self.delta_z) 
                    segment_l = math.floor(bj)
                    segment_u = math.ceil(bj)
                    lower_prob = ref_target_dist[next_act_idx[batch_idx]][batch_idx][atom] * (segment_u - bj)
                    upper_prob = ref_target_dist[next_act_idx[batch_idx]][batch_idx][atom] * (bj - segment_l)
                    m_prob[actions[batch_idx]][batch_idx][int(segment_l)] += lower_prob
                    m_prob[actions[batch_idx]][batch_idx][int(segment_u)] += upper_prob
            
            # Q value following distribution update
            q_update = np.sum(m_prob[actions[batch_idx]][batch_idx])            
            error.append(abs(prev_q - q_update))

        return update_input, m_prob, error

    def replay_update(self):
        """
        Updates the "main_model" based on "batch_size" stochastically sampled
        memories.
        """
        
        # Sample memories based on priorities. Also returns the index 
        # of each sampled experience (to facilitate updates).
        mini_batch = random.sample(self.memory, self.batch_size)
        
        # Extract the observed state from the minibatch and compute the update targets
        # and error.
        update_input, prediction, _ = self.compute_targets(mini_batch)

        # Fit the current model to the updated q values predictions through
        # a single gradient descent update which implements bellman's equation.
        # Q(state, action) = reward + y(max(Q(next_state, next_action))
        loss = self.main_model.train_on_batch(update_input, prediction)

        return np.amax(prediction[-1]), loss

    def initialize_action_switch(self):
        """
        Initialize a mapping from the index of an action 
        selection to an action executable within the environment. 
        There are only 8 button combinations relevant to the Sonic 
        game, each which can be encoded as an array of 12 booleans
        representing button presses.

        Arguments:
            - act_idx: The index of the action selected by the model.
        Returns:
            - action_switch: A mapping from the index of an action to
              an array encoding the button presses to execute the action.
        """
        # Map indicies to arrays with 12 values encoding button presses
        # (B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z)
        action_switch = {
            # No Operation
            0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Left
            1: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            # Right
            2: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            # Left, Down
            3: [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            # Right, Down
            4: [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            # Down
            5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            # Down, B
            6: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            # B
            7: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }

        return action_switch

    def load_models(self, main_path, target_path):
        """
        Set the working models to models saved in the input
        locations.
        
        Arguments:
            - main_path: Path to the saved main model.
            - target_path: Path to the saved target model.
        """
        self.main_model = load_model(main_path)
        self.target_model = load_model(target_path)
        print("Models Loaded")

    def save_models(self, main_path, target_path):
        """
        Save the current models to the input locations.
        
        Arguments:
            - main_path: Storage location for main model.
            - target_path: Storage location for target model.
        """
        self.main_model.save(main_path)
        self.target_model.save(target_path)

        print("Models saved")
