"""
Basic DQN Implementation.

DQNs approximate the q function which maps (state, action) pairs 
to values. 

Only 8 of the valid 2^12 Sega Genesis button combinations
are relevant to Sonic, thus the action space is restriticted to
these 8 actions to expedite training and exploration.
"""

from collections import deque
from keras.models import load_model
import numpy as np
import random


class DQN_Agent:
    
    def __init__(self, input_size, action_size):
        """
        args:
            input_size: 
                The dimension of input observations.
            action_size:
                The dimension of the available actions.           
        """
        # Model structure parameters
        self.input_size = input_size
        self.action_size = action_size

        # Learning Parameters
        self.gamma = 0.99
        self.main_lr = 0.0001
        self.target_lr = 0.0001
        
        # Exploration rate.
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.0001
        
        self.frame_per_action = 4
        self.update_target_freq = 8192 
        self.timestep_per_train = 100 # Number of timesteps between training interval

        self.batch_size = 32
        
        # Memory for experience replay
        self.memory = deque(maxlen=50000)
        
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

        args:
            input_state:
                Information about the world used to predict the 
                optimal aciton.
        Return:
            act_idx: The index of the optimal action action as 
            determined by the current model.

            new_action: The executable action (button combination)
            represented by act_idx.
        """

        if np.random.rand() <= self.epsilon:
            action_idx = random.randrange(self.action_size)
        else:
            # Predict the value associated with each action given the 
            # input state. Greedily choose the best action.
            action_values = self.main_model.predict(input_state)
            action_idx = np.argmax(action_values)
            
        new_action = self.action_switch[action_idx]

        return action_idx, new_action
    
    def save_memory(self, state, act_idx, reward, next_state, done):
        """
        Save an experience <state, action, reward, next_state, done> tuple
        to memory to be used in training. Memories are saved to separate
        the exploration/ information gathering process from learning.
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


    def initialize_action_switch(self):
        """
        Initialize a mapping from the index of an action 
        selection to an action executable within the environment. 
        There are only 8 button combinations relevant to the Sonic 
        game, each which can be encoded as an array of 12 booleans
        representing button presses.

        args:
            act_idx: The index of the action selected by the model.
        returns:
            action_switch: A mapping from the index of an action to
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

    def load_models(self):
        """
        Set the working models to models saved in the following
        locations:
        main_model    <-  "dqn_main.h5"
        target_model  <-  "dqn_target.h5"
        """

        self.main_model = load_model("dqn_main.h5")
        self.target_model = load_model("dqn_target.h5")

    def save_models(self, name):
        """
        Save the current models to the following locations:
        main_model    ->  "dqn_main.h5"
        target_model  ->  "dqn_target.h5"
        
        Useful for pausing and restarting training.
        """
        self.main_model.save("dqn_main.h5")
        self.target_model.save("dqn_target.h5")
