"""
Basic DQN Implementation.

DQNs approximate the q function which maps state, action pairs to 
values. 

Only 8 of the valid 2^12 Sega Genesis button combinations
are relevant to Sonic, thus the action space is restriticted to
these actions to expedite training
"""

from collections import deque
import numpy as np
import random


class DQN_Agent:
    
    def __init__(self, input_size, action_size, main_model, target_model):
        """
        args:
            input_size: 
                The dimensionality of input observations.
            action_size:
                The dimensionality of the available actions.
            main_model:
                Model used to predict q values associated with 
                state aciton pairs.
            target_model:
                Second model with the same capabilities as the
                main model with fixed weights - improves convergence
        """
        # Model structure parameters
        self.input_size = input_size
        self.action_size = action_size

        # Learning parameters
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.0001
        self.batch_size = 32
        self.observe = 5000
        self.explore = 50000 
        self.frame_per_action = 4
        self.update_target_freq = 3000 
        self.timestep_per_train = 100 # Number of timesteps between training interval

        # Memory for experience replay
        self.memory = deque(maxlen=50000)
        
        # The main and target models used to predict q-values
        self.model = main_model
        self.target_model = target_model

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
            action_values = self.model.predict(input_state)
            action_idx = np.argmax(q)
            
        new_action = self.action_switch[action_idx]

        return action_idx, new_action
    

    def initialize_action_switch(self, act_idx):
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
        self.action_switch = {
            # No Operation
            0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            # Left
            1: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
            # Right
            2: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            # Left, Down
            3: [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
            # Right, Down
            4: [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
            # Down
            5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            # Down, B
            6: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            # B
            7: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }

        return action_switch

    def update_target_model(self):
        """
        After some time interval update the target model to be same with model
        """
        self.target_model.set_weights(self.model.get_weights())



    def load_model(self, name):
        """
        Load a saved model
        """
        self.model.load_weights(name)

    def save_model(self, name):
        """
        Save a model. 
        Useful for pausing and restarting training.
        """
        self.model.save_weights(name)
