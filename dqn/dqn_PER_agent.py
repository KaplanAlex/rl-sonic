

from collections import deque
from keras.models import load_model
import numpy as np
import random

from priority_memory import PriorityMemory


class DQN_PER_Agent:
    """
    DQN with Prioritized Experience Replay (PER)

    This implementation improves "dqn_agent.py" by changing the uniform sampling 
    experience replay policy to prioritized experience replay (PER). PER encourages 
    the selection of experiences with large error between the predicted Q and target
    values, as these memories provide the best insight into performance of the model. 
    
    PER has been shown to decrease training time significantly (by a factor of 2)
    in many ALE environments.

    Depending on the selected models, this agent can also implement
    the following extensions:
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
        self.observation_timesteps = 10000

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
        
        # Memory for experience replay.
        self.memory_len = 5000
        self.memory = PriorityMemory(self.memory_len)
        
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
            action_values = self.main_model.predict(input_state)
            action_idx = np.argmax(action_values)
            
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
        experience = (state, act_idx, reward, next_state, done)
        
        # Determine the error between the q and target values for this experience.
        _, _, error = self.compute_targets([experience])

        # Save the memory with its priority to encourage sampling based on its 
        # influence.
        self.memory.add(experience, error[0])
    
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
        in a minibatch.

        Arguments:
            - mini_batch: A collection of 1+ experiences sampled from memory.
        
        Returns:
            - update_input: Collection of input states parsed from sampled 
              experiences. Formatted for network training.
            - prediction: New predicted Q values associated with each input 
              state.
            - error: The difference between the previous and updated q value 
              for the changed action for each sampled experience.
        """
        # Batch size experiences - (batch_size, img_rows, img_cols, 4)
        update_input = np.zeros(((self.batch_size,) + self.input_size)) 
        update_target = np.zeros(((self.batch_size,) + self.input_size)) 
        action, reward, done = [], [], []

        # Extract information from the sampled memories.
        for i in range(self.batch_size):
            # Append each state (1, 128, 128, 4) to update_input
            update_input[i,:,:,:] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            # Append Next State (1, 128, 128, 4) to update_target
            update_target[i,:,:,:] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        # Predict the q values for each input - (batch_size, action_size)
        prediction = self.main_model.predict(update_input) 
        
        # Predict the values associated with acting optimally from next state.
        next_state_pred = self.main_model.predict(update_target)
        target_pred = self.target_model.predict(update_target)

        error = []
        for sample_idx in range(self.batch_size):
            # Q-value prior to update
            prev_q = next_state_pred[sample_idx][action[sample_idx]]

            # For terminal actions, set the value associated with the action to
            # the observed reward
            if done[sample_idx]:
                predicted_val = reward[sample_idx]
            else:
                # Otherwise, set the value to the observed reward + the discounted
                # predicted value of acting optimally from the next state on.
                #
                # The predicted update values are taken from the "fixed" target_model,
                # while the action is selected from the dynamic main_model. This provides
                # consistency between training iterations as the q values of the "fixed"
                # model do not change every iteartion
                next_action = np.argmax(next_state_pred[sample_idx])
                predicted_val = reward[sample_idx] + self.gamma * (target_pred[sample_idx][next_action])
            error.append(abs(prev_q - predicted_val))
        
        # Update the q value associated with the action taken.
        prediction[sample_idx][action[sample_idx]] = predicted_val

        return update_input, prediction, error



    def replay_update(self):
        """
        Updates the "main_model" based on "batch_size" stochastically sampled
        memories.
        """
        
        # Sample memories based on priorities. Also returns the index 
        # of each sampled experience (to facilitate updates).
        mini_batch, indicies = self.memory.sample(self.batch_size)
        
        # Extract the observed state from the minibatch and compute the update targets
        # and error.
        update_input, prediction, error = self.compute_targets(mini_batch)

        # Update the priorities of all sampled memories.
        for i in range(self.batch_size):
            idx = indicies[i]
            self.memory.update(idx, error[i])

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

    def load_models(self):
        """
        Set the working models to models saved in the following
        locations:
        main_model    <-  "dqn_main.h5"
        target_model  <-  "dqn_target.h5"
        """
        self.main_model = load_model("dqn_main.h5")
        self.target_model = load_model("dqn_target.h5")
        print("Models Loaded")

    def save_models(self):
        """
        Save the current models to the following locations:
        main_model    ->  "dqn_main.h5"
        target_model  ->  "dqn_target.h5"
        
        Useful for pausing and restarting training.
        """
        self.main_model.save("dqn_main.h5")
        self.target_model.save("dqn_target.h5")

        print("Models saved")
