# # Starting agent
import numpy as np
from retro_contest.local import make
import skimage
from skimage.viewer import ImageViewer

from dqn_agent import DQN_Agent
from networks import Networks
from parameters import EPISODES
from util import preprocess_obs

def main():
    env = make(game='SonicTheHedgehog2-Genesis', state='EmeraldHillZone.Act1')
    first_obs  = env.reset()
    
    # Parameters for observation image size processing.
    img_rows = 128
    img_cols = 128          
    img_stack = 4        

    action_size = 8         # 8 valid button combinations
    input_size = (img_rows, img_cols, img_stack)
    

    dqn_agent = DQN_Agent(input_size, action_size)
    dqn_agent.main_model = Networks.dqn(input_size, action_size, dqn_agent.main_lr)
    dqn_agent.target_model = Networks.dqn(input_size, action_size, dqn_agent.target_lr)
    
    
    # Experiences are a stack of the img_stack most frames to provide temporal information.
    # Initialize this sequence to the first observation stacked 4 times.
    processed = preprocess_obs(first_obs, size=(img_rows, img_cols))
    # (img_rows, img_cols, img_stack)
    exp_stack = np.stack(([processed]*img_stack), axis = 2)
    # Expand dimension to stack and submit multiple exp_stacks in  a batch
    # (1, img_rows, img_cols, img_stack).
    exp_stack = np.expand_dims(exp_stack, axis=0) # 1x64x64x4

    
    

    
    
    print(exp_stack.shape)
    num_samples = 10

    zero_test = np.zeros(((num_samples,) + input_size))
    print(zero_test.shape)
    
    # One episode is 4500 steps if not completed 
    # 5 minutes of frames at 1/15th of a second = 4 60Hz frames
    for episode in range(EPISODES):
        done = False
        timestep = 0
        while not done:
                action = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                obs, rew, done, info = env.step(action)
                
                timestep += 1
                print("Epsisode: ", episode, " Timestep: ", timestep)
        dqn_agent.save_models()

    # while True:
    #     # Left
    #     action = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    #     # Right
    #     action = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    #     # B
    #     action = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     # Down, B
    #     action = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    #     # Right, down
    #     action = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]
    #     #B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z
    #     # print(action)
    #     obs, rew, done, info = env.step(action)
    #     env.render()
    #     if done:
    #         obs = env.reset()


if __name__ == '__main__':
    main()