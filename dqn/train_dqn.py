# # Starting agent
from retro_contest.local import make
import skimage
from skimage.viewer import ImageViewer

import time

#from dqn_agent import DQN_Agent
#from networks import Networks
from util import preprocess_obs

def main():
    env = make(game='SonicTheHedgehog2-Genesis', state='EmeraldHillZone.Act1')
    env.reset()
    
    # Following preprocessing, the game scene will be reflected
    # as a stack of 4 64x64 black and white images.
    img_rows = 128
    img_cols = 128          
    # img_stack = 4        

    # input_size = (img_rows, img_cols, img_stack)
    # action_size = 8         # 8 valid button combinations

    # dqn_agent = DQN_Agent(input_size, action_size)
    # dqn_agent.main_model = Networks.dqn(input_size, action_size, dqn_agent.main_lr)
    # dqn_agent.target_model = Networks.dqn(input_size, action_size, dqn_agent.target_lr)
    action = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    obs, rew, done, info = env.step(action)
    preprocess_obs(obs, size=(img_rows, img_cols))
    
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