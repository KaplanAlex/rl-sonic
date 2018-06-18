# # Starting agent
import numpy as np
from retro_contest.local import make
import skimage
from skimage.viewer import ImageViewer

from dqn_agent import DQN_Agent
from networks import Networks
from parameters import EPISODES, LOAD_MODELS
from util import preprocess_obs

def main():
    env = make(game='SonicTheHedgehog2-Genesis', state='EmeraldHillZone.Act1')
    
    # Parameters for observation image size processing.
    img_rows = 128
    img_cols = 128          
    img_stack = 4        

    action_size = 8         # 8 valid button combinations
    
    # Inputs to the agent's prediction network will have the following shape.
    input_size = (img_rows, img_cols, img_stack)
    
    dqn_agent = DQN_Agent(input_size, action_size)

    # Load previous models, or instantiate new networks.
    if (LOAD_MODELS):
        dqn_agent.load_models()
    else:
        dqn_agent.main_model = Networks.dqn(input_size, action_size, dqn_agent.main_lr)
        dqn_agent.target_model = Networks.dqn(input_size, action_size, dqn_agent.target_lr)
    
    # One episode is 4500 steps if not completed 
    # 5 minutes of frames at 1/15th of a second = 4 60Hz frames
    for episode in range(EPISODES):
        done = False


        first_obs  = env.reset()


        # Experiences are a stack of the img_stack most frames to provide 
        # temporal information. Initialize this sequence to the first 
        # observation stacked 4 times.
        processed = preprocess_obs(first_obs, size=(img_rows, img_cols))
        # (img_rows, img_cols, img_stack)
        exp_stack = np.stack(([processed]*img_stack), axis = 2)
        # Expand dimension to stack and submit multiple exp_stacks in  a batch
        # (1, img_rows, img_cols, img_stack).
        exp_stack = np.expand_dims(exp_stack, axis=0) # 1x64x64x4


        timestep = 0
        while not done:
                
                # Predict an action to take based on the most recent
                # experience. 
                # 
                # Note that the first dimension 
                # (1, img_rows, img_cols, img_stack) is ignored by the
                # network here as it represents a batch size of 1.
                act_idx, action = dqn_agent.act(exp_stack)
                obs, reward, done, info = env.step(action)
                timestep += 1
                # env.render()
                
                obs = preprocess_obs(obs, size=(img_rows, img_cols))
                
                # Create a 1st dimension for stacking experiences and a 4th for 
                # stacking img_stack frames.
                obs = np.reshape(obs, (1, img_rows, img_cols, 1))
                
                # Append the new observation to the front of the stack and remove
                # the oldest (4th) frame.
                exp_stack_new = np.append(obs, exp_stack[:, :, :, :3], axis=3)

                # Save the experience: <state, action, reward, next_state, done>. 
                dqn_agent.save_memory(exp_stack, act_idx, reward, exp_stack_new, done)
                exp_stack = exp_stack_new
                
                # In the observation phase (episode 0, less than "observation_timesteps"), 
                # skip training updates and decrmenting epsilon.
                if (episode != 0 or timestep > dqn_agent.observation_timesteps):
                     
                    # Update the target model with the main model's weights.
                    if (timestep % dqn_agent.update_target_freq == 0):
                        dqn_agent.update_target_model()

                    # Train the agent on saved experiences.
                    if (timestep % dqn_agent.timestep_per_train == 0):
                            dqn_agent.replay_update()
                            dqn_agent.save_models()
                        
                    if (dqn_agent.epsilon > dqn_agent.final_epsilon):
                        # Decrease epsilon by a fraction of the range such that epsilon decreases
                        # for "exploration_timesteps".
                        dec = ((dqn_agent.initial_epsilon - dqn_agent.final_epsilon) / dqn_agent.exploration_timesteps)
                        dqn_agent.epsilon -= dec

                print("Epsisode: ", episode, " Timestep: ", timestep, "Action: ", act_idx)
                
                # info: {'score': 0, 'screen_x_end': 10656, 'screen_y': 525, 
                # 'lives': 3, 'x': 96, 'zone': 0, 'act': 0, 'y': 594, 
                # 'level_end_bonus': 0, 'screen_x': 0, 'rings': 4, 'game_mode': 12}
                print(info)
                print(reward)

        dqn_agent.save_models()


# Run main
if __name__ == '__main__':
    main()