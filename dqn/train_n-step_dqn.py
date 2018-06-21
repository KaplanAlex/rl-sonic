"""
Logic to train the n-step DQN extension.
This process differs from the general training process only how experiences are 
processed and recorded. For this reason, all other extensions to the basic DQN
model can be trained in conjunction with n-step with this script.

n-step DQN improves the temporal awareness of the agent by saving a state,
action pair in memory with the discounted sum of rewards over the next 'n' 
steps and the 'nth' future state.
"""
from collections import deque
import numpy as np
from retro_contest.local import make

from dqn_agent import DQN_Agent
from dqn_PER_agent import DQN_PER_Agent
from networks import Networks
from parameters import EPISODES, LOAD_MODELS, EPSILON, START, \
                       MIDDLE, FINAL, PER_AGENT, DUELING, NOISY
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
    
    # File paths
    stat_path = '../statistics/dqn_n-step'

    # Priortized Experience Replay.
    if (PER_AGENT):
        print('PER agent')
        stat_path += '_PER'
        dqn_agent = DQN_PER_Agent(input_size, action_size)
    else:
        dqn_agent = DQN_Agent(input_size, action_size)

    # Load previous models, or instantiate new networks.
    if (LOAD_MODELS):
        dqn_agent.load_models()
    else:
        # Use the Noisy Dueling Network.
        if (NOISY):
            stat_path += '_Noisy_Dueling'
            print('NOISY Dueling agent')
            dqn_agent.main_model = Networks.noisy_dueling_dqn(input_size, action_size, dqn_agent.main_lr)
            dqn_agent.target_model = Networks.noisy_dueling_dqn(input_size, action_size, dqn_agent.target_lr)
            dqn_agent.noisy = True
        # Use the normal dueling network.
        elif (DUELING):
            stat_path += '_Dueling'
            print('Dueling agent')
            dqn_agent.main_model = Networks.dueling_dqn(input_size, action_size, dqn_agent.main_lr)
            dqn_agent.target_model = Networks.dueling_dqn(input_size, action_size, dqn_agent.target_lr)
        # Normal DQN.
        else:
            dqn_agent.main_model = Networks.dqn(input_size, action_size, dqn_agent.main_lr)
            dqn_agent.target_model = Networks.dqn(input_size, action_size, dqn_agent.target_lr)
    
    stat_path += '_stats.csv'

    # TODO Temporary
    stat_path = 'dueling_stats.csv'
    
    if (EPSILON == START):
        dqn_agent.epsilon = dqn_agent.initial_epsilon
    elif (EPSILON == MIDDLE):
        dqn_agent.epsilon = ((dqn_agent.initial_epsilon - dqn_agent.final_epsilon) / 2)
    else:
        dqn_agent.epsilon = dqn_agent.final_epsilon

    # Store rewards and states from the previous n state, action pairs to 
    # create experiences.
    prev_n_rewards = deque(maxlen=dqn_agent.n_step) 
    prev_n_exp = deque(maxlen=dqn_agent.n_step)
    
    # One episode is 4500 steps if not completed 
    # 5 minutes of frames at 1/15th of a second = 4 60Hz frames
    total_timestep = 0              # Total number of timesteps over all episodes.
    for episode in range(EPISODES):
        done = False
        reward_sum = 0          # Average reward within episode.
        timestep = 0            # Track timesteps within the episode.
        
        # Rewards and states must be consecutive to improve temporal awareness.
        # Reset at the start of each episode to compensate for sudden scene change.
        prev_n_rewards.clear()
        prev_n_exp.clear()

        # Experiences are a stack of the img_stack most frames to provide 
        # temporal information. Initialize this sequence to the first 
        # observation stacked 4 times.
        first_obs  = env.reset()
        processed = preprocess_obs(first_obs, size=(img_rows, img_cols))
        # (img_rows, img_cols, img_stack)
        exp_stack = np.stack(([processed]*img_stack), axis = 2)
        # Expand dimension to stack and submit multiple exp_stacks in  a batch
        # (1, img_rows, img_cols, img_stack).
        exp_stack = np.expand_dims(exp_stack, axis=0) # 1x64x64x4
        
        # Continue until the end of the zone is reached or 4500 timesteps have 
        # passed.
        while not done:
                # Predict an action to take based on the most recent
                # experience. 
                # 
                # Note that the first dimension 
                # (1, img_rows, img_cols, img_stack) is ignored by the
                # network here as it represents a batch size of 1.
                act_idx, action = dqn_agent.act(exp_stack)
                obs, reward, done, info = env.step(action)
                # env.render()
                
                timestep += 1
                total_timestep += 1
                reward_sum += reward                
                
                # Create a 1st dimension for stacking experiences and a 4th for 
                # stacking img_stack frames.
                obs = preprocess_obs(obs, size=(img_rows, img_cols))
                obs = np.reshape(obs, (1, img_rows, img_cols, 1))
                
                # Append the new observation to the front of the stack and remove
                # the oldest (4th) frame.
                exp_stack_new = np.append(obs, exp_stack[:, :, :, :3], axis=3)
                

                # Save the previous state, selected action, and resulting reward
                prev_n_rewards.appendleft(reward)
                prev_n_exp.append((exp_stack, act_idx, done))
                exp_stack = exp_stack_new
                
                # Once sufficent steps have been taken, discount rewards and save nth 
                # previous experience
                if (len(prev_n_rewards) >= dqn_agent.n_step):
                    # Compute discounted reward
                    reward_sum = 0
                    for idx in range(prev_n_rewards):
                        reward = prev_n_rewards[idx]
                        # rewards are append left so that the most recent rewards are 
                        # discounted the least.
                        reward_sum += ((dqn_agent.gamma ** idx) * reward)
                    
                    # Experiences are pushed forward into the deque as more are appened. The
                    # nth previous experience is at the last index.
                    original_state, original_act, _ = prev_n_exp[-1]
                    nth_state, _, nth_done = prev_n_exp[0]

                # Save the nth previous state and predicted action the discounted sum of rewards
                # and final state over the next n steps.
                dqn_agent.save_memory(original_state, original_act, reward_sum, nth_state, nth_done)
                
                # In the observation phase skip training updates and decrmenting epsilon.
                if (total_timestep >= dqn_agent.observation_timesteps):
                     
                    # Update the target model with the main model's weights.
                    if ((total_timestep % dqn_agent.update_target_freq) == 0):
                        dqn_agent.update_target_model()

                    # Train the agent on saved experiences.
                    if ((total_timestep % dqn_agent.timestep_per_train) == 0):
                            dqn_agent.replay_update()
                            dqn_agent.save_models()
                        
                    if (dqn_agent.epsilon > dqn_agent.final_epsilon):
                        # Decrease epsilon by a fraction of the range such that epsilon decreases
                        # for "exploration_timesteps".
                        dec = ((dqn_agent.initial_epsilon - dqn_agent.final_epsilon) / dqn_agent.exploration_timesteps)
                        dqn_agent.epsilon -= dec

                # print(info)
                print("Epsisode:", episode, " Timestep:", timestep, " Action:", act_idx, " Episode Reward Sum:", reward_sum, " Epsilon:", dqn_agent.epsilon)
        
        # Save mean episode reward at the end of the episode - append to stats file            
        with open(stat_path, "a") as stats_fd:
            reward_str = "Epsiode Cummulative Reward: " + str(reward_sum) + ", Episode Timestpes: " +  str(timestep) + ",\n"
            stats_fd.write(str(reward_str))
            
# Run main
if __name__ == '__main__':
    main()