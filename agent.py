# # Starting agent
from retro_contest.local import make
import time

def main():
    env = make(game='SonicTheHedgehog2-Genesis', state='EmeraldHillZone.Act1')
    obs = env.reset()
    while True:
        # Left
        action = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        # Right
        action = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        # B
        action = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # Down, B
        action = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        # Right, down
        action = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]
        #B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z
        print(action)
        obs, rew, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()




