## Utils.py

import gym
import numpy as np

def create_env(config):
    env = gym.make("CartPole-v1")
    env = RewardShapingWrapper(env)
    env = TimeStepAppendWrapper(env)
    
    return env

class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        
    def step(self, action):
        s_next, r, done, info = self.env.step(action)
        r = 0.1 * r
        return s_next, r, done, info
    
class TimeStepAppendWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        self.step_count = 0

if __name__ == "__main__":
    print("Creating environment...")