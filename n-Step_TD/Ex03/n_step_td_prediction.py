## n-step_td_prediction.py

import numpy as np
import matplotlib.pyplot as plt
from environment import Env

gamma = 0.9
alpha_init = 3e-1
k_alpha = 1e-1

def get_state_index(state_space, state):
    for i_s, s in enumerate(state_space):
        if (s == state).all():
            return i_s
    assert False, "Couldn't find the state from the state space"
    
def calc_return(gamma, rewards):
    n = len(rewards)
    rewards = np.array(rewards)
    gammas = gamma * np.ones([n])
    powers = np.arange(n)
    
    power_of_gammas = np.power(gammas, powers)
    discounted_rewards = rewards * power_of_gammas 
    
    g = np.sum(discounted_rewards)
    return g
        