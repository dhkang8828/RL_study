# n_step_td_prediction.py

import numpy as np
import matplotlib.pyplot as plt
from environment import Env

gamma = 0.9
alpha_init = 2e-1
k_alpha = 2.5e-1

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
    ## [r, gamma * r, gamma^2 * r, ...]
    discounted_rewards = rewards * power_of_gammas 
    g = np.sum(discounted_rewards)
    
    return g
    
def n_step_td_value_prediction(env, policy, n, color):
    value_vector = np.zeros([len(env.state_space)])
    plot_buffer = {'x':[], 'y':[]}
    
    ## Repeat Policy Evaluation
    for loop_count in range(1000):
        trajectory = {
            'states': list(),
            'actions': list(),
            'rewards': list(),            
        }
        done = False
        step_count = 0
        s = env.reset()
        trajectory['states'].append(s)
        
        ## Generate a trajectory
        while not done:
            i_s = get_state_index(env.state_space, s)
            pi_s = policy[i_s]
            a = np.random.choice(env.action_space, p=pi_s)
            r, s_next, done = env.step(a)
            
            trajectory['states'].append(s_next)
            trajectory['actions'].append(a)
            trajectory['rewards'].append(r)
            
            s = s_next
            step_count += 1
            
            if step_count >= n + 1:
                ## Remove updated transition
                trajectory['states'].pop(0)
                trajectory['actions'].pop(0)
                trajectory['rewards'].pop(0)
                
            if step_count >= n:
                assert len(trajectory['rewards']) == n, f"Trajectory length should" 
                + f"be n={n} but {len(trajectory['rewards'])}"
                
                s_t_sub_n = trajectory['states'][0]
                i_s_t_sub_n = get_state_index(env.state_space, s_t_sub_n)
                s_t = trajectory['states'][-1]
                i_s_t = get_state_index(env.state_space, s_t)
                
                alpha = alpha_init / (1 + k_alpha * loop_count) 
            
            