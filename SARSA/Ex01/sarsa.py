## SARSA
import numpy as np
from environment import Env

gamma = 0.9
k_alpha = 1e-3
k_eps = 2e-4

def get_state_index(state_space, state):
    for i_s, s in enumerate(state_space):
        if (s == state).all():
            return i_s
    assert False, "Couldn't find the state from the state space"

def sarsa(env):
    action_value_matrix = np.zeros([len(env.state_space), len(env.action_space)])

    def sample_action(eps, action_value):
        a_max = action_value.argmax()
        pi = np.zeros([len(env.action_space)])
        pi[:] = eps / len(env.action_space)
        pi[a_max] = pi[a_max] + 1 - eps
        a = np.random.choice(env.action_space, p=pi)
        return a 
