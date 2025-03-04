## TD_Prediction.py

import numpy as np
from environment import Env

gamma = 0.9
alpha = 1e-3

def get_state_index(state_space, state):
    for i_s, s in enumerate(state_space):
        if (s == state).all():
            return i_s
    assert False, "Couldn't find the state from the state space"

def td_value_prediction(env, policy):
    value_vector = np.zeros([len(env.state_space)])


