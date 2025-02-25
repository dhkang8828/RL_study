'''
    Copy Right: FastCampus
'''
## TD Prediction

import numpy as np
from environment import Env

gamma = 0.9
alpha = 5e-3

def get_state_index(state_space, state):
    for i_s, s in enumerate(state_space):
        if (s == state).all():
            return i_s
    assert False, "Couldn't find the state from the state space" 

def td_value_prediction(env, policy):
    value_vector = np.zeros([len(env.state_space)])

    ## Repeat Policy Evaluation
    for loop_count in range(10000):
        done = False
        step_count = 0
        s = env.reset()

        ## Generate an episode
        while not done:
            i_s = get_state_index(env.state_space, s)
            pi_s = policy[i_s]
            a = np.random.choice(env.action_space, p=pi_s)
            r, s_next, done = env.step(a)

            i_s_next = get_state_index(env.state_space, s_next)
            td = r + gamma * value_vector[i_s_next] - value_vector[i_s]
            value_vector[i_s] = value_vector[i_s] + alpha * td 

            if done:
                value_vector[i_s_next] = 0

            step_count += 1
            s = s_next

        if (loop_count + 1) % 100 == 0:
            print(f"[{loop_count}] value_vector: \n{value_vector}")

    return value_vector

def td_action_value_prediction(env, policy):
    action_value_matrix = np.zeros([len(env.state_space), len(env.action_space)])

    ## Repeat Policy Evaluation
    for loop_count in range(10000):
        done = False
        step_count = 0
        s = env.reset()

        i_s = get_state_index(env.state_space, s)
        pi_s = policy[i_s]
        a = np.random.choice(env.action_space, p=pi_s)

        ## Generate an Episode
        while not done:

            r, s_next, done = env.step(a)
            i_s_next = get_state_index(env.state_space, s_next)
            pi_s_next = policy[i_s_next]
            a_next = np.random.choice(env.action_space, p=pi_s_next)

            td = r + gamma * action_value_matrix[i_s_next][a_next] - action_value_matrix[i_s][a]
            action_value_matrix[i_s][a] = action_value_matrix[i_s][a] + 5 * alpha * td

            if done:
                action_value_matrix[i_s_next] = 0

            step_count += 1 
            s = s_next
            i_s = i_s_next
            a = a_next

        if (loop_count + 1) % 100 == 0:
            print(f"[{loop_count}] action_value_matrix: \n{action_value_matrix}")

    return action_value_matrix

if __name__ == "__main__":
    np.set_printoptions(formatter={'float':'{: 0.3f}'.format})

    env = Env()
    policy1 = list()
    





