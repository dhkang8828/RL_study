import numpy as np
from environment import Env

gamma = 0.9

def get_state_index(state_space, state):
    for i_s, s in enumerate(state_space):
        if (s == state).all():
            return i_s
    assert False, "Couldn't find the state from the state_space"

def calc_return(gamma, rewards):
    n = len(rewards)
    rewards = np.array(rewards)
    gammas = gamma * np.ones([n])
    powers = np.arange(n)

    power_of_gammas = np.power(gammas, powers)
    ## [r,gamma * r,gamma^2 * r,gamma^3, ...]
    discounted_rewards = rewards * power_of_gammas
    g = np.sum(discounted_rewards)

    return g

def mc_value_prediction(env, policy):
    value_vector = np.zeros([len(env.state_space)])
    returns = [{'n':0, 'avg':0} for s in env.state_space]

    ## Repeat Policy Evaluation
    for loop_count in range(10000):
        episode = {
            'states' : list(),
            'actions' : list(),
            'rewards' : list(),
        }
        done = False
        step_count = 0
        s = env.reset()

        ## Generate an episode
        while not done:
            i_s = get_state_index(env.state_space, s)
            pi_s = policy[i_s]
            a = np.random.choice(env.action_space, p=pi_s)
            r, s_next, done = env.step(a)

            episode['states'].append(s)
            episode['actions'].append(a)
            episode['rewareds'].append(r)

            step_count += 1
            s = s_next
        episode['states'].append(s) ##append s_T(the termination state)

        for t in range(step_count):
            s_t = episode['states'][t]
            i_s_t = get_state_index(env.state_space, s_t)
            g_t = calc_return(gamma, episode['rewards'][t:])

            n_prev, avg_prev = returns[i_s_t]['n'], returns[i_s_t]['avg']
            returns[i_s_t]['avg'] = (avg_prev * n_prev + g_t) / (n_prev + 1)
            returns[i_s_t]['n'] = n_prev + 1
            value_vector[i_s_t] = returns[i_s_t]['avg']

        if (loop_count + 1) % 100 == 0:
            print(f"[{loop_count}] value_vector: \n{value_vector}")

    return value_vector


