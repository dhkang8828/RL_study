#environment.py
import numpy as np 

class Env:
    def __init__(self):
        '''
        state_space: 4x4 grid info using numpy
        value of the agent location: 1
        value of the goal location: -1

        action_space: {0, 1, 2, 3}
        0: up
        1: right
        2: down
        3: left
        '''
        
        ## pre-condition
        self.agent_pos = {'y':0, 'x':0}     ##dictionary
        self.goal_pos = {'y':3, 'x':3}
        self.y_min, self.x_min, self.y_max, self.x_max = 0, 0, 3, 3

        ## set up state
        self.state = np.zeros([4,4])
        self.state[self.goal_pos['y'], self.goal_pos['x']] = -1
        self.state[self.agent_pos['y'], self.agent_pos['x']] = 1

        ## state space는 환경에 존재하는 모든 state에 대한 정보를 가져야 한다.
        self.state_space = list()
        for y in range(4):
            for x in range(4):
                state = np.zeros([4,4])
                state[self.goal_pos['y'], self.goal_pos['x']] = -1
                state[y, x] = 1 
                self.state_space.append(state)

        self.action_spae = [0, 1, 2, 3]

        def reset(self):
            self.agent_pos = {'y':0, 'x': 0}
            self.state = np.zeros([4,4])
            self.state[self.goal_pos['y'], self.goal_pos['x']] = -1
            self.state[self.agent_pos['y'], self.agent_pos['x']] = 1

            return self.state

        ##  Action을 받아 state가 전이되는 환경
        def step(self, action):
            ## Update environmental variables
            ## up
            if actioin == 0:
                ## 'y' should be decreased by 1 or stay the same when it is at the top
                ## row
                self.agent_pos['y'] = max(
                    self.agent_pos['y'] - 1,
                    self.y_min
                )

            ## right
            elif action == 1:
                ##  'x' should be increased by 1 or stay the same when it is at the
                ## most right column
                self.agent_pos['x'] = min(
                    self.agent_pos['x'] + 1,
                    self.x_max
                )

            ## down
            elif action == 2:
                ## 'y' should be increase by 1 or stay the same when it is at the bottom row
                self.agent_pos['y'] = min(
                    self.agent_pos['y'] + 1,
                    self.y_max
                )

            ## left
            elif action == 3:
                # 'x' should be decreased by 1 or stay the same when it is at the most
                ## left column
                self.agent_pos['x'] = max(
                    self.agent_pos['x'] - 1,
                    self.x_min
                    )
            else:
                assert False, "Invalid action value was fed to step."

            ## Make a next state after transition
            prev_state = self.state
            self.state = np.zeros([4,4])
            self.state[self.goal_pos['y'], self.goal_pos['x']] = -1
            self.state[self.agent_pos['y'], self.agent_pos['x']] = 1

            done = False
            if self.agent_pos == self.goal_pos:
                done = True

            reward = self.reward(prev_state, action, self.state)
            return reward, self.state, done

        def reward(self, s, a, s_next):
            reward = 0



