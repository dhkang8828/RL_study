## Utils.py
## 학습에 사용되는 각종 class

import gym
import gym.spaces
import numpy as np

def create_env(config):
    env = gym.make("CartPole-v1")
    env = RewardShapingWrapper(env)
    env = TimeStepAppendWrapper(env)
    return env

## Reward에 0.1을 곱한다.
## reward(1)/(1-gamma) = 100
## 100의 값으로 Neural Network를 학습하는 것은 안정성이 떨어진다.
class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        
    def step(self, action):
        s_next, r, done, info = self.env.step(action)
        r = 0.1 * r
        return s_next, r, done, info

## 4개 짜리 state에다가 time-step이 얼마나 지났는지 추가하여 
## 총 5개의 state를 구성
## Agent가 인식하는 환경의 state에 대해 더 명확하게 하고자 추가 
## 500 Step이 되면 게임이 끝나는데, 500Step에서 균형을 유지하며 끝내는 것과,
## 넘어지며 끝내는 것은 다르다.
## 500 Step기준으로 얼마나 시간을 소모했는가?를 표현 
class TimeStepAppendWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        self.step_count = 0
        self.observation_space = gym.spaces.Box(
            np.concatenate([self.env.observation_space.low, [0]], axis=-1),     ##state가 가질 수 있는 값의 최소값 
            np.concatenate([self.env.observation_space.high, [5]], axis=-1),    ##state가 가질 수 있는 값의 최대값  (500 * 0.01 = 5)  
            (self.env.observation_space.shape[0] + 1, ),    ##State dimmension, 기존4개에서 1개 추가
            np.float32
        )
    
    def reset(self):
        self.step_count = 0
        s = self.env.reset()
        s = np.concatenate([s, [0.01 * self.step_count]], axis=-1)
        return s
    
    def step(self, action):
        s_next, r, done, info = self.env.step(action)
        self.step_count += 1
        s_next = np.concatenate([s_next, [0.01 * self.step_count]], axis=-1)
        return s_next, r, done, info
    
class AttrDict(dict):
    __setattr__ = dict.__setitem__
    
    def __getattribute__(self, item):
        if item in self:
            return self[item]
        else:
            return super().__getattribute__(item)
        
    @classmethod
    def from_nested_dics(cls, data):
        if not isinstance(data, dict):
            return data
        else:
            return cls({key:cls.from_nested_dics(data[key]) for key in data})

if __name__ == "__main__":
    print("Creating environment...")