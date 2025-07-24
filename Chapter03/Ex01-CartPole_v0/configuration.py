## Configuration 
from utils import AttrDict

## 학습에서 쓰이는 각종 parameter 및 hyper-parameter
config = AttrDict(
    gamma = 0.99,               ##config.gamma = 0.99
    lr=5e-5,                    ##learning_rate
    batch_size=64,
    hidden_size=128,
    replay_capacity=10000,      ##Replay memory size
    replay_init_ratio=0.3,      ##Replay memory를 몇 percente 채우고 시작할지 
    train_env_step=200000,      ##얼마나 많은 샘플을 볼 것인가?
    target_update_period=100,   ##target network를 업데이트 (100번의 end state를 돌 때마다)
    eps_init=1.0,               
    eps_final=0.05,             ##Eps의 최종 수렴값
    eps_decrease_step=100000,   ##Eps의 값을 10만 스텝동안 선형적으로 감소
    num_eval_episode=20,        ##Evaluation을 실행해서 학습이 잘되는지 확인. 
                                ##20개의 episode를 실행해서 결과의 평균으로 학습을 평가
    eval_period=500,            ##500번의 env.step을 실행하고 그리고 나서 evaluation 진행
)