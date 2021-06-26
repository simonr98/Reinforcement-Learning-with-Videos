from .acrobot_env import get_acrobot
import gym
from RLV.torch_rlv.environments import custom_envs


def get_environment(name):
    if name == "acrobot_continuous":
        return gym.make('AcrobotContinuous-v1')
    else:
        print('acrobot not available, created InvertedPendulumBulletEnv-v0')
        return gym.make('InvertedPendulumBulletEnv-v0')


        #base_env = gym.make('FetchPush-v1')
        #env = gym.wrappers.FlattenObservation(base_env)