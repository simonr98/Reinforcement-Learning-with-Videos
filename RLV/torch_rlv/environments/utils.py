from .acrobot_env import get_acrobot
import gym
from RLV.torch_rlv.environments import custom_envs


def get_environment(name):
    if name == "acrobot_continuous":
        return gym.make('AcrobotContinuous-v1')

    if name == "visual_pusher_gym":
        base_env = gym.make('FetchPush-v1')
        return gym.wrappers.FlattenObservation(base_env)

    if name == "visual_pusher_multiworld":
        base_env = gym.make('SawyerPush-v0')
        return FlatGoalEnv(base_env)

    if name == "visual_pusher_multi_world":
        base_env = gym.make('SawyerPush-v0')
        return FlatGoalEnv(base_env)

