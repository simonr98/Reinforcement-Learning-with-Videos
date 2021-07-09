import gym
import multiworld
from RLV.torch_rlv.environments import custom_envs
from multiworld.core.flat_goal_env import FlatGoalEnv
import experiment_envs


def get_environment(name):
    if name == "acrobot_continuous":
        multiworld.register_all_envs()
        return gym.make('AcrobotContinuous-v1')

    if name == "visual_pusher_gym":
        base_env = gym.make('FetchPush-v1')
        multiworld.register_all_envs()
        return gym.wrappers.FlattenObservation(base_env)

    if name == 'cart_pole':
        return gym.make('CartPole-v1')

    if name == 'pusher_2d':
        return gym.make('Pusher2dEnv-v1')

    if name == 'Humanoid-v2':
        return gym.make('Humanoid-v2')

    if name == "visual_pusher_multi_world":
        base_env = gym.make('SawyerReachXYZEnv-v1')
        multiworld.register_all_envs()
        return FlatGoalEnv(base_env, append_goal_to_obs=True)

    if name == "visual_door_opener_multi_world":
        base_env = gym.make('SawyerDoorHookResetFreeEnv-v0')
        multiworld.register_all_envs()
        return FlatGoalEnv(base_env, append_goal_to_obs=True)

