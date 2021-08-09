import gym
import experiment_envs
from RLV.torch_rlv.environments import custom_envs
from stable_baselines3.common.env_util import make_vec_env
from gym_framework.mujoco_envs.push_env.push_env import PushMocapCtrl


def get_environment(name):
    if name == "acrobot_continuous":
        return gym.make('AcrobotContinuous100-v1')

    if name == "pick_and_place":
        return PushMocapCtrl(render=True, max_steps=500, nsubsteps=12, random_env=False)

    if name == "visual_pusher_gym":
        base_env = gym.make('FetchPush-v1')
        return gym.wrappers.FlattenObservation(base_env)

    if name == 'cart_pole':
        return gym.make('CartPole-v1')



