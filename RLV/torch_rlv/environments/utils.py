import gym
from RLV.torch_rlv.environments import custom_envs
from stable_baselines3.common.env_util import make_vec_env
from gym_framework.mujoco_envs.push_env.push_env import PushMocapCtrl
from gym_framework.mujoco_envs.pick_and_place_env.pick_and_place_env import PickAndPlaceMocapCtrl
from gym_framework.mujoco_envs.reach_env.reach_env import ReachEnvJointVelCtrl


def get_environment(name):
    if name == "acrobot_continuous":
        return gym.make('AcrobotContinuous100-v1')

    if name == "pick_and_place":
        return PickAndPlaceMocapCtrl(render=True, max_steps=500, nsubsteps=12, random_env=False)

    if name == "push":
        return PushMocapCtrl(render=True, max_steps=500, nsubsteps=12, random_env=False)

    if name == "reach":
        return ReachEnvJointVelCtrl(render=True, max_steps=250, nsubsteps=12, random_env=False)

    if name == "visual_pusher_gym":
        base_env = gym.make('FetchPush-v1')
        return gym.wrappers.FlattenObservation(base_env)

    if name == 'cart_pole':
        return gym.make('CartPole-v1')



