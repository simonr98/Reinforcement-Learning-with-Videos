import gym
from gym_framework.mujoco_envs.push_env.push_env import PushMocapCtrl

def get_environment(name):
    if name == "acrobot_continuous":
        return gym.make('AcrobotContinuous100-v1')

    if name == "visual_pusher":
        return PushMocapCtrl(render=False, max_steps=500, nsubsteps=12, random_env=False)