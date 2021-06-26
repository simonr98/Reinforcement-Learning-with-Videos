## FetchPush von Gym

# import gym
# env = gym.make('FetchPush-v1')
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         observation = observation['observation']
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()

import multiworld
import gym
from multiworld.core.flat_goal_env import FlatGoalEnv

from gym import envs
import custom_envs
envids = [spec.id for spec in envs.registry.all()]
for envid in sorted(envids):
    print(envid)

multiworld.register_all_envs()

base_env = gym.make('SawyerReachXYZEnv-v1')
env = FlatGoalEnv(base_env)

observation = env.reset()
for i_episode in range(60):
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation_, reward, done, info = env.step(action)
        observation = observation_
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()