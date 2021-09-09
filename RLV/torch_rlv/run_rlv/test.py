import gym
from RLV.torch_rlv.algorithms.sac.sac import SAC
from gym_framework.mujoco_envs.push_env.push_env import PushMocapCtrl
from RLV.torch_rlv.environments.utils import get_environment



#env = PushMocapCtrl(render=True, max_steps=2000, nsubsteps=12, random_env=False)

env = get_environment('acrobot_continuous')

model = SAC.load("../data/acrobot_continuous_data/acrobot_sac_trained_for_1000000_steps")

print(model.wandb_config)


obs = env.reset()

for i in range(50000):
    action, state_ = model.predict(obs)

    env.render()

    next_obs, reward, done, _ = env.step(action)

    if not done:
        obs = next_obs
    else:
        obs = env.reset()



