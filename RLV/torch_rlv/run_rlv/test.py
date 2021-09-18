import gym
from RLV.torch_rlv.algorithms.sac.sac import SAC
from gym_framework.mujoco_envs.push_env.push_env import PushMocapCtrl
from RLV.torch_rlv.environments.utils import get_environment
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from fastgrab import screenshot
import torch as T



env = PushMocapCtrl(render=True, max_steps=300, nsubsteps=12, random_env=False)

# env = get_environment('acrobot_continuous')

# model = SAC.load("../data/visual_pusher_data/527735_sac_trained_for_500000_steps")

model = SAC.load("../data/visual_pusher_data/478666_sac_trained_for_500000_steps")

print(model.wandb_config)

obs = env.reset()

# img = env.render()
# imgplot = plt.imshow(img)
# plt.show()

print(env.action_space)

for i in range(50000):
    action, state_ = model.predict(obs)
    print(action)
    #.render
    next_obs, reward, done, _ = env.step(action)

    if not done:
        obs = next_obs
    else:
        # img = env.render()
        # imgplot = plt.imshow(img)
        # plt.show()
        obs = env.reset()
        obs = T.from_numpy(obs)



