import gym
from RLV.torch_rlv.algorithms.sac.sac import SAC
from gym_framework.mujoco_envs.push_env.push_env import PushMocapCtrl



env = PushMocapCtrl(render=True, max_steps=2000, nsubsteps=12, random_env=False)

model = SAC.load("../data/visual_pusher_data/trained_for_1000000.zip")

print(model.wandb_config)


obs = env.reset()

for i in range(50000):
    action, state_ = model.predict(obs)

    # env.render()

    next_obs, reward, done, _ = env.step(action)

    if not done:
        obs = next_obs
    else:
        obs = env.reset()



