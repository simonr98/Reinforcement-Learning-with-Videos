from RLV.torch_rlv.algorithms.sac.sac import SAC
from RLV.torch_rlv.environments.utils import get_environment

env = get_environment('visual_pusher')
model = SAC.load("../output/sac_models/pusher/trained_for_1000000.zip")

print(model.env_name)
print(model.wandb_config)

obs = env.reset()

for i in range(5000):
    action, state_ = model.predict(obs)

    # env.render()

    next_obs, reward, done, _ = env.step(action)

    if not done:
        obs = next_obs
    else:
        obs = env.reset()



