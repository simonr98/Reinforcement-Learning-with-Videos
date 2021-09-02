from RLV.torch_rlv.algorithms.sac.sac import SAC
from RLV.torch_rlv.environments.utils import get_environment

env = get_environment('acrobot_continuous')
model = SAC.load("sac_models/trained_for_3000")

obs = env.reset()

for i in range(1000):
    action, state_ = model.predict(obs)

    env.render()

    next_obs, reward, done, _ = env.step(action)

    if not done:
        obs = next_obs
    else:
        obs = env.reset()



