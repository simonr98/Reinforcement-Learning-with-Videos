from RLV.torch_rlv.algorithms.sac.softactorcritic import SoftActorCritic
from RLV.torch_rlv.algorithms.rlv.rlv import RLV


def init_algorithm(alg_name, experiment):
    if alg_name == "sac":
        return SoftActorCritic(policy='MlpPolicy', env_name=experiment.env_name, env=experiment.env,
                               verbose=1, learning_starts=1000)
    if alg_name == "rlv":
        return RLV(warmup_steps=500, beta_inverse_model=0.0003, env_name=experiment.env_name,
                   policy=experiment.policy, env=experiment.env, learning_rate=0.0003, buffer_size=1000000)
