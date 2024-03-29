from RLV.torch_rlv.algorithms.sac.softactorcritic import SoftActorCritic
from RLV.torch_rlv.algorithms.rlv.rlv import RLV
from RLV.torch_rlv.algorithms.rlv.rlwithvideos import RlWithVideos

def init_algorithm(alg_name, experiment):
    if alg_name == "sac":
        return SoftActorCritic(policy='MlpPolicy', env_name=experiment.env_name, env=experiment.env,
                               verbose=1, learning_starts=1000, learning_rate=experiment.lr_sac,
                               buffer_size=experiment.buffer_size, device=experiment.device,
                               wandb_log=experiment.wandb_log, batch_size=experiment.batch_size,
                               gamma=experiment.gamma, tau=experiment.tau, train_freq=experiment.train_freq,
                               gradient_steps=experiment.gradient_steps, project_name=experiment.project_name,
                               run_name=experiment.run_name, log_dir=experiment.log_dir,
                               total_steps=experiment.total_steps, algo_name=experiment.algo_name)
    if alg_name == "rlv":
        return RlWithVideos(env_name=experiment.env_name, policy=experiment.policy, wandb_log=experiment.wandb_log,
                            learning_rate_inverse_model=experiment.lr_inverse_model, env=experiment.env,
                            learning_rate=experiment.lr_sac, buffer_size=experiment.buffer_size,
                            learning_starts=experiment.learning_starts, batch_size=experiment.batch_size,
                            gamma=experiment.gamma, tau=experiment.tau, train_freq=experiment.train_freq,
                            gradient_steps=experiment.gradient_steps, project_name=experiment.project_name,
                            run_name=experiment.run_name, acrobot_paper_data=experiment.acrobot_paper_data, verbose=1,
                            log_dir=experiment.log_dir,  total_steps=experiment.total_steps,
                            algo_name=experiment.algo_name, device=experiment.device)

