from RLV.torch_rlv.environments.utils import get_environment
from RLV.torch_rlv.algorithms.utils import init_algorithm


class Experiment:
    def __init__(self, config):
        self.config = config
        self.env_name = config['env_name']
        self.env = get_environment(self.env_name)
        self.algo_name = config['algo_name']
        self.policy = config['policy']
        self.device = config['device']
        self.lr_inverse_model = config['lr_inverse_model']
        self.wandb_log = config['wandb_log']
        self.acrobot_paper_data = config['acrobot_paper_data']
        self.lr_sac = config['lr_sac']
        self.total_steps = config['total_steps']
        self.buffer_size = config['buffer_size']
        self.learning_starts = config['learning_starts']
        self.batch_size = config['batch_size']
        self.tau = config['tau']
        self.gamma = config['gamma']
        self.train_freq = config['train_freq']
        self.gradient_steps = config['gradient_steps']
        self.project_name = config['project_name']
        self.run_name = config['run_name']
        self.log_dir = config['log_dir']

    def run_experiment(self):
        algorithm = init_algorithm(self.algo_name, self)
        algorithm.run()

