from RLV.torch_rlv.executor.experiment import Experiment


def run_torch_rlv(config):
    test_experiment = Experiment(config)
    test_experiment.run_experiment()


if __name__ == '__main__':
    test_experiment = Experiment({
        'env_name': 'acrobot_continuous',
        'algo_name': 'sac'
    })
    test_experiment.run_experiment()
