from RLV.torch_rlv.executor.experiment import Experiment


def run_torch_rlv(config):
    test_experiment = Experiment(config)
    test_experiment.run_experiment()
