from cw2.cw_data import cw_logging
from cw2 import experiment, cluster_work, cw_error
from RLV.torch_rlv.run_rlv.experiment import Experiment

class CustomExperiment(experiment.AbstractExperiment):

    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        pass

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        my_config = config.get("params")  # move into custom config. This is now everything that you specified

        my_experiment = Experiment(my_config)
        my_experiment.run_experiment()


    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        # Skip for Quickguide
        pass


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(CustomExperiment)
    cw.run()