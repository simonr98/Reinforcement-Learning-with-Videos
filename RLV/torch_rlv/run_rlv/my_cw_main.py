from cw2.cw_data import cw_logging
from cw2 import experiment, cluster_work, cw_error
from RLV.torch_rlv.run_rlv.experiment import Experiment
from RLV.torch_rlv.scripts.save_data_of_model import DatasetCreator

class CustomExperiment(experiment.AbstractExperiment):

    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        pass

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        my_config = config.get("params")  # move into custom config. This is now everything that you specified

        # creator = DatasetCreator(env_name='visual_pusher', num_steps=50000, max_length_episode=200,
        #                          model_path="../data/visual_pusher_data/478666_sac_trained_for_500000_steps")
        # creator.save_data_of_model()

        my_experiment = Experiment(my_config)
        my_experiment.run_experiment()


    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        # Skip for Quickguide
        pass


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(CustomExperiment)
    cw.run()