from abc import abstractmethod
from .metric import Metric
from .models import load_model


class DistributionalMetric(Metric):
    
    def __init__(self, path: str, reference_path: str, backbone: str,
                 sr: int=20050, sample_size: int=10000,
                 num_runs: int=1, window_size: int=480,
                 conditional: bool=True):
        """
        Parent class for distances based on ditrubutional metrics, such as Frechet distance and MMD
        Args:
            path (str): path to .wav files to be evaluated 
            reference_path (str): path to reference .wav files to be compared with 
            backbone (str): model to be used as a feature extractor
            sr (int, optional): Sampling rate. Defaults to 20050.
            sample_size (int, optional): Number of files to be used for evaluation. Defaults to 10000.
            num_runs (int, optional): Number of runs with different subsets of files for computation of mean and std. Defaults to 1.
            window_size (int, optional): Number of timesteps within one window for feature computation
                                         (the features are averaged for all windows). Defaults to 480.
            conditional (bool, optional): Defines whether to compute conditional version of the distance of not. Defaults to True.
        """
        backbone = load_model(backbone)
        super().__init__(path, reference_path=reference_path,
                         backbone=backbone, sr=sr, sample_size=sample_size,
                         num_runs=num_runs, window_size=window_size,
                         conditional=conditional)

    def compute_statistics(self, use_cached=True):
        """This method computes features and saves them as .npy file within directory (caching)
           and assigns corresponding attributes of self
        Args:
            use_cached (bool, optional): Defines whether to load cached features if there are any. Defaults to True.
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_metric(self, use_cached=True):
        pass


class FrechetDistance(DistributionalMetric):
    """Class for computation of Frechet Distances 
    """
    def calculate_metric(self, use_cached=True):
        raise NotImplementedError

class MMD(DistributionalMetric):
    """Class for computation of Maximum Mean Discrepancies
    """
    def calculate_metric(self, use_cached=True):
        raise NotImplementedError