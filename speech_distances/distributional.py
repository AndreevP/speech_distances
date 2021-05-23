from abc import abstractmethod
from .metric import Metric
from .models import load_model
import torchaudio
import glob
import torch
import os
import tqdm


class DistributionalMetric(Metric):
    
    def __init__(self, path: str, reference_path: str, backbone: str,
                 sr: int=20050, sample_size: int=10000,
                 num_runs: int=1, window_size: int=320,
                 conditional: bool=True, use_cached: bool=True):
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
            use_cached (bool, optional): Try to reuse extracted features if possible?. Defaults to True.
        """
        
        assert window_size%160==0, "STFT step size is assumed to be equal to 160"
        
        backbone_name = backbone
        backbone = load_model(backbone, "cuda")
        super().__init__(path, reference_path=reference_path,
                         backbone=backbone, backbone_name=backbone_name,
                         sr=sr, sample_size=sample_size,
                         num_runs=num_runs, window_size=window_size,
                         conditional=conditional)
        
        self.features = self.extract_features(self.path, use_cached)
        self.reference_features = self.extract_features(self.reference_path, use_cached)
    
    @torch.no_grad()
    def extract_features(self, path, use_cached=True):
        """This method computes features from the backbone
           and saves them in a file within provided directory (caching)
           
        Args:
            use_cached (bool, optional): Defines whether to load cached features if there are any. Defaults to True.
        Returns:
            extracted features (torch.tensor)
        """
        path_to_features = os.path.isfile(os.path.join(path, self.backbone_name + "_features"))
        
        if os.path.isfile(path_to_features) and use_cached:
            return torch.load(path_to_features)
            
        required_sr = self.backbone.preprocessor._sample_rate
        files = sorted(glob.glob(os.path.join(path, "*.wav")))
        
        features = []
        for file in tqdm.tqdm(files, desc="Extracting features..."):
            wav, sr = torchaudio.load(file)
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=required_sr)(wav)
            wav = wav[..., :wav.shape[-1]//160 * 160]
            
            x, s = self.backbone.preprocessor(input_signal=wav.cuda(), length=torch.LongTensor([wav.shape[-1]]).cuda())
            n = int(self.window_size / 160 - 1) # window size in stft steps
            x = x[..., :s[0].cpu().item()//n*n].view(x.shape[0], x.shape[1], -1, n).transpose(-3, -2)
            x = x.reshape((-1, x.shape[-2], x.shape[-1]))
            y = self.backbone.encoder.forward(audio_signal=x, length=torch.LongTensor([x.shape[-1]]))[0]
            
            features.append(y.mean(-1).mean(0))            
            
        features = torch.stack(features)
        torch.save(features, path_to_features)
        return features
            
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