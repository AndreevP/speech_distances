from abc import abstractmethod
from .metric import Metric
from .models import load_model
import torchaudio
import glob
import torch
import os
import tqdm
from scipy.linalg import sqrtm


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
        if window_size != None:
            assert window_size%160==0, "STFT step size is assumed to be equal to 160"
        
        backbone_name = backbone
        backbone = load_model(backbone, "cuda")
        super().__init__(path, reference_path=reference_path,
                         backbone=backbone, backbone_name=backbone_name,
                         sr=sr, sample_size=sample_size,
                         num_runs=num_runs, window_size=window_size,
                         conditional=conditional)
        
        self.conditional = conditional
        self.num_runs = num_runs
        self.sample_size = sample_size
        
        self.features = self.extract_features(self.path, use_cached).cpu()
        self.reference_features = self.extract_features(self.reference_path, use_cached).cpu()
    
    @torch.no_grad()
    def extract_features(self, path, use_cached=True):
        """This method computes features from the backbone
           and saves them in a file within provided directory (caching)
           
        Args:
            use_cached (bool, optional): Defines whether to load cached features if there are any. Defaults to True.
        Returns:
            extracted features (torch.tensor)
        """
        path_to_features = os.path.join(path, self.backbone_name + f"window{self.window_size}" + "_features")
        
        if os.path.isfile(path_to_features) and use_cached:
            return torch.load(path_to_features)
            
        required_sr = self.backbone.preprocessor._sample_rate
        resample = torchaudio.transforms.Resample(orig_freq=self.sr, new_freq=required_sr)
        files = sorted(glob.glob(os.path.join(path, "*.wav")))
        
        features = []
        for file in tqdm.tqdm(files, desc="Extracting features..."):
            wav, sr = torchaudio.load(file)
            wav = resample(wav)
            if self.window_size != None:
                wav = wav[..., :wav.shape[-1]//self.window_size * self.window_size]
            
            if not ("wav2vec2" in self.backbone_name):
                x, s = self.backbone.preprocessor(input_signal=wav.cuda(), length=torch.LongTensor([wav.shape[-1]]).cuda())
                if self.window_size != None:
                    n = int(self.window_size / 160 - 1) # window size in stft steps
                    x = x[..., :s[0].cpu().item()//n*n].view(x.shape[0], x.shape[1], -1, n).transpose(-3, -2)
                    x = x.reshape((-1, x.shape[-2], x.shape[-1]))
                y = self.backbone.encoder.forward(audio_signal=x,
                                                  length=torch.LongTensor([x.shape[-1]]))[0]
            else:
                wav = wav[None, :]
                if self.window_size != None:
                    wav = wav.view(1, -1, self.window_size)
                    wav = wav.reshape((-1, 1, self.window_size))
                y = self.backbone(wav)
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
        """
        calculates mean and std of the FID evaluated self.num_runs times on pairs of random samples from
        (self.features, self.reference_features). Samples are similarly indexed if self.conditional ==  True and
        independently different otherwise
        """
        n, _ = self.features.shape
        metric_results = []

        for _ in range(self.num_runs):

            if self.conditional:
                permutation = torch.randperm(n)
                mask = permutation[:self.sample_size]
                mask_ref = mask
            else:
                permutation = torch.randperm(n)
                mask = permutation[:self.sample_size]
                permutation_ref = torch.randperm(n)
                mask_ref = permutation_ref[:self.sample_size]

            sample_features = self.features[mask]
            sample_reference_features = self.reference_features[mask_ref]
            metric = self.distance(sample_features, sample_reference_features)
            metric_results.append(metric)

        metric_results = torch.tensor(metric_results)
        return metric_results.mean(), metric_results.std()

    @staticmethod
    def distance(X, Y):
        n, d = X.shape
        mean_X = X.mean(0).reshape((1, d))
        mean_Y = Y.mean(0).reshape((1, d))
        unbiased_X = X - torch.ones((n, 1)) @ mean_X
        unbiased_Y = Y - torch.ones((n, 1)) @ mean_Y
        cov_X = 1 / (n - 1) * unbiased_X.T @ unbiased_X
        cov_Y = 1 / (n - 1) * unbiased_Y.T @ unbiased_Y
        bias_term = torch.sum((mean_X - mean_Y) ** 2)
        cross_var = sqrtm(cov_X * cov_Y)

        var_term = torch.trace(cov_X + cov_Y - 2 * cross_var)
        return torch.sqrt(bias_term + var_term)


class MMD(FrechetDistance):
    """Class for computation of Maximum Mean Discrepancies
    """

    @staticmethod
    def distance(X, Y):

        def kernel(x, y):
            return (x.T @ y / x.shape[1] + 1) ** 3

        n, d = X.shape
        kernel_X = (kernel(X, X) * (1 - torch.eye(d))).sum() / (n * (n - 1))
        kernel_Y = (kernel(Y, Y) * (1 - torch.eye(d))).sum() / (n * (n - 1))
        kernel_XY = kernel(X, Y).sum() / n ** 2
        return torch.sqrt(kernel_X + kernel_Y + kernel_XY)
    
