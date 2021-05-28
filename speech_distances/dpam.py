import torch
import torch.nn.functional as F
from nemo.collections.tts.losses.stftlosses import MultiResolutionSTFTLoss
from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types.elements import AudioSignal, LossType, NormalDistributionSamplesType, VoidType
from nemo.core.neural_types.neural_type import NeuralType
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import glob
import os

from PerceptualAudio_Pytorch.models import JNDnet

from .metric import Metric


class MetricDPAM(Metric):

    def __init__(self, path: str, reference_path: str, batch_size: int, device: str):
        super().__init__(path)
        model = JNDnet(nconv=14, 
                       nchan=16, 
                       dist_dp=0., 
                       dist_act='no', 
                       ndim=[8,4], 
                       classif_dp=0.2, 
                       classif_BN=2, 
                       classif_act='sig', 
                       dev=device)
        state = torch.load('../PerceptualAudio_Pytorch/pretrained/dataset_combined_linear.pth', 
                           map_location="cpu")['state']
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        self.dpam_model = model
        self.reference_path = reference_path
        self.batch_size = batch_size

    def distance(self, X, Y):
        with torch.no_grad():
            distances = self.dpam_model.model_dist.forward(X, Y)
        return distances.mean()

    def calculate_metric(self):
        files = sorted(glob.glob(os.path.join(self.path, "*.wav")))
        ref_files = sorted(glob.glob(os.path.join(self.reference_path, "*.wav")))
        distances = []
        for i in range(0, len(files), self.batch_size):
            wavs = []
            for file in files[i:i+self.batch_size]:
                wav, sr = torchaudio.load(file)
                wavs.append(torch.squeeze(wav))
            for file in ref_files[i:i+self.batch_size]:
                wav, sr = torchaudio.load(file)
                wavs.append(torch.squeeze(wav))
            wavs = pad_sequence(wavs, batch_first=True, padding_value=0.0)
            split = wavs.shape[0] // 2
            distances.append(self.distance(wavs[:split], wavs[split:]))
        return torch.stack(distances).mean()


class CustomLoss(Loss):
    """A Loss module that unites UniGlow loss with preceptual loss based on DPAM"""

    def __init__(self, seq_len, stft_loss_coef=0.1, dpam_loss_coef=1.0):
        super(CustomLoss, self).__init__()
        self.stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240], window="hann_window"
        )
        self.seq_len = seq_len
        self.stft_loss_coef = stft_loss_coef
        self.dpam_loss_coef = dpam_loss_coef
        model = JNDnet(nconv=14, 
                       nchan=16, 
                       dist_dp=0., 
                       dist_act='no', 
                       ndim=[8,4], 
                       classif_dp=0.2, 
                       classif_BN=2, 
                       classif_act='sig', 
                       dev='cuda')
        state = torch.load('../PerceptualAudio_Pytorch/pretrained/dataset_combined_linear.pth', 
                           map_location="cpu")['state']
        model.load_state_dict(state)
        model.to('cuda')
        model.eval()
        self.dpam_model = model

    @property
    def input_types(self):
        return {
            "z": NeuralType(('B', 'flowgroup', 'T'), NormalDistributionSamplesType()),
            "logdet": NeuralType(elements_type=VoidType()),
            "gt_audio": NeuralType(('B', 'T'), AudioSignal()),
            "predicted_audio": NeuralType(('B', 'T'), AudioSignal()),
            "sigma": NeuralType(optional=True),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, *, z, logdet, gt_audio, predicted_audio, sigma=1.0):
        nll_loss = torch.sum(z * z) / (2 * sigma * sigma) - logdet
        nll_loss = nll_loss / (z.size(0) * z.size(1) * z.size(2))
        
        shape_diff = self.seq_len - predicted_audio.shape[1]
        predicted_audio = F.pad(predicted_audio, (0, shape_diff), mode='constant', value=0)
            
        sc_loss, mag_loss = self.stft_loss(x=predicted_audio, y=gt_audio)
        sc_loss = sum(sc_loss) / len(sc_loss)
        mag_loss = sum(mag_loss) / len(mag_loss)
        stft_loss = sc_loss + mag_loss
        dpam_loss = torch.mean(model.model_dist.forward(predicted_audio, gt_audio))
#         print('dpam_component:', dpam_loss.item())
        loss = nll_loss + self.stft_loss_coef * stft_loss + self.dpam_loss_coef * dpam_loss
        return loss