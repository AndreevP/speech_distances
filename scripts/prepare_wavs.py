import argparse
import sys
import os
sys.path.append('.')
from speech_distances.models import load_model
from tqdm import tqdm
import numpy as np
import torchaudio
import torch
from torch import nn
import librosa


class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251
    frequency_mask_max_percentage: float = 0.15
    time_mask_max_percentage: float = 0.0
    mask_probability: float = 0.3


class MelSpectrogram(nn.Module):

    def __init__(self, config=MelSpectrogramConfig):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = config.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio):
        mel = self.mel_spectrogram(audio) \
                .clamp_(min=1e-5) \
                .log_()
        return mel


def parse_args():
    parser = argparse.ArgumentParser(description='inference wavenets')
    parser.add_argument('--model_name',
        help='name of the model, from load_model, supported: melgan, wavenet')
    parser.add_argument('--folder_in', help='folder to load wavs')
    parser.add_argument('--folder_out', help='folder to take save generated wavs')
    args = parser.parse_args()
    return args


def load_wav(wav_file, target_sample_rate):
    wav, sample_rate = torchaudio.load(wav_file)
    wav = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(wav)
    return wav


def infer_melgan(args):
    target_sample_rate = 22050
    model = load_model(args.model_name)
    files = [item for item in os.listdir(args.folder_in) if item.endswith('wav')]
    for idx, audio in enumerate(files):
        wav_path = os.path.join(args.folder_in, audio)
        wav = load_wav(wav_path, target_sample_rate)
        with torch.no_grad():
            mel = model(wav)
            waveform = model.inverse(mel)
        path = os.path.join(args.folder_out, audio)
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torchaudio.save(path, waveform.cpu(), target_sample_rate)

def infer_waveglow(args):
    target_sample_rate = 22050
    n_mels = 80
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.model_name, device=device)
    meller = MelSpectrogram().to(device)
    files = [item for item in os.listdir(args.folder_in) if item.endswith('wav')]
    for idx, audio in enumerate(files):
        wav_path = os.path.join(args.folder_in, audio)
        wav = load_wav(wav_path, target_sample_rate).to(device)
        mel = meller(wav)
        if mel.shape[1] != n_mels:
            mel = mel.permute(0, 2, 1)
        waveform = model.inference(mel)
        path = os.path.join(args.folder_out, audio)
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torchaudio.save(path, waveform.cpu(), target_sample_rate)


def infer_wavenet(args):
    import sys
    sys.path.append('thirdparty/wavenet_vocoder')

    from train import build_model
    from synthesis import wavegen
    from tqdm import tqdm
    target_sample_rate = 22050

    hparams, model = load_model(args.model_name)
    meller = MelSpectrogram()
    files = [item for item in os.listdir(args.folder_in) if item.endswith('wav')]
    for idx, audio in enumerate(files):
          wav_path = os.path.join(args.folder_in, audio)
          wav = load_wav(wav_path, target_sample_rate)
          c = meller(wav)[0]
          if c.shape[1] != hparams.num_mels:
              c = c.transpose(0, 1)
          # Range [0, 4] was used for training Tacotron2 but WaveNet vocoder assumes [0, 1]
          # c = np.interp(c, (0, 4), (0, 1))

          # Generate
          waveform = wavegen(model, c=c, fast=True, tqdm=tqdm)
          path = os.path.join(args.folder_out, audio)
          folder = os.path.dirname(path)
          if not os.path.exists(folder):
              os.makedirs(folder)
          torchaudio.save(path, waveform, hparams.sample_rate)


if __name__ == '__main__':
    args = parse_args()
    if args.model_name == 'melgan':
        infer_melgan(args)
    elif args.model_name == 'wavenet':
        infer_wavenet(args)
    elif args.model_name == 'waveglow':
        infer_waveglow(args)
    else:
        raise ValueError(f"Model {args.model_name} not supported yet")
