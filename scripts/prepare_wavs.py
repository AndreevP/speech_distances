import argparse
import sys
import os
sys.path.append('.')
from speech_distances.models import load_model
from tqdm import tqdm
import numpy as np
import torchaudio
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='inference wavenets')
    parser.add_argument('--model_name',
        help='name of the model, from load_model, supported: melgan, wavenet')
    parser.add_argument('--folder_in', help='folder to load wavs')
    parser.add_argument('--folder_out', help='folder to take save generated wavs')
    args = parser.parse_args()
    return args


def convert_to_mel(wav_file, target_sample_rate, n_mels):
    wav, sample_rate = torchaudio.load(wav_file)
    wav = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(wav)
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=target_sample_rate, n_mels=n_mels)(wav)
    return mel


def infer_melgan(args):
    target_sample_rate = 22050
    model = load_model(args.model_name)
    files = [item for item in os.listdir(args.folder_in) if item.endswith('wav')]
    for idx, audio in enumerate(files):
        wav_path = os.path.join(args.folder_in, audio)
        wav, sample_rate = torchaudio.load(wav_path)
        wav = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(wav)
        with torch.no_grad():
            mel = model(wav)
            waveform = model.inverse(mel)
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

    hparams, model = load_model(args.model_name)
    files = [item for item in os.listdir(args.folder_in) if item.endswith('wav')]
    for idx, audio in enumerate(files):
          wav_path = os.path.join(args.folder_in, audio)
          c = convert_to_mel(wav_path, hparams.sample_rate, hparams.num_mels)[0]
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
    else:
        raise ValueError(f"Model {args.model_name} not supported yet")
