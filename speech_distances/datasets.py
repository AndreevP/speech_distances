"""This is a module for model load. If possible, use torchaudio build-in datasets.
"""
import torch
import torchaudio
import os


def load_dataset(name: str):
    data_root = os.path.join('./data', name)
    os.makedirs(data_root, exist_ok=True)
    if name.lower() == 'ljspeech':
        return torchaudio.datasets.LJSPEECH(root=data_root, download=True)
    else:
        raise NotImplementedError