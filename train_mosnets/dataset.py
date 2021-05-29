import pandas as pd
import scipy.signal
import librosa
import os
import copy
import json

import numpy as np
import torch


class Dataset:
    def __getitem__(self, *args):
        args = args[0]
        if type(args) is tuple:
            assert(len(args) <= 2)
            assert(len(args) > 0)

            return self.getitem(args[0], args[1])
        else:
            if not hasattr(self, 'subset'):
                sliced_dataset = copy.copy(self)
                sliced_dataset.subset = args
                return sliced_dataset

            else:
                return self.getitem(self.subset, args)

    def __len__(self):
        assert(hasattr(self, 'subset'))
        return self.getlen(self.subset)


class VCC2018Dataset(Dataset):
    def __init__(self, list_path, data_path, n_valid=3000, n_test=4000, sr=16000):
        self.list_path = list_path
        self.data_path = data_path
        self.sr = sr
        self.mos_list = pd.read_csv(self.list_path, header=None)
        
        self.samples = {
            'train': self.mos_list[:-n_test],
            'valid': self.mos_list[-n_valid-n_test:-n_test],
            'test': self.mos_list[-n_test:]
        }
        
    def getlen(self, subset):
        return len(self.samples[subset])
    
    def getitem(self, subset, idx):
        sample_name, mos = self.samples[subset].iloc[idx]
        
        signal,sr = librosa.load(os.path.join(self.data_path, sample_name), sr=self.sr)
        spec = np.abs(librosa.stft(signal, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming).T).astype(np.float32) # [time, 257]
        
        return torch.as_tensor(spec).unsqueeze(0).unsqueeze(1).float(), torch.as_tensor([mos]).float()


class VCC18DatasetMBNet(Dataset):
    def __init__(self, data_path, csv_dir='data/', sr=16000):
        self.data_path = data_path
        self.sr = sr
        self.samples = {
            subset: pd.read_csv(os.path.join(csv_dir, 'mbnet_' + subset + '.csv'))
            for subset in ['train', 'valid', 'test']
        }
        self.judges = json.load(open(os.path.join(csv_dir, 'judges.json'), 'r'))
        
    def getlen(self, subset):
        return len(self.samples[subset])
        
    def getitem(self, subset, idx):
        sample_name, mean_mos, real_mos, judge = self.samples[subset].iloc[idx]
        
        signal,sr = librosa.load(os.path.join(self.data_path, sample_name), sr=self.sr)
        spec = np.abs(librosa.stft(signal, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming).T).astype(np.float32) # [time, 257]
        
        return torch.as_tensor(spec).unsqueeze(0).unsqueeze(1).float(), (torch.as_tensor([mean_mos]).float(), \
                    torch.as_tensor([real_mos]).float(), torch.as_tensor([self.judges[judge]]).long())
    
from deep_speech import  DeepSpeechInputFeatureExtractor

class VCC2018DatasetDeepSpeech(VCC2018Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preproc =  DeepSpeechInputFeatureExtractor()
        self.sr = 16000
    
    def getitem(self, subset, idx):
        sample_name, mos = self.samples[subset].iloc[idx]
        signal,sr = librosa.load(os.path.join(self.data_path, sample_name), sr=self.sr)
        
        return self.preproc(torch.as_tensor(signal).unsqueeze(0))[0][0], torch.as_tensor([mos]).float()
    
    
class VCC2018DatasetWav2Vec2(VCC2018Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sr = 16000
    
    def getitem(self, subset, idx):
        sample_name, mos = self.samples[subset].iloc[idx]
        signal,sr = librosa.load(os.path.join(self.data_path, sample_name), sr=self.sr)
        
        return signal, torch.as_tensor([mos]).float()
    
    

class VCC2018DatasetNoPreporocess(VCC2018Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.preproc =  DeepSpeechInputFeatureExtractor()
    
    def getitem(self, subset, idx):
        sample_name, mos = self.samples[subset].iloc[idx]
        signal, sr = librosa.load(os.path.join(self.data_path, sample_name), sr=self.sr)
        
        return torch.as_tensor(signal), torch.as_tensor([mos]).float()   
    
    
    
def collate_fn_zeros(samples):
    specgrams, mos = zip(*samples)
    max_len = max([s.shape[2] for s in specgrams])
    
    batch = torch.zeros(len(specgrams), 1, max_len, specgrams[0].shape[3])
    for i in range(len(batch)):
        cur_len = specgrams[i].shape[2]
        batch[i, 0, :cur_len, :] = specgrams[i][0,0]
        
    return batch, torch.cat(mos, dim=0)


def collate_fn_lenth(samples):
    wavs, mos = zip(*samples)
    length = [s.shape[0] for s in wavs]
    max_len = max(length)
    length = torch.tensor(length).to(wavs[0].device)
    
    batch = torch.zeros(len(wavs), max_len)
    for i in range(len(batch)):
        cur_len = wavs[i].shape[0]
        batch[i, :cur_len] = wavs[i]
        
    return (batch, length), torch.cat(mos, dim=0)

def collate_fn_reppad(samples):
    specgrams, mos = zip(*samples)
    max_len = max([s.shape[2] for s in specgrams])
    
    batch = torch.zeros(len(specgrams), 1, max_len, specgrams[0].shape[3])
    for i in range(len(batch)):
        cur_len = specgrams[i].shape[2]
        n_reps = int(np.ceil(max_len / cur_len))
        batch[i, 0, :, :] = specgrams[i][0,0].repeat(n_reps, 1)[:max_len]
        
    return batch, torch.cat(mos, dim=0)

def collate_fn_reppad_mbnet(samples):
    specgrams, mos = zip(*samples)
    max_len = max([s.shape[2] for s in specgrams])
    
    batch = torch.zeros(len(specgrams), 1, max_len, specgrams[0].shape[3])
    for i in range(len(batch)):
        cur_len = specgrams[i].shape[2]
        n_reps = int(np.ceil(max_len / cur_len))
        batch[i, 0, :, :] = specgrams[i][0,0].repeat(n_reps, 1)[:max_len]
    
    mean_mos, real_mos, judges = list(zip(*mos))
    return (batch, torch.cat(judges, dim=0)), (torch.cat(mean_mos, dim=0), torch.cat(real_mos, dim=0))


if __name__ == '__main__':
    ds = VCC2018Dataset(list_path='mos_list.txt', data_path='data/wav/')
    ds.getitem('train', 5)[0].shape