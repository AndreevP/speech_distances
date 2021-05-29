import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from deep_speech import DeepSpeech
import sys; sys.path.append('../../')
from speech_distances.speech_distances.models import load_model
from speech_distances.ss_models.synthesis_utils import make_preprocessor_trainable

def mosnet_conv_block(in_ch, out_ch):
    return [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, stride=(1, 3)),
        nn.ReLU()
    ]

def mbnet_conv_block(in_ch, out_ch):
    return [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, stride=(1,3)),
        nn.Dropout(0.3),
        nn.BatchNorm2d(out_ch),
        nn.ReLU()
    ]


class MOSNet(nn.Module):
    def __init__(self, block_type=mosnet_conv_block):
        super().__init__()

        self.layers = nn.Sequential(
            *block_type(1, 16),
            *block_type(16, 32),
            *block_type(32, 64),
            *block_type(64, 128)
        )

        self.bilstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True) # , dropout=0.3
        self.dense = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,1),
        )
        
    def forward(self, x):
        assert len(x.shape) == 4
        assert (x.shape[3] == 257) and (x.shape[1] == 1)
        
        bs, _, time, _ = x.shape
        return self.dense(self.bilstm(self.layers(x).permute(0, 2, 1, 3).reshape(bs, time, 512))[0])


class BiasNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_conv = nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=(1,3))
        self.conv_layers = nn.Sequential(
            nn.Conv2d(16+1, 32, kernel_size=3, padding=1, stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=(1,3)),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.bilstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, 
                              bidirectional=True)
        self.dense = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32,1),
        )
        self.judges = nn.Embedding(num_embeddings=300, embedding_dim=86) # Align n_features with strided STFT size

    def forward(self, x, judge_ids=None):
        batch, _, time, _ = x.shape

        x = self.in_conv(x)
        embs = self.judges(judge_ids)[:, None, None, :].repeat(1, 1, time, 1) # [B, 1, T, 257//3]
        x = self.conv_layers(torch.cat([x, embs], dim=1))
        x = x.permute(0, 2, 1, 3).reshape(batch, time, -1)
        x, (h, c) = self.bilstm(x)
        x = self.dense(x)
        return x


class MBNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_net = MOSNet(block_type=mbnet_conv_block)
        self.bias_net = BiasNet()

    def forward(self, x):
        (x, judge_ids) = x
        mean = self.mean_net(x)
        bias = self.bias_net(x, judge_ids)
        return mean, mean + bias


class MOSNetBatchNorm(MOSNet):
    def __init__(self):
        super().__init__(block_type=mbnet_conv_block)
        
        
class DeepSpeechMOS(nn.Module):
    def __init__(self, train_mode='all'):
        super().__init__()
        self.deepspeech = DeepSpeech.load_model('weights/an4_pretrained_v2.pth')
        self.train_mode = train_mode
        assert self.train_mode in ['all', 'fixconv', 'freeze']

        if self.train_mode == 'freeze':
            self.deepspeech.eval()
            
            for p in self.deepspeech.parameters():
                p.requires_grad_(False)
        elif self.train_mode == 'fixconv':
            self.deepspeech.conv.eval()

            for p in self.deepspeech.conv.parameters():
                p.requires_grad_(False)
        else:
            pass
        
        self.dense = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x, lengths = x
        feats, lengths = self.deepspeech(x, lengths) # [Len, batch, C]
        feats = feats.permute(1, 0, 2) # [batch, len, c]
        feats = self.dense(feats)
        
        batch, time, _ = feats.shape
        
        mask = torch.zeros(batch, time, device=feats.device, dtype=torch.float32)
        for i in range(batch):
            mask[i, :lengths[i]] = 1.0
        
        mos = (feats[:,:,0] * mask).sum(dim=1) / mask.sum(dim=1)
        return mos[:, None, None]
        
    def train(self, mode):
        super().train(mode)
        if self.train_mode == 'freeze':
            self.deepspeech.eval()
        elif self.train_mode == 'fixconv':
            self.deepspeech.conv.eval()
        else:
            pass
        
                
class Wav2Vec2MOS(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.freeze = freeze
        
        self.dense = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        if self.freeze:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad_(False)
                
    def forward(self, x):
        x = self.encoder(x)['last_hidden_state'] # [Batch, time, feats]
        x = self.dense(x) # [batch, time, 1]
        x = x.mean(dim=[1,2], keepdims=True) # [batch, 1, 1]
        return x
                
    def train(self, mode):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        
        

class NemoMOS(nn.Module):
    def __init__(self, model_name='speakerverification_speakernet', train_feats=True):
        super().__init__()
        self.nemo_model = make_preprocessor_trainable(load_model(name=model_name, device='cuda')).cuda()
        del self.nemo_model.decoder
        self.train_feats = train_feats
        if not self.train_feats:
            self.nemo_model.eval()
            
            for p in self.nemo_model.parameters():
                p.requires_grad_(False)
            
        #self.bilstm = nn.LSTM(input_size=1024, hidden_size=lstm_size, num_layers=1, batch_first=True, bidirectional=True)
        
        self.dense = nn.Sequential(
            nn.Linear(1500, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x, lengths = x
        feats, lengths = self.nemo_model.preprocessor.get_features(x, lengths) # [batch, Len, C]
        # print(feats.shape, feats.dtype, lengths.shape, lengths.dtype)
        feats, lengths = self.nemo_model.encoder.encoder(([feats], lengths)) # [batch, Len, C]
        feats = feats[-1]
        feats = feats.permute(0, 2, 1) # [batch, len, c]
        # print(feats.shape)
        feats = self.dense(feats)
        
        batch, time, _ = feats.shape
        
        mask = torch.zeros(batch, time, device=feats.device, dtype=torch.float32)
        for i in range(batch):
            mask[i, :lengths[i]] = 1.0
        
        mos = (feats[:,:,0] * mask).sum(dim=1) / mask.sum(dim=1)
        return mos[:, None, None]
        
    def train(self, mode):
        super().train(mode)
        if not self.train_feats:
            self.nemo_model.eval()