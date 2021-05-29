import argparse
from datetime import datetime
import random
import os
import sys

import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from extensions.progressbar import LiteProgressBar
from extensions.custom_callbacks import *

import models
import opt
import dataset

import deep_speech


class LightningWrapper(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model', type=str, help='Model name to train', default='MOSNet')
        parser.add_argument('--loss-type', type=str, choices=['frame_mse','frame_cmse', 'cmse', 'mse', 'mbnet_loss'], default='frame_mse', help='Loss function type')
        parser.add_argument('--batch', type=int, help='Batch size', default=64)
        parser.add_argument('--workers', type=int, help='Number of workers', default=4)
        parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
        parser.add_argument('--iters', type=float, help='Epoch iterations', default=1000)
        parser.add_argument('--epochs', type=int, help='Number of epochs', default=50)
        parser.add_argument('--padding', type=str, choices=['zero', 'reppad'], default='zero', help='Padding type for batch collation')
        return parser
        
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        
        model = getattr(models, args.model)()      
        self.model = model
        
        ds_args = dict(list_path='data/mos_list.txt', data_path='../../MOSNet/data/wav/')
        if args.model == 'DeepSpeechMOS':
            self.ds = dataset.VCC2018DatasetDeepSpeech(**ds_args)
            self.collate_fn = deep_speech._collate_fn
        elif args.model == 'Wav2Vec2MOS':
            self.ds = dataset.VCC2018DatasetWav2Vec2(**ds_args)
            self.collate_fn = self.wav2vec2_collate
            self.w2v2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        elif args.model == 'NemoMOS':
            self.ds = dataset.VCC2018DatasetNoPreporocess(**ds_args)
            self.collate_fn = dataset.collate_fn_lenth
        elif args.model == 'MBNet':
            self.ds = dataset.VCC18DatasetMBNet(ds_args['data_path'], 'data/')
            self.collate_fn = dataset.collate_fn_reppad_mbnet
        else:
            self.ds = dataset.VCC2018Dataset(**ds_args)
            self.collate_fn = dataset.collate_fn_zeros if self.hparams.padding == 'zero' else dataset.collate_fn_reppad
            
        self.loss_fn = getattr(opt, self.hparams.loss_type)
        self.metrics = [opt.MSE(), opt.LCC(), opt.SRCC()]

    def training_step(self, batch, batch_idx):
        batch, target = batch
        out = self.model(batch)
        loss = self.loss_fn(out, target)
        
        loss_vals = {'Loss/'+k: v.item() for k,v in loss.items()}
        if isinstance(out, tuple):
            loss_vals['Metrics/MSE_train'] = (out[0].mean(dim=(1,2))-target[0]).pow(2).mean()
        else:
            loss_vals['Metrics/MSE_train'] = (out.mean(dim=(1,2))-target).pow(2).mean()
        self.log_dict(loss_vals, on_epoch=True, prog_bar=True, logger=True)

        return loss['total']

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        batch, target = batch
        out = self.model(batch)
        
        for metric in self.metrics:
            metric(out, target)
    
    def validation_epoch_end(self, *args, **kwargs):
        val_dict = {}
        for metric in self.metrics:
            val_dict['Metrics/'+str(metric)] = float(metric.compute())
            metric.reset()
        self.log_dict(val_dict, on_epoch=True, prog_bar=True, logger=True)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.ds['train'], batch_size=self.hparams.batch, shuffle=True, 
                                           num_workers=self.hparams.workers, 
                                           collate_fn = self.collate_fn)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.ds['test'], batch_size=self.hparams.batch, shuffle=False, 
                                           num_workers=self.hparams.workers, 
                                           collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return self.val_dataloader()
    
    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)
    
    def test_epoch_end(self, *args, **kwargs):
        return self.validation_epoch_end(*args,**kwargs)
    
    def test_step_end(self, *args, **kwargs):
        return self.validation_step_end(*args, **kwargs)
    
    def wav2vec2_collate(self, batch):
        
        return 

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--distrib', action='store_true', help='Multi-GPU distributed training')
    parser.add_argument('--gpus', default=torch.cuda.device_count(), type=int, help='Distributed training world size')
    parser.add_argument('--test', action='store_true')
    return parser


def main():
    EXPERIMENTS_PATH = 'experiments/'
    
    experiment_name = sys.argv[1:]
    for index in range(len(experiment_name)):
        while experiment_name[index][0] == '-':
            experiment_name[index] = experiment_name[index][1:]
    experiment_name = '_'.join(experiment_name)

    pl.seed_everything(42)
    parser = arg_parser()
    parser = LightningWrapper.add_model_specific_args(parser)
    args = parser.parse_args()
    
    kwargs = {}
    if args.distrib:
        kwargs['accelerator'] = 'ddp'
        kwargs['sync_batchnorm'] = True

    datetime_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    experiment_folder = os.path.join(EXPERIMENTS_PATH, experiment_name, datetime_str)

    wrapper = LightningWrapper(args)
    
    trainer = pl.Trainer(
        gpus=1 if not args.distrib else args.gpus, 
        max_epochs=args.epochs, 
        limit_train_batches=args.iters, 
        num_sanity_val_steps=0, 
        default_root_dir=None,
        deterministic=True,
        logger=None,
#        logger=[
#             pl_loggers.TensorBoardLogger(TENSORBOARD_PATH, name=args.experiment_name, version=datetime_str),
#             pl_loggers.TensorBoardLogger('/tensorboard')
#        ],
        weights_summary=None,
        callbacks=[
            LiteProgressBar(), 
            CodeSnapshotter(experiment_folder),
            EnvironmentCollector(experiment_folder),
            MetricLogger(experiment_folder),
            ParamsLogger(experiment_folder),
            ModelCheckpoint(dirpath=os.path.join(experiment_folder, 'weights'), filename='{epoch}-{step}', save_top_k=-1),
        ], 
        **kwargs
    )

    trainer.fit(wrapper)
    

if __name__ == '__main__':
    main()
