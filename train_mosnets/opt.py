import torch
import torch.nn.functional as F
import numpy as np


def frame_mse(pred, target, frame_weight=1.0):
    # pred: [batch, time, 1]
    # target: [batch]
    
    frame_mse = (pred - target.view(-1, 1, 1)).pow(2).mean()
    
    global_pred = pred.mean(dim=(1,2)) # [batch,]
    global_mse = (global_pred - target).pow(2).mean()
    
    return {
        'total': global_mse + frame_weight * frame_mse,
        'global': global_mse,
        'frame': frame_mse
    }

def mse(pred, target, frame_weight=1.0):
    # pred: [batch, time, 1]
    # target: [batch]
    
    #frame_mse = (pred - target.view(-1, 1, 1)).pow(2).mean()
    
    global_pred = pred.mean(dim=(1,2)) # [batch,]
    global_mse = (global_pred - target).pow(2).mean()
    
    return {
        'total': global_mse,
        'global': global_mse
    }

def clipped_mse(y_hat, label, tau = 0.5):
    mse = F.mse_loss(y_hat, label, reduction = 'none')
    threshold = torch.abs(y_hat - label)>tau
    mse = torch.mean(threshold*mse)
    return mse


def cmse(pred, target):
    global_cmse = clipped_mse(pred.mean(dim=(1,2)), target.view(-1,1,1))
    return {
        'total': global_cmse,
        'global': global_cmse
    }


def frame_cmse(pred, target):
    frame_mse = clipped_mse(pred, target.view(-1, 1, 1))
    
    global_pred = pred.mean(dim=(1,2))
    global_mse = clipped_mse(global_pred, target)
    
    return {
        'total': global_mse + frame_weight * frame_mse,
        'global': global_mse,
        'frame': frame_mse
    }


def mbnet_loss(pred, target):
    (pred_mean, pred_biased) = pred
    (mean_mos, real_mos) = target

    loss_mean = cmse(pred_mean, mean_mos)['total']
    loss_biased = cmse(pred_biased, real_mos)['total']
    loss_total = loss_mean + 4*loss_biased
    
    return {
        'total': loss_total,
        'mean': loss_mean,
        'biased': loss_biased
    }


################################################ METRICS ############################################################
import torch.distributed as dist

class PLBasicMetric:
    _METRIC_FUNCTION = None
    
    def __init__(self):
        self.reset()
        
    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def update(self, preds, target):
        res, n = self.__class__._METRIC_FUNCTION(preds, target)
        self.res += res.cpu()
        self.n += n

    def compute(self):
        if dist.is_initialized():
            self.res = self.res.cuda(dist.get_rank())
            self.n = self.n.cuda(dist.get_rank())
            
            dist.reduce(self.res, dst=0)
            dist.reduce(self.n, dst=0)
            
        return self.res / self.n
    
    def reset(self):
        self.res = torch.tensor(0.0)
        self.n = torch.tensor(0.0)
        
        
class PLListMetric:
    _METRIC_FUNCTION = None
        
    def __init__(self):
        self.reset()
        
    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def update(self, preds, target):
        self.accum_pred = preds.mean(dim=(1,2)) if self.accum_pred is None else torch.cat([self.accum_pred, preds.mean(dim=(1,2))], dim=0)
        self.accum_gt = target if self.accum_gt is None else torch.cat([self.accum_gt, target], dim=0)

    def compute(self):
        if dist.is_initialized():
            self.accum_pred = self.accum_pred.cuda(dist.get_rank())
            self.accum_gt = self.accum_gt.cuda(dist.get_rank())
            # TODO: gather
    
        return self.__class__._METRIC_FUNCTION(self.accum_pred.cpu().numpy(), self.accum_gt.cpu().numpy())
    
    def reset(self):
        self.accum_pred = None
        self.accum_gt = None
        
        
def mse_metric(preds, target):
    return (preds.mean(dim=(1,2))-target).pow(2).sum(), len(target)
        
class MSE(PLBasicMetric):
    _METRIC_FUNCTION = mse_metric
    
    def __str__(self):
        return 'MSE'
    
    
def lcc(pred, gt):
    return np.corrcoef(pred, gt)[0,1]
    
    
class LCC(PLListMetric):
    _METRIC_FUNCTION = lcc
    
    def __str__(self):
        return 'LCC'
    
    
import scipy.stats
def srcc(pred, gt):
    return scipy.stats.spearmanr(pred, gt)[0]
    
    
class SRCC(PLListMetric):
    _METRIC_FUNCTION = srcc
    
    def __str__(self):
        return 'SRCC'
