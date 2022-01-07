from speech_distances.models import load_model
import numpy as np
import librosa
import speechmetrics
import glob 
from speech_distances import FrechetDistance
import os


def snr(x, y):
    sqrt_l2_loss = np.mean((x-y)**2)
    sqrn_l2_norm = np.mean(y**2)
    snr = 10 * np.log(sqrn_l2_norm/sqrt_l2_loss + 1e-8) / np.log(10.)
    return snr

def get_power(x):
    S = librosa.stft(x, 2048)
    S = np.log(np.abs(S)**2 + 1e-8)
    return S

def lsd(x, y):
    S1 = get_power(x)
    S2 = get_power(y)
    lsd = np.mean(np.sqrt(np.mean((S1-S2)**2 + 1e-8, axis=1)), axis=0)
    return min(lsd, 10.)

def calculate_all_metrics(path, reference_path):
    metrics = {}
    FD = FrechetDistance(path=path, reference_path=reference_path, backbone="deepspeech2",
                         sr=16000, sample_size=10000,
                         num_runs=1, window_size=None,
                         conditional=True, use_cached=True)
    metrics["FDSD"] = FD.calculate_metric()[0].data.item()
    FD.backbone.encoder.cpu()
    mos_pred = load_model("wave2vec_mos")
    
    moses = np.array(mos_pred.calculate(path, False))
    moses_ref = np.array(mos_pred.calculate(reference_path, False))
    mos_pred.cpu()
    metrics["MOS_wav2vec"] = moses.mean()
    metrics["MOS_wav2vec_std"] = moses.std()
    metrics["MOSdeg_wav2vec"] = np.mean(np.maximum(moses_ref - moses, 0))
    metrics["MOSdeg_wav2vec_std"] = np.std(np.maximum(moses_ref - moses, 0))
    metrics["MOSdeg_wav2vec_nonzero"] = np.sum(moses_ref - moses > 0) / len(moses.squeeze())
    
    computer = speechmetrics.load(['bsseval', 'mosnet', 'pesq', 'stoi', "sisdr"], None)
    ll = glob.glob(os.path.join(path, "*.wav"))
    ll_gt = glob.glob(os.path.join(reference_path, "*.wav"))

    scores = []
    for path_to_estimate_file, path_to_reference in zip(ll, ll_gt):
        scores.append(computer(path_to_estimate_file, path_to_reference))
    scores = {k: [dic[k] for dic in scores] for k in scores[0]}
    metrics["MOS_orig"] = np.mean(np.stack(scores['mosnet']))
    metrics["MOS_orig_std"] = np.std(np.stack(scores['mosnet']))
    metrics["sisdr"] = np.mean(np.stack(scores['sisdr']))
    metrics["stoi"] = np.mean(np.stack(scores['stoi']))
    metrics["pesq"] = np.mean(np.stack(scores['pesq']))
    metrics["sdr"] = np.mean(np.stack(scores['sdr']))
    
    scores_ref = []
    for path_to_estimate_file, path_to_reference in zip(ll, ll_gt):
        scores_ref.append(computer(path_to_reference, path_to_reference))
    scores_ref = {k: [dic[k] for dic in scores_ref] for k in scores_ref[0]}
    mosdeg = np.maximum(-np.stack(scores['mosnet']) + np.stack(scores_ref['mosnet']), 0)
    metrics["MOSdeg_orig"] = np.mean(mosdeg)
    metrics["MOSdeg_orig_std"] = np.std(mosdeg)
    metrics["MOSdeg_orig_nonzero"] = np.sum(mosdeg > 0) / len(mosdeg.squeeze())
    
    LSD = []
    SNR = []
    for path_to_estimate_file, path_to_reference in zip(ll, ll_gt):
        x = librosa.load(path_to_estimate_file, sr=16000)[0]
        y = librosa.load(path_to_reference, sr=16000)[0]
        x = librosa.util.normalize(x[:min(len(x), len(y))])
        y = librosa.util.normalize(y[:min(len(x), len(y))])


        SNR.append(snr(x, y))
        LSD.append(lsd(x, y))
        
    metrics["snr"] = np.mean(SNR)
    metrics["lsd"] = np.mean(LSD)
    
    return metrics

path = "/home/pavel/BWE_final/input_4khz" # path to .wav files to be evaluated
reference_path = "/home/pavel/BWE_final/gt" # path to reference .wav files

print(calculate_all_metrics(path, reference_path))
