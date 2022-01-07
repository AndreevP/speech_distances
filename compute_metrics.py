from speech_distances.models import load_model
import numpy as np
import librosa
import speechmetrics
import glob
from speech_distances import FrechetDistance
import os
import argparse
import itertools
from tqdm import tqdm
import logging
import datetime


def snr(x, y):
    sqrt_l2_loss = np.mean((x - y) ** 2)
    sqrn_l2_norm = np.mean(y ** 2)
    snr = 10 * np.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / np.log(10.0)
    return snr


def get_power(x):
    S = librosa.stft(x, 2048)
    S = np.log(np.abs(S) ** 2 + 1e-8)
    return S


def lsd(x, y):
    S1 = get_power(x)
    S2 = get_power(y)
    lsd = np.mean(np.sqrt(np.mean((S1 - S2) ** 2 + 1e-8, axis=1)), axis=0)
    return min(lsd, 10.0)


def calculate_all_metrics(path, reference_path, n_max_files=None):
    metrics = {}
    FD = FrechetDistance(
        path=path,
        reference_path=reference_path,
        backbone="deepspeech2",
        sr=16000,
        sample_size=10000,
        num_runs=1,
        window_size=None,
        conditional=True,
        use_cached=True,
    )
    metrics["FDSD"] = FD.calculate_metric()[0].data.item()
    FD.backbone.encoder.cpu()
    mos_pred = load_model("wave2vec_mos")

    moses = np.array(mos_pred.calculate(path, False))
    moses_ref = np.array(mos_pred.calculate(reference_path, False))
    mos_pred.cpu()
    metrics["MOS_wav2vec"] = moses.mean(), moses.std()
    metrics["MOSdeg_wav2vec"] = np.mean(np.maximum(moses_ref - moses, 0)), np.std(
        np.maximum(moses_ref - moses, 0)
    )
    metrics["MOSdeg_wav2vec_nonzero"] = np.sum(moses_ref - moses > 0) / len(
        moses.squeeze()
    )

    computer = speechmetrics.load(["bsseval", "mosnet", "pesq", "stoi", "sisdr"], None)
    ll = glob.glob(os.path.join(path, "*.wav"))
    ll_gt = glob.glob(os.path.join(reference_path, "*.wav"))

    scores = []
    for path_to_estimate_file, path_to_reference in tqdm(
        itertools.islice(zip(ll, ll_gt), n_max_files),
        total=n_max_files if n_max_files is not None else len(ll),
        desc="Calculating metrics from speechmetrics",
    ):
        scores.append(computer(path_to_estimate_file, path_to_reference))
    scores = {k: [dic[k] for dic in scores] for k in scores[0]}

    scores_ref = []
    for path_to_estimate_file, path_to_reference in tqdm(
        itertools.islice(zip(ll, ll_gt), n_max_files),
        total=n_max_files if n_max_files is not None else len(ll),
        desc="Calculating reference values of metrics",
    ):
        scores_ref.append(computer(path_to_reference, path_to_reference))
    scores_ref = {k: [dic[k] for dic in scores_ref] for k in scores_ref[0]}

    metrics["MOS_orig"] = np.mean(np.stack(scores["mosnet"])), np.std(
        np.stack(scores["mosnet"])
    )
    mosdeg = np.maximum(-np.stack(scores["mosnet"]) + np.stack(scores_ref["mosnet"]), 0)
    metrics["MOSdeg_orig"] = np.mean(mosdeg), np.std(mosdeg)
    metrics["MOSdeg_orig_nonzero"] = np.sum(mosdeg > 0) / len(mosdeg.squeeze())
    metrics["sisdr"] = np.mean(np.stack(scores["sisdr"])), np.std(
        np.stack(scores["sisdr"])
    )
    metrics["stoi"] = np.mean(np.stack(scores["stoi"])), np.std(
        np.stack(scores["stoi"])
    )
    metrics["pesq"] = np.mean(np.stack(scores["pesq"])), np.std(
        np.stack(scores["pesq"])
    )
    metrics["sdr"] = np.mean(np.stack(scores["sdr"])), np.std(np.stack(scores["sdr"]))

    LSD = []
    SNR = []
    for path_to_estimate_file, path_to_reference in tqdm(
        itertools.islice(zip(ll, ll_gt), n_max_files),
        total=n_max_files if n_max_files is not None else len(ll),
        desc="Calculating LSD and SNR metrics",
    ):
        x = librosa.load(path_to_estimate_file, sr=16000)[0]
        y = librosa.load(path_to_reference, sr=16000)[0]
        x = librosa.util.normalize(x[: min(len(x), len(y))])
        y = librosa.util.normalize(y[: min(len(x), len(y))])

        SNR.append(snr(x, y))
        LSD.append(lsd(x, y))

    metrics["snr"] = np.mean(SNR), np.std(SNR)
    metrics["lsd"] = np.mean(LSD), np.std(LSD)

    return metrics


class Logger:
    def __init__(self, name, log_dir):
        self.logger = logging.getLogger(name)
        self.logger.handlers = []
        self.logger.setLevel(logging.INFO)
        log_formatter = logging.Formatter(
            "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self.logger.addHandler(console_handler)
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(f"{log_dir}/{name}_{now}.txt")
        file_handler.setFormatter(log_formatter)
        self.logger.addHandler(file_handler)

        self.logger.propagate = False

    def print_metrics(self, metrics):
        for k, v in metrics.items():
            if isinstance(v, tuple):
                self.logger.info(f"{k} = {v[0]:.5f} +/- {v[1]:.5f}")
            else:
                self.logger.info(f"{k} = {v:.5f}")

    def log(self, msg):
        self.logger.info(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths",
        type=str,
        nargs="+",
        help="path to .wav files to be evaluated",
        default=["./generated_wavs"],
    )
    parser.add_argument(
        "--n_max_files",
        type=int,
        help="max number of .wav files to process",
        default=None,
    )
    parser.add_argument(
        "--gt_path", type=str, help="path to reference .wav files", default="./gt_wavs"
    )
    parser.add_argument("--name", type=str, help="name of the run", default="metrics")
    parser.add_argument(
        "--log_dir", type=str, help="name of the run", default="metrics_log"
    )
    args = parser.parse_args()
    logger = Logger(args.name, args.log_dir)

    for path in args.paths:
        logger.log(f"Metrics for {path}:")
        metrics = calculate_all_metrics(path, args.gt_path, args.n_max_files)
        logger.print_metrics(metrics)
        logger.log("\n")
