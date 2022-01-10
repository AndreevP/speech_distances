from speech_distances.models import load_model
import argparse
from scipy.stats import wilcoxon, mannwhitneyu
import numpy as np

# Test hypothesis that path1 and path2 files have the same quality
# against one-sided alternative that files in path1 are better than path2 files 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path1", type=str, help="path to .wav files which are assumed to be better"
    )
    parser.add_argument(
        "--path2", type=str, help="path to .wav files which are assumed to be worse"
    )
    args = parser.parse_args()

    mos_pred = load_model("wave2vec_mos")

    moses_1 = np.array(mos_pred.calculate(args.path1, False))
    moses_2 = np.array(mos_pred.calculate(args.path2, False))
    
    print("Ratio:", (moses_1 > moses_2).sum() / len(moses_1))
    print("p-value:", wilcoxon(moses_1, moses_2, alternative="greater")[1])