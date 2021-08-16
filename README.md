# Deep Speech Distances (PyTorch)

Please check our [report](https://drive.google.com/file/d/1Ud6zm1KbeTOD6Y5_K5MfW3zNjhRMQO7V/view?usp=sharing) for a detailed description of this project results.

This repo contatins utilities for automatic audio quality assesent. We provide code for distributional (Frechet-style) metrics computation and direct MOS score prediction. According to our experiments these methods for speech quality assessment have high correlation with MOS-es computed by crowd-sourced studies. 


**Keywords:** GAN-TTS, speech distances, MOS-Net, MB-Net

## Getting started

Clone the repo and install requirements (or better create conda environment from .yml file):
```bash
git clone https://github.com/AndreevP/speech_distances.git
pip install -r requirements.txt 
```
## Inference

We provide easy to use interface for distributional (Frechet distance and MMD) metrics calculation:

```python
from speech_distances import FrechetDistance # or MMD

path = "./generated_waveforms" # path to .wav files to be evaluated
reference_path = "./waveforms" # path to reference .wav files

backbone = "deepspeech2" # name of neural network to be used as feature extractor 
                         # available backbones: "deepspeech2", "wav2vec2", "quartznet",
                         # "speakerrecognition_speakernet", "speakerverification_speakernet"
          
sr = 22050 # sampling rate of these audio files
           # audio will be resampled to sampling rate suitable for the particular backbone, typically 16000
           
sample_size = 10000 # number of wav files to be sampled from provided directories and used for evaluation
num_runs = 1 # number of runs with different subsets of files for computation of mean and std

window_size = None # number of timesteps within one window for feature computation
                   # for all windows the features are computed independently and then averaged 
                   # if None use maximum window size and average only resulting feature maps
                   
conditional = True # defines whether to compute conditional version of the distance of not
use_cached = True # try to reuse extracted features if possible?

FD = FrechetDistance(path=path, reference_path=reference_path, backbone=backbone,
                     sr=sr, sample_size=sample_size,
                     num_runs=num_runs, window_size=window_size,
                     conditional=conditional, use_cached=use_cached)
                     
FD.calculate_metric() # outputs mean and std of metric computed for different subsets (num_runs) of audio files 
```
One can also directly predict MOS scores by our wav2vec2_mos model:

```python
from speech_distances.models import load_model

mos_pred = load_model("wave2vec_mos")

path = "./generated_waveforms" # path to .wav files to be evaluated
mos_pred.calculate(path) # outputs predicted MOS
```

According to our experiments these two methods for speech quality assessment have high correlation with MOS-es computed by crowd-sourced studies.


