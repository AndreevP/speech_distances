# Deep Speech Distances

Speech Generation has recently become one of the most trending problems in Deep Learning. Over the last decade DL techniques succesfully coped with such tasks as AI voice assitantce (Siri, Alexa etc.), speech enhancement and many others. Compared to other ML problems, Speech Generation has an important specification - ultimately, the sound quality can be evaluated only by the subjective assesment of a human or as an average of several individuals' assesments (Mean Opinion Score).
This approach, however, is very expensive and time demanding, so the problem of automatic perceptual assessment logically arises. In the current work we experiment with various techniques to  measure the audio quality, including such network architectures as MOS-Net, MB-Net, and compare them with old school non-neural algorithms PESQ and ViSQOLv3. Moreover, we experiment with loss-nets (with DPAM, CDPAM and LPIPS architectures) inducing differenctiable metrics w.r.t which we could fine-tune the audio generating Vocoders.

**Keywords:** GAN-TTS, speech distances, MOS-Net, MB-Net

## Getting started

Clone the repo and install requirements:
```bash
git clone https://github.com/AndreevP/speech_distances.git
pip install -r requirements.txt
```
## Inference

We offer several metrics for audio quality evaluation, among them:

- DPAM (deep perceptual audio similarity metric)
- Frechet distributional metric
- MMD (Maximum Mean Descrepancy) distributional metric

We provide easy to use interface for distributional metrics calculation:

```python
from speech_distances import FrechetDistance

path = "./generated_waveforms" # path to .wav files to be evaluated
reference_path = "./waveforms" # path to reference .wav files

backbone = "deepspeech2" # name of neural network to be used as feature extreactor 
                         # availble backbones: "deepspeech2", "wav2vec2", "quartznet",
                         # "speakerrecognition_speakernet", "speakerverification_speakernet"
          
sr = 22050 # sampling rate of these audio files
           # audio will be resampled to sampling rate suitable for particular backbone, typically 16000
           
sample_size = 10000 # number of wav files to be sampled from provided directories and used for evaluation
num_runs = 1 # number of runs with different subsets of files for computation of mean and std

window_size = None # number of timesteps within one window for feature computation
                   # for all windows the features are computed independently and then averaged 
                   # None if to use maximum window size and average only resulting feature maps
                   
conditional = True # defines whether to compute conditional version of the distance of not
use_cached = True # try to reuse extracted features if possible?

FD = FrechetDistance(path=path, reference_path=reference_path, backbone=backbone,
                     sr=sr, sample_size=sample_size,
                     num_runs=num_runs, window_size=window_size,
                     conditional=conditional, use_cached=use_cached)
                     
FD.calculate_metric() # outputs mean and std runs
```

In addition, demo notebooks for evaluation may be found in 
```bash
notebooks/demo_metric_*.ipynb
```

Beyond this, we offer console interface for loading numerous vocoder models and infer them on given input .wav files with further generation. This step is made for fair evaluation of vocoder models: input wavs are simplified to mel-spectrograms at the beginning, then vocoder is launched to generate waveform. 

```bash
usage: python prepare_wavs.py [--model_name MODEL_NAME] [--folder_in FOLDER_IN] [--folder_out FOLDER_OUT]

arguments:
  --model_name          name of vocoder model
  --folder_in           name of folder with input wavs
  --folder_out          name of folder to save output wavs
```
