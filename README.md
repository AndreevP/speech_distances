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

Demo notebooks for evaluation may be found in 
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
