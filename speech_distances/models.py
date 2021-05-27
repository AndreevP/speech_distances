"""This is a module for model load. If possible, use nemo or torch model zoo.
Imports are hided in if statements for efficiency.
"""
import os.path
import wget


def load_model(name: str, device="cpu"):
    if name.lower() == 'tacotron':
        import nemo.collections.tts as nemo_tts
        return nemo_tts.models.Tacotron2Model.from_pretrained(model_name="Tacotron2-22050Hz", map_location=device)
    elif name.lower() == 'quartznet':
        import nemo.collections.asr as nemo_asr
        return nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En", map_location=device)
    elif name.lower() == 'speakerverification_speakernet':
        import nemo.collections.asr as nemo_asr
        stt = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="speakerverification_speakernet", map_location=device)
        return stt
    elif name.lower() == 'speakerrecognition_speakernet':
        import nemo.collections.asr as nemo_asr
        stt = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="speakerrecognition_speakernet", map_location=device)
        return stt
    elif name.lower() == 'jasper':
        import nemo.collections.asr as nemo_asr
        stt = nemo_asr.models.ASRModel.from_pretrained(model_name="stt_en_jasper10x5dr")
        return stt
    elif name.lower() == 'quartznet_de':
        import nemo.collections.asr as nemo_asr
        stt = nemo_asr.models.ASRModel.from_pretrained(model_name="stt_de_quartznet15x5")
        return stt
    elif name.lower() == 'deepspeech2':
        from .deep_speech import DeepSpeechEncoderWrapper
        if os.path.isfile('weights/an4_pretrained_v2.pth'):
             return DeepSpeechEncoderWrapper("weights/an4_pretrained_v2.pth", device=device)
        else:
            os.mkdir("weights")
            wget.download("https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/an4_pretrained_v2.pth",
                          out="weights")
            return DeepSpeechEncoderWrapper("weights/an4_pretrained_v2.pth", device=device)
    elif name.lower() == 'wav2vec2':
        from .wav2vec2 import Wav2Vec2FullEncoder
        return Wav2Vec2FullEncoder(device)
    elif name.lower() == 'wav2vec2_conv':
        from .wav2vec2 import Wav2Vec2ConvEncoder
        return Wav2Vec2ConvEncoder(device)
    elif name.lower() == 'melgan':
        vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
        return vocoder
    elif name.lower() == 'wavenet_vocoder':
        wn_preset = "weights/20180510_mixture_lj_checkpoint_step000320000_ema.json"
        wn_checkpoint_path = "weights/20180510_mixture_lj_checkpoint_step000320000_ema.pth"

        if not os.path.exists(wn_preset):
            os.makedirs("weights", exist_ok=True)
            wget.download(
                'curl -O -L "https://www.dropbox.com/s/0vsd7973w20eskz/20180510_mixture_lj_checkpoint_step000320000_ema.json"',
                out="weights"
            )
        if not os.path.exists(wn_checkpoint_path):
            os.makedirs("weights", exist_ok=True)
            wget.download(
                'curl -O -L "https://www.dropbox.com/s/zdbfprugbagfp2w/20180510_mixture_lj_checkpoint_step000320000_ema.pth"',
                out="weights"
            )

        from hparams import hparams
        with open(wn_preset) as f:
            hparams.parse_json(f.read())

        import sys
        sys.path.append('thirdparty/wavenet_vocoder')

        from train import build_model
        from synthesis import wavegen
        import torch

        model = build_model().to(device)

        print("Load checkpoint from {}".format(wn_checkpoint_path))
        checkpoint = torch.load(wn_checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])

        return model
    else:
        raise NotImplementedError

## Todo: wav2vec

# nvidia-nemo TTS+vocoder-inference example:
# https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/tts/1_TTS_inference.ipynb#scrollTo=46eLhKnTPXS9
