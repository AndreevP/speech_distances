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
        import torch
        import gdown
        os.makedirs("weights", exist_ok=True)
        # wget.download('https://github.com/descriptinc/melgan-neurips/archive/master.zip', out="weights")
        url = 'https://drive.google.com/uc?id=' + '1vNp5ZsfEBZQBXqsUOJZUYTkTedk6HZQS'
        gdown.download(url, 'weights/melgan-neurips-master.zip', quiet=True)
        os.system('unzip weights/melgan-neurips-master.zip -d weights/')
        vocoder = torch.hub.load('weights/melgan-neurips-master', 'load_melgan', source='local')
        return vocoder
    elif name.lower() == 'waveglow':
        from .waveglow import Vocoder
        vocoder = Vocoder().to(device)
        return vocoder
    elif name.lower() == 'wavenet':
        wn_preset = "weights/20180510_mixture_lj_checkpoint_step000320000_ema.json"
        wn_checkpoint_path = "weights/20180510_mixture_lj_checkpoint_step000320000_ema.pth"

        if not os.path.exists(wn_preset):
            os.makedirs("weights", exist_ok=True)
            # wget.download(
            #     'https://www.dropbox.com/s/0vsd7973w20eskz/20180510_mixture_lj_checkpoint_step000320000_ema.json',
            #     out="weights"
            # )
            os.system('curl -L "https://www.dropbox.com/s/0vsd7973w20eskz/20180510_mixture_lj_checkpoint_step000320000_ema.json" -o weights/20180510_mixture_lj_checkpoint_step000320000_ema.json')
        if not os.path.exists(wn_checkpoint_path):
            os.makedirs("weights", exist_ok=True)
            # wget.download(
            #     'https://www.dropbox.com/s/zdbfprugbagfp2w/20180510_mixture_lj_checkpoint_step000320000_ema.pth',
            #     out="weights"
            # )
            os.system('curl -L "https://www.dropbox.com/s/zdbfprugbagfp2w/20180510_mixture_lj_checkpoint_step000320000_ema.pth" -o weights/20180510_mixture_lj_checkpoint_step000320000_ema.pth')


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
        checkpoint = torch.load(wn_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

        return (hparams, model)

    elif name.lower() in ['hifigan', 'hifigan_v1', 'hifigan_v2', 'hifigan_v3']:
        import gdown

        name = name.lower()
        header = "https://drive.google.com/uc?id="

        if name in ['hifigan', 'hifigan_v1']:
            name = 'hifigan_v1'
            model_url = header + "1QEBKespXTmsMzsSRBXWdpIT0Ve7nnaRZ"
            config_url = header + "1l5EUVBKM0SK7ec4HWf_wZvEITAsdOLFC"
        elif name == 'hifigan_v2':
            model_url = header + "1I415g2Cdx5FWy6ECma0zEc9GhX_TnbFv"
            config_url = header + "11LnhSum3EAeo5zag-tpU8HKk0MdbrQxF"
        else:
            model_url = header + "1fnkOteyRdPq4Gh2cfso3gqqrC6inLWsF"
            config_url = header + "1mke75axgO2sdJ41GL2HTrcb4KyAl0i45"

        if not os.path.exists(f'pretrained/{name}'):
            os.makedirs(f'pretrained/{name}', exist_ok=True)

        model_output = f'pretrained/{name}/model.pth'
        config_output = f'pretrained/{name}/config.json'
        gdown.download(model_url, model_output, quiet=True)
        gdown.download(config_url, config_output, quiet=True)

    else:
        raise NotImplementedError

## Todo: wav2vec

# nvidia-nemo TTS+vocoder-inference example:
# https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/tts/1_TTS_inference.ipynb#scrollTo=46eLhKnTPXS9
