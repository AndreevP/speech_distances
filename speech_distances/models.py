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
    elif name.lower() == 'deepspeech2':
        from .deep_speech import DeepSpeechEncoderWrapper
        if os.path.isfile('weights/an4_pretrained_v2.pth'):
             return DeepSpeechEncoderWrapper("weights/an4_pretrained_v2.pth", device=device)
        else:
            os.mkdir("weights")
            wget.download("https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/an4_pretrained_v2.pth",
                          out="weights")
            return DeepSpeechEncoderWrapper("weights/an4_pretrained_v2.pth", device=device)
            