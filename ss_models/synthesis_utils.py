"""
Utils for speech synthesis using different models are placed here
"""

from nemo.collections.asr.parts.features import FilterbankFeatures

def make_preprocessor_trainable(stt):
    big_dict = {k: v for k, v in stt.preprocessor.featurizer.__dict__.items() if not k.startswith('_') and k != 'forward'}
    st = stt.preprocessor.featurizer.state_dict()
    stt.preprocessor.featurizer = FilterbankFeatures(use_grads=True)
    stt.preprocessor.featurizer.load_state_dict(st)
    _ = {setattr(stt.preprocessor.featurizer, k, v) for k, v in big_dict.items()}
#     stt = stt.cuda()
    return stt
