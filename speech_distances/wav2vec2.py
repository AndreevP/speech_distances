from transformers import Wav2Vec2Model, Wav2Vec2Processor



class Wav2Vec2ConvEncoder:

    def __init__(self, device="cuda"):
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").feature_extractor
        self.encoder.eval()
        self.encoder = self.encoder.to(device)
        self.preprocessor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.preprocessor._sample_rate = 16000
        self.device = device

    def __call__(self, x):
        # x - [bs, 1, time]
        x = x[:, 0]
        input_values = (x - x.mean(-1)[:, None]) / (x.std(-1)[:, None] + 1e-6)
        hidden_states = self.encoder(input_values.to(self.device))
        return hidden_states
    
class Wav2Vec2FullEncoder:

    def __init__(self, device="cuda"):
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.encoder.eval()
        self.encoder = self.encoder.to(device)
        self.preprocessor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.preprocessor._sample_rate = 16000
        self.device = device

    def __call__(self, x):
        # x - [bs, 1, time]
        x = x[:, 0]
        input_values = (x - x.mean(-1)[:, None]) / (x.std(-1)[:, None] + 1e-6)
        hidden_states = self.encoder(input_values.to(self.device)).last_hidden_state
        return hidden_states.transpose(-2, -1)