from .metric import Metric

# to be refined based on the new project structure
class DPAM(Metric):
    
    def __init__(self, model: str, dataset: str):
        if model not in DPAM.list_available_models():
            raise ValueError('Model is not supported. List of supported models: {}'.format(
                self.list_available_models()))
        if dataset not in DPAM.list_available_datasets():
            raise ValueError('Dataset is not supported. List of supported datasets: {}'.format(
                self.list_available_datasets()))
        super().__init__(model, dataset)
    
    @classmethod
    def list_available_models(cls):
        return ['quartznet']
    
    @classmethod
    def list_available_datasets(cls):
        return ['ljspeech']
    
    def calculate(self):
        raise NotImplementedError