"""This module is for abstract metric class
"""
from abc import ABC, abstractmethod
from .models import load_model
from .datasets import load_dataset


class Metric(ABC):
    """This is an abstract class for all the metrics
    that are used in this project. 
    Child classes must implement all the methods with @abstractmethod decorator.
    """
    def __init__(self, model: str, dataset: str):
        self.model = load_model(model)
        self.dataset = load_dataset(dataset)
        
    @classmethod
    @abstractmethod 
    def list_available_models(cls):
        pass
    
    @classmethod
    @abstractmethod 
    def list_available_datasets(cls):
        pass
    
    @abstractmethod
    def calculate(self):
        pass