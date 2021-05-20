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
    def __init__(self, path: str, **kwargs):
        """
        Args:
            path (str): path to wav files to be assessed
        """
        self.path = path
        for k in kwargs.keys():
            setattr(self, k, kwargs[k])
        
    @abstractmethod
    def calculate_metric(self):
        pass