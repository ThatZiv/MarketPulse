import os
from typing import *
import torch


class ForecastModel:
    """
    base class for all forecast models
    """
    def __init__(self, model: torch.nn, name: str, ticker: str = None):
        """
        initializes a forecast model
        :param model: torch.nn model
        :param name: name of the model
        :param ticker: ticker of the stock
        
        """
        self.model = model
        self.name = name
        self.ticker = ticker
        self.model_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    def train(self, data_set):
        """ method used to invoke training a given data set """
        raise Exception("You must implement a training method for your model")

    def evaluate(self, test_set):
        """ method used to evaluate a model given a test set """
        raise Exception("You must implement an evaluation method for your model")

    def run(self, forecast_days: int):
        """ method used to run a model and forecast the next n days """
        raise Exception("You must implement a run (forecast/predict) method for your model")

    def save(self, path: str):
        torch.save(self.model.state_dict(), os.path.join(path, \
        self.name + ('-' + self.ticker.upper() if self.ticker else '') + ".pth"))
