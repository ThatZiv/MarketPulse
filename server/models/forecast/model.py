import os
from abc import ABC, abstractmethod
from typing import *
import torch
from models.forecast.forecast_types import DatasetType, DataForecastType

class ForecastModel(ABC):
    """
    base class for all forecast models
    """
    def __init__(self, model: torch.nn, name: str, ticker: str = None):
        """
        initializes a forecast model

        :param model: torch.nn model
        :param name: name of YOUR model
        :param ticker: ticker of the stock
        """
        self.model = model
        self.name = name
        self.ticker = ticker
        self.model_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
        self.model_path = os.path.join(self.model_dir, \
        self.name + ('-' + self.ticker.upper() if self.ticker else '') + ".pth")
        # if os.path.exists(self.model_path):
        #     self.load()

    @abstractmethod
    def train(self, data_set: DatasetType):
        """ method used to invoke training a given data set """

    @abstractmethod
    def run(self, input_data: DatasetType, num_forecast_days: int) -> DataForecastType:
        """ method used to run a model and forecast the next n days """

    def save(self):
        """ method used to save a model to a given path """
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, \
        self.model_path))
        print(f"Saved model to {self.model_dir} for {self.name}")

    def load(self):
        """ method used to load a model from a given path """
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(os.path.join(self.model_dir, \
            self.model_path)))
            print(f"Loaded model from {self.model_dir} for {self.name}")
