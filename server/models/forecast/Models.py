import os
from typing import *
import torch


class ForecastModels:
    def __init__(self, *args, **kwargs):
        self.models: List[ForecastModel] = args

    def train_all(self, data_set):
        for model in self.models:
            model.train(data_set)

    def run_all(self, forecast_days: int) -> List[Dict[name: str, forecast: List[float]]]:
        # TODO: implement ingestion to stock_prediction table here
        return {
            "forecast": [model.run(forecast_days) for model in self.models],
            "name": model.name
        }
