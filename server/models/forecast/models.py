from typing import *
from .model import ForecastModel
from .forecast_types import DatasetType, ForecastSeriesType

class ForecastModels:
    """
    class used to manage all forecast models
    """
    def __init__(self, *args: Optional[List[type[ForecastModel]]]):
        self.models: List[type[ForecastModel]] = args or []

    def train_all(self, data_set: DatasetType):
        """ method used to train all models """
        for model in self.models:
            model.train(data_set)

    def run_all(self, forecast_days: int) -> ForecastSeriesType:
        """ method used to run all models and forecast the next n days """
        # TODO: implement ingestion to stock_prediction table here
        return [
            { "forecast": model.run(forecast_days),"name": model.name } \
                for model in self.models
        ]
