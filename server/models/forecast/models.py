
import copy
from typing import *
from models.forecast.model import ForecastModel
from models.forecast.forecast_types import DatasetType, ForecastSeriesType


class ForecastModels:
    """
    class used to manage all forecast models
    """
    def __init__(self, *args: Optional[List[type[ForecastModel]]]):
        self.models: List[type[ForecastModel]] = args or []

    def train_all(self, data_set: DatasetType):
        """ method used to train all models """

        for model in self.models[0]:
            print(model)
            model.train(copy.deepcopy(data_set))

    def run_all(self, data_set: DatasetType, forecast_days: int) -> ForecastSeriesType:
        """ method used to run all models and forecast the next n days """
        
        return [
            { "forecast": model.run(copy.deepcopy(data_set), forecast_days),"name": model.name } \
                for model in self.models[0]
        ]
