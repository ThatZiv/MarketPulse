import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from models.zav2 import Transformer

from .ForecastTypes import DataForecastType, DatasetType
from .Model import ForecastModel


class ZavTransformer(ForecastModel):
    """
    ZavTransformer implementation of ForecastModel abstract class
    """
    def __init__(self, my_model: Transformer, name: str, ticker: str = None):
        super().__init__(my_model.model, name, ticker)
        self.my_model = my_model


    def train(self, data_set: DatasetType):
        tr_data, val_data = self.my_model.get_data(np.array(data_set), 0.6)
        self.my_model.training_seq(tr_data, val_data)
        self.save() # save the model after training

    def run(self, input_data: DatasetType, num_forecast_days: int) -> DataForecastType:
        _, val_data = self.my_model.get_data(np.array(input_data), 0.6)
        test_result, truth = self.my_model.forecast_seq(val_data)
        plt.plot(truth, color='red', alpha=0.7)
        plt.plot(test_result, color='blue', linewidth=0.7)
        plt.title('Actual vs Forecast')
        plt.legend(['Actual', 'Forecast'])
        plt.xlabel('Time Steps')
        plt.show()


if __name__ == "__main__":
    model = ZavTransformer(Transformer(), "zav-transformer", "TSLA")
    data = yf.download(model.ticker, start='2020-01-01', end='2024-12-27')
    data = pd.DataFrame(data['Close'].values)
    model.train(data)
    # model.load()
    model.run(data, 30)