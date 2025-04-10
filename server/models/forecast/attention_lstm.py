"""
LSTM for price forecasting
"""

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=duplicate-code



import copy
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, select, exc
from sqlalchemy.orm import sessionmaker
from models.forecast.forecast_types import DataForecastType, DatasetType
from models.forecast.model import ForecastModel
from models.lstm_attention import AttentionLstm
from database.tables import Stock_Info


'''Attention LSTM implementation'''
class AttentionLSTM(ForecastModel):

    def __init__(self, my_model: AttentionLstm, name: str, ticker: str = None):
        super().__init__(my_model.model, name, ticker)
        self.my_model = my_model

    def train(self, data_set):
        model_input, testing_out, validation_out, _, _, _ = self.my_model.format_data(data_set)
        train_data, test_data, validation_data = self.my_model.get_data(model_input,  validation_out, testing_out, 0.1)
        self.my_model.model_training(train_data, test_data, 20)
        self.my_model.evaluate(validation_data)
        self.save()

    def run(self, input_data: DatasetType, num_forecast_days: int) -> DataForecastType:
        data, _, _, multiple, minimum, sentiment = self.my_model.format_data(input_data)
        data = self.my_model.create_prediction_sequence(data, 10)
        output = self.my_model.forecast_seq(data, sentiment, period = num_forecast_days)
        print(output)
        output = [x * multiple + minimum for x in output]
        return np.array(output, dtype = float).tolist()

