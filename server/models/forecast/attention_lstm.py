"""
LSTM for price forecasting
"""

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=duplicate-code



import copy
import numpy as np
from models.forecast.forecast_types import DataForecastType, DatasetType
from models.forecast.model import ForecastModel
from models.lstm_attention import AttentionLstm


'''Attention LSTM implementation'''
class AttentionLSTM(ForecastModel):

    def __init__(self, my_model: AttentionLstm, name: str, ticker: str = None):
        super().__init__(my_model.model, name, ticker)
        self.my_model = my_model

    def train(self, data_set):
        count = 0
        # This is to avoid models with low r2 values. This is to retrain models that
        # may have gotten stuck in a local minimum resulting in lower than expected correlation.
        # This is was added to in theory models that more closly match the validation data set.
        # After 5 attempts this returns the "best" model.
        # With to high a required r2 value the results are a cherry picked model that are not be the
        # good desipite having better measured statistics.
        model = copy.deepcopy(self.my_model)
        model_input, testing_out, validation_out, _, _, _ = model.format_data(data_set)
        train_data, test_data, validation_data = model.get_data(model_input,  validation_out, testing_out, 0.1)
        model.model_training(train_data, test_data, 20)
        r2 = model.evaluate(validation_data)
        best_model = model
        while(r2<.80 and count <4):
            model = copy.deepcopy(self.my_model)
            model_input, testing_out, validation_out, _, _, _ = model.format_data(data_set)
            train_data, test_data, validation_data = model.get_data(model_input,  validation_out, testing_out, 0.1)
            model.model_training(train_data, test_data, 20)
            new_r2 = model.evaluate(validation_data)
            if new_r2 > r2:
                best_model = model
                r2 = new_r2
            count = count+1
        self.my_model = best_model
        self.save()

    def run(self, input_data: DatasetType, num_forecast_days: int) -> DataForecastType:
        data, _, _, multiple, minimum, sentiment = self.my_model.format_data(input_data)
        data = self.my_model.create_prediction_sequence(data, 10)
        output = self.my_model.forecast_seq(data, sentiment, period = num_forecast_days)
        print(output)
        output = [x * multiple + minimum for x in output]
        return np.array(output, dtype = float).tolist()

