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
from models.lstm_attention_7 import AttentionLstm
from database.tables import Stock_Info



class AttentionLSTM(ForecastModel):

    def __init__(self, my_model: AttentionLstm, name: str, ticker: str = None):
        super().__init__(my_model.model, name, ticker)
        self.my_model = my_model

    def train(self, data_set):
        model_input, testing_out, validation_out, _, _ = self.my_model.format_data(data_set)
        train_data, test_data, validation_data = self.my_model.get_data(model_input,  validation_out, testing_out, 0.1)
        self.my_model.model_training(train_data, test_data, 20)
        self.my_model.evaluate(validation_data)
        self.save()

    def run(self, input_data: DatasetType, num_forecast_days: int) -> DataForecastType:
        data, _, _, multiple, minimum = self.my_model.format_data(input_data)
        data = self.my_model.create_prediction_sequence(data, 10)
        output = self.my_model.forecast_seq(data, period = num_forecast_days)
        print(output)
        output = [x * multiple + minimum for x in output]
        return np.array(output, dtype = float).tolist()

if __name__ == "__main__":

    model = AttentionLSTM(AttentionLstm(), "attention_lstm", "TSLA")
    load_dotenv()
    USER = os.getenv("user")
    PASSWORD = os.getenv("password")
    HOST = os.getenv("host")
    PORT = os.getenv("port")
    DBNAME = os.getenv("dbname")

    DATABASE_URL = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}?sslmode=require"

    engine = create_engine(DATABASE_URL)
    try:
        with engine.connect() as connection:
            print("Connection successful!")
    except exc.OperationalError as e:
        print(e)
    except exc.TimeoutError as e:
        print(e)

    stock_q= select(Stock_Info).where(Stock_Info.stock_id == 1)
    Session = sessionmaker(bind=engine)
    session = Session()
    data2 = session.connection().execute(stock_q).all()
    s_open = []
    s_close = []
    s_high = []
    s_low = []
    s_volume = []
    for row in data2:
        s_open.append(row[3])
        s_close.append(row[1])
        s_high.append(row[4])
        s_low.append(row[5])
        s_volume.append(row[2])
    data2 = {'Close': s_close, 'Open': s_open, 'High':s_high, 'Low':s_low, 'Volume':s_volume}
    data2 = pd.DataFrame(data2)
    data_copy = copy.deepcopy(data2)

    model.train(data2)

    print(model.run(data_copy, 30))
