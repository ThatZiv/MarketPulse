import copy
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sqlalchemy import create_engine, select, exc
from sqlalchemy.orm import sessionmaker
from models.forecast.model import ForecastModel
from database.tables import Stock_Info
from models.xgBoost_implementation import XGBoostModel
from models.forecast.forecast_types import DataForecastType, DatasetType
from datetime import date
import yfinance as yf

class XGBoost(ForecastModel):
    """
    XGBoost implementation of ForecastModel abstract class
    """
    def __init__(self, name: str, ticker: str = None):
        self.model = XGBoostModel()
        self.optimal_model = None
        super().__init__(self.model, name, ticker)
    
    def train(self, data_set: DatasetType):
        # Currently works with close value as input
        self.optimal_model = self.model.model_actual_run(data_set)

    
    def run(self, input_data: DatasetType, num_forecast_days: int) -> DataForecastType:
        predictions = self.model.future_predictions(self.optimal_model, DatasetType, num_forecast_days)
        print("Predictions:\n")
        for i in range(len(predictions)):
            print(predictions[i], end='\n')
        predicted_list = predictions.tolist()
        return predicted_list
    
    def save(self):
        self.model.save()
    
    def load(self):
        self.model.load()

if __name__ == "__main__":
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

    stock_q = select(Stock_Info).where(Stock_Info.stock_id == 1)
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
    data2 = {'Close': s_close, 'Open': s_open, 'High': s_high, 'Low': s_low, 'Volume': s_volume}
    data2 = pd.DataFrame(data2)
    data_copy = copy.deepcopy(data2)

    model = XGBoost("XGBoost-model", "TSLA")
    model.train(data2)




    print(model.run(data_copy, 7))
