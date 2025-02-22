"""
LSTM for price forecasting
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..lstm__attention_2 import AttentionLstm

from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker
from database.tables import Base, Account, User_Stocks, Stocks, Stock_Info

from .forecast_types import DataForecastType, DatasetType
from .model import ForecastModel


class AttentionLSTM(ForecastModel):

    def __init__(self, my_model: AttentionLstm, name: str, ticker: str = None):
        super().__init__(my_model.model, name, ticker)
        self.my_model = my_model

    def def train(self, data_set: DatasetType):
        self.save()
    
    def run(self, input_data: DatasetType, num_forecast_days: int) -> DataForecastType:
        return 0

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
    except Exception as e:
        print(f"Failed to connect: {e}")
    
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
    data = {'Close': s_close, 'Open': s_open, 'High':s_high, 'Low':s_low, 'Volume':s_volume}
    data = pd.DataFrame(data) 

    model.train(data)
    model.run(data, 30)