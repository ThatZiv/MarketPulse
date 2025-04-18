"""
ARIMA & SARIMA for price forecasting with sentiment adjustment
"""

import copy
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, exc, select
from sqlalchemy.orm import sessionmaker
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from database.tables import Stock_Info
from models.forecast.forecast_types import DataForecastType, DatasetType
from models.forecast.model import ForecastModel


# Normalize sentiment to [0,1] range
def normalize_sentiment(sentiment_series):
    return (sentiment_series - sentiment_series.min()) / (sentiment_series.max() - sentiment_series.min())


# Compute a small adjustment based on the latest sentiment relative to average
def get_sentiment_adjustment(sentiment_series):
    average = sentiment_series.mean()
    latest = sentiment_series.iloc[-1]
    adjustment = 0.05 * (latest - average)
    return adjustment


class AzArima(ForecastModel):
    """
    AzArima implementation of ForecastModel abstract class with sentiment adjustment
    """
    def __init__(self, name: str, ticker: str = None, arima_order=(2, 1, 2)):
        super().__init__(None, name, ticker)
        self.arima_order = arima_order
        self.model = None
        self.sentiment = None

    def train(self, data_set: DatasetType, sentiment_series: pd.Series = None):
        self.sentiment = normalize_sentiment(sentiment_series) if sentiment_series is not None else None
        self.model = ARIMA(data_set['Close'], order=self.arima_order).fit()

    def run(self, input_data: DatasetType, num_forecast_days: int) -> DataForecastType:
        raw_preds = self.model.forecast(steps=num_forecast_days).tolist()
        if self.sentiment is not None:
            adjustment = get_sentiment_adjustment(self.sentiment)
            # Apply sentiment-based multiplicative adjustment
            adjusted_preds = [pred * (adjustment ** (i + 1)) for i, pred in enumerate(raw_preds)]
            return adjusted_preds
        return raw_preds


class AzSarima(ForecastModel):
    """
    AzSarima implementation of ForecastModel abstract class with sentiment adjustment
    """
    def __init__(self, name: str, ticker: str = None, sarima_order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        super().__init__(None, name, ticker)
        self.sarima_order = sarima_order
        self.seasonal_order = seasonal_order
        self.model = None
        self.sentiment = None

    def train(self, data_set: DatasetType, sentiment_series: pd.Series = None):
        self.sentiment = normalize_sentiment(sentiment_series) if sentiment_series is not None else None
        self.model = SARIMAX(data_set['Close'], order=self.sarima_order, seasonal_order=self.seasonal_order).fit()

    def run(self, input_data: DatasetType, num_forecast_days: int) -> DataForecastType:
        raw_preds = self.model.get_forecast(steps=num_forecast_days).predicted_mean.tolist()
        if self.sentiment is not None:
            adjustment = get_sentiment_adjustment(self.sentiment)
            # Apply sentiment-based multiplicative adjustment
            adjusted_preds = [pred * (adjustment ** (i + 1)) for i, pred in enumerate(raw_preds)]
            return adjusted_preds
        return raw_preds


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

    s_open, s_close, s_high, s_low, s_volume, s_sentiment = [], [], [], [], [], []

    for row in data2:
        s_open.append(row[3])
        s_close.append(row[1])
        s_high.append(row[4])
        s_low.append(row[5])
        s_volume.append(row[2])
        s_sentiment.append(row[6])

    data2 = {
        'Close': s_close,
        'Open': s_open,
        'High': s_high,
        'Low': s_low,
        'Volume': s_volume,
        'Sentiment_Data': s_sentiment
    }
    data2 = pd.DataFrame(data2)
    data_copy = copy.deepcopy(data2)


    # Train ARIMA
    arima_model = AzArima("az-arima", "TSLA")
    arima_model.train(data2, data2['Sentiment_Data'])
    arima_model.load()
    print("\nARIMA Predictions:\n", arima_model.run(data_copy, 7))

    # Train SARIMA
    sarima_model = AzSarima("az-sarima", "TSLA")
    sarima_model.train(data2, data2['Sentiment_Data'])
    sarima_model.load()
    print("\nSARIMA Predictions:\n", sarima_model.run(data_copy, 7))

    session.close()

