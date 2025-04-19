"""
Transformer for price forecasting
"""

# import matplotlib.pyplot as plt
import copy
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, exc, select
from sqlalchemy.orm import sessionmaker

from database.tables import Stock_Info
from models.forecast.forecast_types import DataForecastType, DatasetType
from models.forecast.model import ForecastModel
# import yfinance as yf
from models.zav2 import Transformer
from scipy.special import logit


class ZavTransformer(ForecastModel):
    """
    ZavTransformer implementation of ForecastModel abstract class
    """

    def __init__(self, my_model: Transformer, name: str, ticker: str = None):
        """ ZavTransformer constructor """
        super().__init__(my_model.model, name, ticker)
        self.my_model = my_model

    def train(self, data_set: DatasetType):
        """ train the ZavTransformer model """
        tr_data, val_data = self.my_model.get_data(
            np.array(data_set['Close'].values), 0.8)
        self.my_model.training_seq(tr_data, val_data)
        self.save()  # save the model after training

    def run(self, input_data: DatasetType, num_forecast_days: int) -> DataForecastType:
        """ run model and apply sentiment adjustment """
        close_data = np.array(input_data['Close'].values)
        # _, _val_data = self.my_model.get_data(close_data, 0.8)
        # result, _ = self.my_model.forecast_seq(val_data)

        # res = self.my_model.predict_future(input_data, num_forecast_days)
        mean_pred, _lower_bound, _upper_bound = self.my_model.predict_with_confidence(
            close_data, days_ahead=num_forecast_days, num_samples=100, confidence_level=0.95)
        # return np.array(mean_pred, dtype=float).tolist()

        last_sentiment = input_data['Sentiment_Data'].values[-1]
        last_news = input_data['News_Data'].values[-1]

        norm_sentiment = safe_logit(last_sentiment)
        norm_news = safe_logit(last_news)

        # 1% of the sentiment and 3% of the news
        adjustment = 0.01 * norm_sentiment + 0.03 * norm_news

        # apply the adjustment from sentiment data
        adjusted_pred = [float(p + adjustment) for p in mean_pred]

        return adjusted_pred

        # IF YOU WANT TO CHART IT OUT
        # print(input_data[-1], res)
        # plt.plot(_, color='red', alpha=0.7)
        # plt.plot(range(len(_), len(_) + len(res)), res, color='blue', alpha=0.7)
        # plt.plot(range(len(_), len(_) + len(mean_pred)), mean_pred, color='green', alpha=0.7)

        # plt.plot(input_data, color='green', linewidth=0.7)
        # plt.title('Actual vs Forecast vs og')
        # plt.legend(['Actual', 'Forecast', "og"])
        # plt.xlabel('Time Steps')
        # plt.show()


# normalize func w logit
def safe_logit(x):
    return logit(np.clip(x, 1e-5, 1 - 1e-5))


if __name__ == "__main__":
    # this is just a test
    model = ZavTransformer(Transformer(), "zav-transformer", "TSLA")
    # data = yf.download(model.ticker, start='2020-01-01', end='2024-12-27')
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
    s_sentiment = []
    s_news = []
    for row in data2:
        s_open.append(row[3])
        s_close.append(row[1])
        s_high.append(row[4])
        s_low.append(row[5])
        s_volume.append(row[2])
        s_sentiment.append(row[6])
        s_news.append(row[8])
    data2 = {'Close': s_close, 'Open': s_open, 'High': s_high, 'Low': s_low,
             'Volume': s_volume, 'Sentiment_Data': s_sentiment, 'News_Data': s_news}
    data2 = pd.DataFrame(data2)
    data_copy = copy.deepcopy(data2)

    model.train(data2)
    model.load()

    print(model.run(data_copy, 30))
    session.close()
