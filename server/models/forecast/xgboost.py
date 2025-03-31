import copy
import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, select, exc
from sqlalchemy.orm import sessionmaker
from database.tables import Stock_Info

from models.forecast.model import ForecastModel
from models.xgboost_model import XGBoostModel
from models.forecast.forecast_types import DataForecastType, DatasetType

class XGBoost(ForecastModel):
    """
    XGBoost implementation of ForecastModel abstract class
    """
    def __init__(self, name: str, ticker: str = None):
        self.model = XGBoostModel(ticker)
        self.optimal_model = None
        super().__init__(self.model, name, ticker)

    def train(self, data_set: DatasetType):
        # Currently works with close value as input
        self.optimal_model = self.model.model_actual_run(data_set)

    def run(self, input_data: DatasetType, num_forecast_days: int) -> DataForecastType:
        predictions = self.model.future_predictions(self.optimal_model,
                                                    input_data, num_forecast_days)
        predicted_list = predictions.tolist()
        return predicted_list

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

    stock_q = select(Stock_Info).where(Stock_Info.stock_id == 2)
    Session = sessionmaker(bind=engine)
    session = Session()
    data2 = session.connection().execute(stock_q).all()
    s_open = []
    s_close = []
    s_high = []
    s_low = []
    s_volume = []
    s_sentiment_data = []
    s_news_data = []
    for row in data2:
        s_open.append(row[3])
        s_close.append(row[1])
        s_high.append(row[4])
        s_low.append(row[5])
        s_volume.append(row[2])
        s_sentiment_data.append(row[6])
        s_news_data.append(row[8])
    data2 = {'Close': s_close, 'Open': s_open, 'High':s_high, 'Low':s_low, 'Volume':s_volume, 'Sentiment_Data':s_sentiment_data, 'News_Data':s_news_data}
    data2 = pd.DataFrame(data2)
    data_copy = copy.deepcopy(data2)
    model = XGBoost("XGBoost-model", "F")
    model.train(data2)
    print(model.run(data_copy, 7))
