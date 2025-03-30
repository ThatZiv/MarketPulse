# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=duplicate-code


import copy
import json
from datetime import date
from sqlalchemy import select, exc
from sqlalchemy.orm import sessionmaker
import pandas as pd
from flask import current_app
from models.forecast.models import ForecastModels
from models.forecast.attention_lstm import AttentionLSTM
from models.forecast.cnn_lstm import CNNLSTMTransformer
from models.lstm_attention import AttentionLstm
from models.forecast.transformer import ZavTransformer
from models.forecast.azad import AzSarima
from models.forecast.xgboost import XGBoost
from models.zav2 import Transformer
from database.tables import Stock_Info, Stock_Predictions, Stocks
from engine import get_engine, global_engine
def run_models():
    try:
        session = sessionmaker(bind=global_engine())
    except exc.OperationalError:
        with current_app.config["MUTEX"]:
            session = sessionmaker(bind=get_engine())
    session = session()
    stock_list = select(Stocks)

    stock_out = session.connection().execute(stock_list).all()

    for stock in stock_out:
        stock_q= select(Stock_Info).where(Stock_Info.stock_id == stock.stock_id)

        output = session.connection().execute(stock_q).all()
        s_open = []
        s_close = []
        s_high = []
        s_low = []
        s_volume = []
        s_sentiment_data = []
        s_news_data = []

        for row in output:
            s_open.append(row[3])
            s_close.append(row[1])
            s_high.append(row[4])
            s_low.append(row[5])
            s_volume.append(row[2])
            s_sentiment_data.append(row[6])
            s_news_data.append(row[7])

        data = {'Close': s_close, 'Open': s_open, 'High':s_high, 'Low':s_low, 'Volume':s_volume, 'Sentiment_Data':s_sentiment_data, 'News_Data':s_news_data}
        data = pd.DataFrame(data)
        one_day = []


        one_day.append(AttentionLSTM(AttentionLstm(), "attention_lstm", stock.stock_ticker))
        one_day.append(CNNLSTMTransformer("cnn-lstm", stock.stock_ticker))
        one_day.append(ZavTransformer(Transformer(), "transformer", stock.stock_ticker))
        one_day.append(XGBoost("XGBoost-model", stock.stock_ticker))
        one_day.append(AzSarima("az-sarima", stock.stock_ticker))
        # one_day.append(AzArima("az-arima", stock.stock_ticker))

        prediction = ForecastModels(one_day)

        print (data)
        prediction.train_all(copy.deepcopy(data))

        pred = prediction.run_all(copy.deepcopy(data), 7)
        print(pred[3])
        print(pred[4])
        model_1 = pred[0]
        model_2 = pred[1]
        model_3 = pred[2]
        model_4 = pred[3]
        model_5 = pred[4]

        new_row = Stock_Predictions(stock_id=stock.stock_id, model_1=json.dumps(model_1), model_2=json.dumps(model_2), model_3=json.dumps(model_3), model_4=json.dumps(model_4), model_5=json.dumps(model_5), created_at = date.today())
        session.add(new_row)
    try:
        session.commit()
    except  exc.SQLAlchemyError as e:
        print(e)
    session.close()
