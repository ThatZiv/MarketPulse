import threading
import copy
import json
from datetime import date, datetime
from engine import get_engine
from sqlalchemy import select, func, exc
from sqlalchemy.orm import sessionmaker
import pandas as pd
from models.forecast.models import ForecastModels
from models.forecast.attention_lstm import AttentionLSTM
from models.forecast.attention_lstm_7 import AttentionLSTM as AttentionLSTM_7
from models.lstm_attention import AttentionLstm
from models.lstm_attention_7 import AttentionLstm as AttentionLstm_7
from database.tables import Stock_Info, Stock_Predictions


def run_models():

    stock_q= select(Stock_Info).where(Stock_Info.stock_id == 1)
    Session = sessionmaker(bind=get_engine())
    session = Session()
    output = session.connection().execute(stock_q).all()
    s_open = []
    s_close = []
    s_high = []
    s_low = []
    s_volume = []
    for row in output:
        s_open.append(row[3])
        s_close.append(row[1])
        s_high.append(row[4])
        s_low.append(row[5])
        s_volume.append(row[2])
    data = {'Close': s_close, 'Open': s_open, 'High':s_high, 'Low':s_low, 'Volume':s_volume}
    data = pd.DataFrame(data)
    data_copy = copy.deepcopy(data)
    one_day = []
   

    one_day.append(AttentionLSTM(AttentionLstm(), "attention_lstm", "TSLA"))

    prediction = ForecastModels(one_day)
 
    prediction.train_all(copy.deepcopy(data))

    pred = prediction.run_all(copy.deepcopy(data), 7)

    model_1 = pred[0]
    model_2 = pred[0]
    model_3 = pred[0]
    model_4 = pred[0]
    model_5 = pred[0]

    new_row = Stock_Predictions(stock_id=1, model_1=json.dumps(model_1), model_2=json.dumps(model_2), model_3=json.dumps(model_3), model_4=json.dumps(model_4), model_5=json.dumps(model_5), created_at = date.today())
    session.add(new_row)
    try:
        session.commit()
    except  exc.SQLAlchemyError as e:
        print(e)
    session.close()

def model_thread():
    thread = threading.Thread(target=run_models)
    thread.start()
