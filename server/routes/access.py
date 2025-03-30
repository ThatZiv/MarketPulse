# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import json
import copy
from sqlalchemy.orm import sessionmaker
from sqlalchemy import desc, select, exc
from flask import  current_app
from database.tables import Stock_Predictions, Stocks
from engine import get_engine, global_engine
from cache import cache

def stock_query_single(query, session):
    return session.connection().execute(query).first()

def stock_query_all(query, session):
    return session.connection().execute(query).all()

def create_session():
    try:
        session = sessionmaker(bind=global_engine())
    except exc.OperationalError:
        with current_app.config["MUTEX"]:
            session = sessionmaker(bind=get_engine())
    return session()

def get_forcasts(ticker, lookback):
    session = create_session()

    s_id = select(Stocks).where(Stocks.stock_ticker == ticker)
    output_id = stock_query_single(s_id, session)
    if output_id :
        forecast = select(Stock_Predictions).where(Stock_Predictions.stock_id == output_id.stock_id).order_by(desc(Stock_Predictions.created_at)).limit(lookback)
        output = stock_query_all(forecast, session)
        out = []
        out_array = []
        columns = [column.key for column in Stock_Predictions.__table__.columns if column.key.startswith("model_")]
        session.close()
        for o in output:
            # pylint: disable=protected-access
            output_dict = o._mapping
            out = []
            # pylint: enable=protected-access
            for column in columns:
                forecast_data = output_dict[column]
                out.append(json.loads(forecast_data))

            # "model" for average of all models
            forecast_window = len(out[0]['forecast'])
            model_len = len(out)
            avg_model = {'name': 'average', 'forecast': [0] * forecast_window}

            for j in range(forecast_window):
                day_avg = 0
                for i in range(model_len):
                    model = out[i]
                    day_avg += model['forecast'][j]
                day_avg /= model_len
                avg_model['forecast'][j] = day_avg

            out.append(avg_model)

            out_array.append({
                "stock_id": output_dict["stock_id"],
                "created_at": output_dict["created_at"],
                "output": copy.deepcopy(out)
            })
            cache.set("forecast_"+ticker+lookback, out_array, timeout = 10000)
            return out_array
    session.close()
    return None
