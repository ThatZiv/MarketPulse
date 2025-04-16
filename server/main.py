# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import os
import threading
import flask_jwt_extended as jw
from flask import Flask, request, Response
from dotenv import load_dotenv
from flask_cors import CORS
from sqlalchemy import select, exc
from sqlalchemy.orm import sessionmaker
from flask_jwt_extended import jwt_required
from flask_apscheduler import APScheduler
from engine import get_engine, global_engine
from database.tables import Stocks
from database.yfinanceapi import real_time_data
from routes.auth import auth_bp
from load_data import load_stocks
from cache import cache


load_dotenv()

LEGACY = os.environ.get("LEGACY") == "true"
if not LEGACY:
    from models.run_models import run_models
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

#supabase: Client = create_client(url, key)
# https://stackoverflow.com/questions/7102754/jsonify-a-sqlalchemy-result-set-in-flask
def dump_datetime(value):
    if value is None:
        return None
    return [value.strftime("%x"), value.strftime("%H:%M:%S")]

def stock_query_single(query, session):
    return session.connection().execute(query).first()

def create_app():

    ap = Flask(__name__)
    ap.config["MUTEX"] = threading.Lock()
    ap.config["JWT_SECRET_KEY"] = os.environ.get("SUPABASE_JWT")
    cache.init_app(ap)
    ap.register_blueprint(auth_bp, url_prefix='/auth')
    CORS(ap, supports_credentials=True)
    jwt = jw.JWTManager()
    jwt.init_app(ap)
    scheduler = APScheduler()
    scheduler.init_app(ap)
    def stock_job():
        with app.app_context():
            load_stocks()

    def model_job():
        with app.app_context():
            if not LEGACY:
                run_models()


    scheduler.add_job(func=stock_job, trigger='cron', hour='21', minute ='0' ,id="load_stocks")
    if not LEGACY:
        scheduler.add_job(func=model_job, trigger='cron', hour='21', minute ='30' ,id="model_predictions")
    scheduler.start()

    return ap

app = create_app()

def create_session():

    try:
        session = sessionmaker(bind=global_engine())
    except exc.OperationalError:
        with app.config["MUTEX"]:
            session = sessionmaker(bind=get_engine())
    return session()

@app.route('/stockrealtime', methods = ['GET'] )
@jwt_required()
def realtime():
    '''Handles a request from the page to make access get realtume stock data for a particular stock'''
    if request.method == 'GET':
        session = create_session()
        ticker = request.args['ticker']
        s_id = select(Stocks).where(Stocks.stock_ticker == ticker)
        try:
            output_id = stock_query_single(s_id, session)
        except exc.OperationalError:
            return Response(status=503, mimetype='application/json')
        if output_id:
            cache_output = cache.get(output_id)
            if cache_output is not None:
                print("Cache Output")
                return cache_output

            stock_data = real_time_data(ticker)
            close_rt =stock_data["Close"].astype(float).tolist()
            open_rt = stock_data["Open"].astype(float).tolist()
            low_rt = stock_data["Low"].astype(float).tolist()
            high_rt = stock_data["High"].astype(float).tolist()
            volume_rt = stock_data["Volume"].astype(int).tolist()
            json_output = []
            for i in range(len(stock_data['Close'])):
                json_output.append({'stock_id' : output_id.stock_id,
                                    'stock_close' : close_rt[i],
                                    'stock_volume' : volume_rt[i],
                                    'stock_open' : open_rt[i],
                                    'stock_high' : high_rt[i],
                                    'stock_low' : low_rt[i],
                                    'sentiment_data'  : 0,
                                    'news_data': 0,
                                    'time_stamp' : dump_datetime(stock_data["Datetime"][i])})
            cache.set(output_id, json_output, timeout = 60*5)
            print("Reset Cache")
            session.close()
            return json_output
        session.close()
        return Response(status=400, mimetype='application/json')
    return Response(status=401, mimetype='application/json')

if __name__ == '__main__':

    app.run(debug=LEGACY, host='0.0.0.0')
