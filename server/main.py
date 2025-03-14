# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import os
import flask_jwt_extended as jw
from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv
from flask_cors import CORS
from sqlalchemy import create_engine, select, exc
from sqlalchemy.orm import sessionmaker
from flask_jwt_extended import jwt_required
from flask_caching import Cache
from flask_apscheduler import APScheduler
from engine import get_engine
from database.tables import Base, Stocks
from database.yfinanceapi import real_time_data
from routes.auth import auth_bp
from load_data import stock_thread

load_dotenv()

LEGACY = os.environ.get("LEGACY") == "true"
if not LEGACY:
    from models.run_models import model_thread

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

#supabase: Client = create_client(url, key)
# https://stackoverflow.com/questions/7102754/jsonify-a-sqlalchemy-result-set-in-flask
def dump_datetime(value):
    if value is None:
        return None
    return [value.strftime("%x"), value.strftime("%H:%M:%S")]

def create_app():

    app = Flask(__name__)
    
    app.config["JWT_SECRET_KEY"] = os.environ.get("SUPABASE_JWT")

    app.register_blueprint(auth_bp, url_prefix='/auth')
    CORS(app, supports_credentials=True)
    jwt = jw.JWTManager()
    jwt.init_app(app)
    
    Base.metadata.create_all(get_engine())
    scheduler = APScheduler()
    scheduler.init_app(app)
    scheduler.add_job(func=stock_thread, trigger='cron', hour='23', id="load_stocks")
    if not LEGACY:
        scheduler.add_job(func=model_thread, trigger='cron', hour='0', id="model_predictions")
    scheduler.start()
    return app

app = create_app()
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/test', methods=['GET', 'POST'])
@jwt_required()
def route():
    return jsonify('hello')

@app.route('/stockrealtime', methods = ['GET'] )
@jwt_required()
def realtime():
    if request.method == 'GET':
        session_a = sessionmaker(bind=get_engine())
        session = session_a()
        ticker = request.args['ticker']
        s_id = select(Stocks).where(Stocks.stock_ticker == ticker)
        output_id = session.connection().execute(s_id).first()
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
    
    app.run(debug=True, host='0.0.0.0')
