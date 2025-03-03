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
from models.run_models import model_thread

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
#supabase: Client = create_client(url, key)
# https://stackoverflow.com/questions/7102754/jsonify-a-sqlalchemy-result-set-in-flask
def dump_datetime(value):
    if value is None:
        return None
    return [value.strftime("%x"), value.strftime("%H:%M:%S")]


def create_app():

    app_1 = Flask(__name__)

    app_1.config["JWT_SECRET_KEY"] = os.environ.get("SUPABASE_JWT")

    app_1.register_blueprint(auth_bp, url_prefix='/auth')

    return app_1


if __name__ == '__main__':

    app = create_app()
    CORS(app)
    jwt = jw.JWTManager()
    jwt.init_app(app)
    cache = Cache(app, config={'CACHE_TYPE': 'simple'})

    USER = os.getenv("user")
    PASSWORD = os.getenv("password")
    HOST = os.getenv("host")
    PORT = os.getenv("port")
    DBNAME = os.getenv("dbname")

    scheduler = APScheduler()
    scheduler.init_app(app)
    scheduler.start()

    DATABASE_URL = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}?sslmode=require"

    engine = create_engine(DATABASE_URL)
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            print("Connection successful!")
    except exc.OperationalError as e:
        print(e)
    except exc.TimeoutError as e:
        print(e)
    # This only needs to run once but running on
    # start will make sure all tables are in the database.
    # It appears that editing existing tables requires dropping the table or useing altertable sql.
    Base.metadata.create_all(engine)

    # For testing
    #stock_thread()
    # run the load_stocks job at a specific time.
    scheduler.add_job(func=stock_thread, trigger='cron', hour='23', id="load_stocks")
    scheduler.add_job(func=model_thread, trigger='cron', hour='0', id="model_predictions")

    @app.route('/test', methods=['GET', 'POST'])
    @jwt_required()
    def route():
        return jsonify('hello')

    @app.route('/stockrealtime/', methods = ['GET'] )
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
                                        'time_stamp' : dump_datetime(stock_data["Datetime"][i])})
                cache.set(output_id, json_output, timeout = 60)
                print("Reset Cache")
                return json_output
            return Response(status=400, mimetype='application/json')

        return Response(status=401, mimetype='application/json')

    app.run(debug=True, host='0.0.0.0')
