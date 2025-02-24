# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import os
from datetime import date, datetime
import requests
import flask_jwt_extended as jw
from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv
from flask_cors import CORS
from sqlalchemy import create_engine, select, func, exc
from sqlalchemy.orm import sessionmaker
from flask_jwt_extended import jwt_required
from flask_caching import Cache
from flask_apscheduler import APScheduler
from database.ddg_news import add_news
from database.reddit import daily_reddit_request
from database.tables import Base, Stocks, Stock_Info, Stock_Predictions
from database.yfinanceapi import add_daily_data, real_time_data
from routes.auth import auth_bp

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
    def load_stocks():
        print("Starting job")

        #Find the most recent entry for all stocks joined to the stock infromation
        stock_q=select(func.max(Stock_Info.time_stamp), Stock_Info.stock_id,
                        Stocks.stock_ticker, Stocks.search).select_from(Stock_Info).join(Stocks,
                        Stocks.stock_id == Stock_Info.stock_id).group_by(Stock_Info.stock_id,
                        Stocks.stock_ticker, Stocks.search)
        session = sessionmaker(bind=engine)
        session = session()
        recent = session.connection().execute(stock_q).all()
        #print(recent)

        for stock in recent:
            extra_data = []
            stock_data = []
            diff = date.today() - stock[0]
            if diff.days > 0:
                extra_data = add_daily_data(stock[2], diff.days)

                # Type casting to match types that can be added to the database
                retype=[]
                retype.append(extra_data["Close"].astype(float).tolist())
                retype.append(extra_data["Open"].astype(float).tolist())
                retype.append(extra_data["Low"].astype(float).tolist())
                retype.append(extra_data["High"].astype(float).tolist())
                retype.append(extra_data["Volume"].astype(int).tolist())
                session.flush()
                stock_data = []

            # create a list of dates that need to be added
                for i in range(len(extra_data["Date"])):
                    if (datetime.strptime(extra_data["Date"][i]
                        .strftime("%b %d %H:%M:%S %Y"), "%b %d %H:%M:%S %Y")
                        -datetime(stock[0].year, stock[0].month, stock[0].day)).days>0:

                        stock_data.append({"time_stamp" : extra_data["Date"][i],
                                        "ticker": stock.stock_ticker, "search": stock.search,
                                        "stock_id" : stock.stock_id, "stock_close" : retype[0][i], 
                                        "stock_volume" : retype[4][i], "stock_open": retype[1][i], 
                                        "stock_high": retype[3][i], "stock_low":retype[2][i], 
                                        "news_data":0, "sentiment_data":0 })

            #Add the social media and news sentiment
            reddit = []
            news = []
            print(stock.stock_ticker)
            if len(stock_data)>0:
                try:
                    reddit = daily_reddit_request("Stocks", stock_data)
                    news = add_news(stock_data)
                except requests.exceptions.ConnectionError as e:
                    print(e)
            for i in stock_data:
                for r in reddit:
                    if r["time_stamp"].strftime('%Y-%m-%d')==i["time_stamp"]:
                        i["sentiment_data"] = r["answer"]

                for n in news:
                    if n["time_stamp"] == i["time_stamp"]:
                        i["news_data"] = n["news"]
                new_row = Stock_Info(stock_id = i["stock_id"], stock_close = i["stock_close"],
                                stock_volume = i["stock_volume"], stock_open=i["stock_open"],
                                stock_high = i["stock_high"], stock_low=i["stock_low"],
                                sentiment_data=i["sentiment_data"], time_stamp=i["time_stamp"],
                                news_data=i["news_data"])
                session.add(new_row)
            try:
                session.commit()
            except  exc.SQLAlchemyError as e:
                print(e)

    # For testing
    #load_stocks()
    # run the load_stocks job at a specific time.
    scheduler.add_job(func=load_stocks, trigger='cron', hour='0', id="load_stocks")
    @app.route('/test', methods=['GET', 'POST'])
    @jwt_required()
    def route():
        return jsonify('hello')

    @app.route('/stockrealtime/', methods = ['GET'] )
    @jwt_required()
    def realtime():
        if request.method == 'GET':
            session_a = sessionmaker(bind=engine)
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


    #What do we want this path to return?
    @app.route('/stockchart/', methods=['GET'])
    @jwt_required()
    def chart():
        if request.method == 'GET':
            session_a = sessionmaker(bind=engine)
            session = session_a()
            ticker = request.args['ticker']
            s_id = select(Stocks).where(Stocks.stock_ticker == ticker)
            output_id = session.connection().execute(s_id).first()
            #validating the ticker from the frontend
            if output_id :
                stock_data = select(Stock_Info).where(Stock_Info.stock_id == output_id.stock_id)
                output = session.connection().execute(stock_data).all()
                json_output = []
                for i in output:
                    json_output.append({'stock_id' : i.stock_id,
                                        'stock_close' : i.stock_close,
                                        'stock_volume' : i.stock_volume,
                                        'stock_open' : i.stock_open,
                                        'stock_high' : i.stock_high, 'stock_low' : i.stock_low,
                                        'sentiment_data'  : i.sentiment_data,
                                        'time_stamp' : dump_datetime(i.time_stamp) })
                return jsonify(json_output)
            return Response(status=400, mimetype='application/json')

        return Response(status=401, mimetype='application/json')

    @app.route('/forcast', methods=['Get'])
        @jw.jwt_required()
        def forcast():
            ticker = request.args.get('ticker')
            if not ticker:
                return Response(status=400, mimetype='application/json')
            if request.method == 'GET':
                session = sessionmaker(bind=engine)
                session = session()
                ticker = request.args['ticker']
                s_id = select(Stocks).where(Stocks.stock_ticker == ticker)
                output_id = session.connection().execute(s_id).first()
                if output_id :
                    forcast = select(Stock_Predictions).where(Stock_Predictions.stock_id == output_id.stock_id).order_by(Stock_Predictions.created_at.desc()).limit(7)
                    output = session.connection().execute(forcast).all()
                    return jsonify(output)

                return Response(status=400, mimetype='application/json') 
        
            return Response(status=500, mimetype='application/json')
    
    
    app.run(debug=True, host='0.0.0.0')
