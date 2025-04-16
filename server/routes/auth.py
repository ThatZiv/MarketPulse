# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import os
import flask_jwt_extended as jw
import requests
from flask import Blueprint, Response, jsonify, request, send_file
from sqlalchemy import desc, select
from sqlalchemy.exc import SQLAlchemyError
from database.tables import Stock_Info, Stocks
from routes.llm import llm_bp
from routes.access import get_forcasts, create_session
from cache import cache

auth_bp = Blueprint('auth', __name__)
LOGODEV_API_KEY = os.getenv('LOGODEV_API_KEY')
auth_bp.register_blueprint(llm_bp)

def dump_datetime(value):
    '''Change the data format into something that can be stored in db'''
    if value is None:
        return None
    return [value.strftime("%x"), value.strftime("%H:%M:%S")]

def stock_query_single(query, session):
    '''Function to make a query that returns the first row.  Takes the query and connection as args'''
    return session.connection().execute(query).first()

def stock_query_all(query, session):
    '''Function to make a query that returns all rows.  Takes the query and connection as args'''
    return session.connection().execute(query).all()

@auth_bp.route('/logo', methods=['GET'])
@jw.jwt_required()
def ticker_logo():
    '''Route to return stock images used on dashboard'''
    ticker = request.args.get('ticker')
    cache_dir = f"{os.getcwd()}/public/cache"
    if not ticker:
        return {"error": "Ticker parameter is required"}, 400
    # sanitize input
    ticker=ticker.replace('/', '').replace('..', '')\
        .replace(' ', '').replace('\\', '').replace('..', '')
    loc = f"{cache_dir}/{ticker}.png"
    if not os.path.exists(loc):
        base_url = "https://img.logo.dev/ticker/"
        url=f'{ticker}?token={LOGODEV_API_KEY}&size=300&format=png&fallback=monogram'
        r = requests.get(base_url+url, timeout=10)
        if r.status_code == 200:
            with open(loc, 'wb') as f:
                f.write(r.content)
        else:
            base_url = "https://ui-avatars.com/api/?name="
            url = f"{ticker}&format=png&size=300&background=a9a9a9&length=4"
            fallback = requests.get(base_url+url, timeout=5)
            with open(loc, 'wb') as f:
                f.write(fallback.content)
    return send_file(f'{cache_dir}/{ticker}.png', mimetype='image/png')


@auth_bp.route('stockchart', methods=['GET'])
@jw.jwt_required()
def chart():
    '''Return stock_info table to be used for creating the stock chart. Takes stock ticker string as an arg'''
    ticker = request.args['ticker']
    limit = request.args.get('limit', 7)
    if not ticker or not limit:
        return Response(status=400, mimetype='application/json')
    cache_value = cache.get("Chart_"+ticker)

    if cache_value is not None:
        return cache_value
    if request.method == 'GET':
        session = None
        try:
            session = create_session()
            s_id = select(Stocks).where(Stocks.stock_ticker == ticker)
            output_id = stock_query_single(s_id, session)
            #validating the ticker from the frontend
            if output_id:
                stock_data = select(Stock_Info).where(Stock_Info.stock_id == output_id.stock_id)\
                    .order_by(desc(Stock_Info.time_stamp)).limit(limit)
                output = stock_query_all(stock_data, session)

                json_output = []
                for i in output:
                    json_output.append({'stock_id' : i.stock_id,
                                        'stock_close' : i.stock_close,
                                        'stock_volume' : i.stock_volume,
                                        'stock_open' : i.stock_open,
                                        'stock_high' : i.stock_high, 'stock_low' : i.stock_low,
                                        'hype_meter'  : i.sentiment_data,
                                        'impact_factor': i.news_data,
                                        'time_stamp' : dump_datetime(i.time_stamp) })
                json_output.reverse()
                session.close()
                cache.set(f"Chart_{ticker}", jsonify(json_output), timeout = 12000)
                print("Cache Set")
                return jsonify(json_output)
            session.close()
            return Response(status=400, mimetype='application/json')
        except (SQLAlchemyError, requests.RequestException) as e:
            print(e)
            session.close()
            return Response(status=503)
        finally:
            if session:
                session.close()
    return Response(status=401, mimetype='application/json')


@auth_bp.route('forecast', methods=['GET'])
@jw.jwt_required()
def forecast_route():
    '''Route to return model forecast data. Takes ticker string and lookback number as args'''
    ticker = request.args.get('ticker')
    lookback = request.args['lookback']

    if not ticker or not lookback:
        return Response(status=400, mimetype='application/json')

    cache_value = cache.get("forecast_"+ticker+lookback)

    if cache_value is not None:
        return cache_value

    if request.method == 'GET':
        response = get_forcasts(ticker=ticker, lookback=lookback)

        if response is None:
            return Response(status=400, mimetype='application/json')
        return jsonify(response)
    return Response(status=503, mimetype='application/json')
