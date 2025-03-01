# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# app/routes/auth_routes.py

import os
import requests
from routes.llm import llm_bp
from flask import Blueprint, request, send_file, jsonify, Response
import flask_jwt_extended as jw
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from engine import get_engine
from database.tables import Stocks, Stock_Info, Stock_Predictions


auth_bp = Blueprint('auth', __name__)
LOGODEV_API_KEY = os.getenv('LOGODEV_API_KEY')

auth_bp.register_blueprint(llm_bp)

def dump_datetime(value):
    if value is None:
        return None
    return [value.strftime("%x"), value.strftime("%H:%M:%S")]


@auth_bp.route('/private', methods=['GET', 'POST'])
@jw.jwt_required()
def private():
    #current_user = jw.get_jwt_identity()
    return 'Private route'

@auth_bp.route('/logo', methods=['GET'])
@jw.jwt_required()
def ticker_logo():
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



#What do we want this path to return?
@auth_bp.route('/stockchart/', methods=['GET'])
@jw.jwt_required()
def chart():
    if request.method == 'GET':
        session_a = sessionmaker(bind=get_engine())
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

# Has been tested with out any data
@auth_bp.route('/forecast/', methods=['Get'])
@jw.jwt_required()
def forecast():
    ticker = request.args.get('ticker')
    if not ticker:
        return Response(status=400, mimetype='application/json')
    if request.method == 'GET':
        session = sessionmaker(bind=get_engine())
        session = session()
        ticker = request.args['ticker']
        s_id = select(Stocks).where(Stocks.stock_ticker == ticker)
        output_id = session.connection().execute(s_id).first()
        if output_id :
            forcast = select(Stock_Predictions).where(Stock_Predictions.stock_id == output_id.stock_id).order_by(Stock_Predictions.created_at).limit(7)
            output = session.connection().execute(forcast).all()
            return jsonify(output)

        return Response(status=400, mimetype='application/json')
    return Response(status=500, mimetype='application/json')
