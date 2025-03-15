# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import os
import json

import flask_jwt_extended as jw
import requests
from flask import Blueprint, Response, jsonify, request, send_file
from sqlalchemy import desc, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from database.tables import Stock_Info, Stock_Predictions, Stocks
from engine import get_engine
from routes.llm import llm_bp

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
@auth_bp.route('stockchart', methods=['GET'])
@jw.jwt_required()
def chart():
    if request.method == 'GET':
        try:
            session = sessionmaker(bind=get_engine())
            session = session()
            ticker = request.args['ticker']
            limit = request.args.get('limit', 7)
            s_id = select(Stocks).where(Stocks.stock_ticker == ticker)
            output_id = session.connection().execute(s_id).first()
            #validating the ticker from the frontend
            if output_id:
                stock_data = select(Stock_Info).where(Stock_Info.stock_id == output_id.stock_id)\
                    .order_by(desc(Stock_Info.time_stamp)).limit(limit)
                output = session.connection().execute(stock_data).all()
                json_output = []
                for i in output:
                    json_output.append({'stock_id' : i.stock_id,
                                        'stock_close' : i.stock_close,
                                        'stock_volume' : i.stock_volume,
                                        'stock_open' : i.stock_open,
                                        'stock_high' : i.stock_high, 'stock_low' : i.stock_low,
                                        'sentiment_data'  : i.sentiment_data,
                                        'news_data': i.news_data,
                                        'time_stamp' : dump_datetime(i.time_stamp) })
                json_output.reverse()
                session.close()
                return jsonify(json_output)
            session.close()
            return Response(status=400, mimetype='application/json')
        except (SQLAlchemyError, requests.RequestException) as e:
            print(e)
            session.close()
            return Response(status=500)
        finally:
            if session:
                session.close()
    session.close()
    return Response(status=401, mimetype='application/json')

# Has been tested with out any data
@auth_bp.route('forecast', methods=['GET'])
@jw.jwt_required()
def forecast_route():
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
            forecast = select(Stock_Predictions).where(Stock_Predictions.stock_id == output_id.stock_id).order_by(desc(Stock_Predictions.created_at))
            output = session.connection().execute(forecast).first()
            
            out = []
            columns = [column.key for column in Stock_Predictions.__table__.columns if column.key.startswith("model_")]
            # pylint: disable=protected-access
            output_dict = output._mapping
            # pylint: enable=protected-access
            for column in columns:
                forecast_data = output_dict[column]
                out.append(json.loads(forecast_data))
            session.close()
            return jsonify({
                "stock_id": output_dict["stock_id"],
                "created_at": output_dict["created_at"],
                "output": out
            })
        session.close()
        return Response(status=400, mimetype='application/json')
    session.close()
    return Response(status=500, mimetype='application/json')
