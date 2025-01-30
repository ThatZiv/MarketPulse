from flask import Flask, session, redirect, url_for, request, jsonify
from dotenv import load_dotenv
import os
import json
import numpy as np
from supabase import create_client, Client
import flask_jwt_extended as jw
from stockdataload import loadData
from flask_cors import CORS\

from sqlalchemy import create_engine, select

from flask_jwt_extended import JWTManager,jwt_required
from database.tables import Base, Account, User_Stocks, Stocks, Stock_Info
from sqlalchemy.orm import sessionmaker
load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
#supabase: Client = create_client(url, key)

def dump_datetime(value):
    if value is None:
        return None
    return [value.strftime("%Y-%m-%d"), value.strftime("%H:%M:%S")]


def create_app():
    app = Flask(__name__)
  
    from routes.auth import auth_bp

    app.config["JWT_SECRET_KEY"] = os.environ.get("SUPABASE_JWT")
    jwt = JWTManager(app)
    app.register_blueprint(auth_bp, url_prefix='/auth')
    
    return app


if __name__ == '__main__':  

    app = create_app()
    CORS(app)
    jwt = jw.JWTManager()
    jwt.init_app(app) 


    USER = os.getenv("user")
    PASSWORD = os.getenv("password")
    HOST = os.getenv("host")
    PORT = os.getenv("port")
    DBNAME = os.getenv("dbname")

    DATABASE_URL = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}?sslmode=require"

    engine = create_engine(DATABASE_URL)
    try:
        with engine.connect() as connection:
            print("Connection successful!")
    except Exception as e:
        print(f"Failed to connect: {e}")

    # This only needs to run once but running on start will make sure all tables are in the database.
    # It appears that editing existing tables requires dropping the table or useing altertable sql.
    Base.metadata.create_all(engine)
    
    # Load 2 years of of historic stock data into the database due to running twice in dev this will crash the server after successful upload
    #loadData(engine)

    @app.route('/test', methods=['GET', 'POST'])
    @jwt_required()
    def route():
        return jsonify('hello')
    
    @app.route('/stockchart/', methods=['GET'])
    @jwt_required()
    def chart():
        if request.method == 'GET':
            Session = sessionmaker(bind=engine)
            session = Session()

            ticker = request.args['ticker']
            sId = select(Stocks).where(Stocks.stock_ticker == ticker)
            print(sId)
            output = session.connection().execute(sId).first()
            if(output):
                print(output.stock_id)
                stock_data = select(Stock_Info).where(Stock_Info.stock_id == output.stock_id)
                print(stock_data)
                output = session.connection().execute(stock_data).all()
            
                json_output = []
                for i in output:
                    json_output.append({'stock_id' : i.stock_id, 'stock_close' : i.stock_close, 'stock_volume' : i.stock_volume, 'stock_open' : i.stock_open, 'stock_high' : i.stock_high, 'stock_low' : i.stock_low, 'sentiment_data'  : i.sentiment_data, 'time_stamp'      : dump_datetime(i.time_stamp) })
                return jsonify(json_output)
            else:
                return jsonify("Invalid ticker")
        else:
            return jsonify('Failed')
    
    
    app.run(debug=True, host='0.0.0.0')