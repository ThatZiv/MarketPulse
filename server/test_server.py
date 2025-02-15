#Created as a test server to allow for testing api's without adding them to the main server
# pylint: disable=all

from flask import Flask, session, redirect, url_for, request, jsonify, Response
from dotenv import load_dotenv
import os
import json
import numpy as np
from datetime import date
from flask_cors import CORS
from database.tables import Base, Account, User_Stocks, Stocks, Stock_Info
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, select, update
from database.newsapi import news_search
from database.reddit import reddit_request, add_to_database, daily_reddit_request
load_dotenv()

def create_app():

    app = Flask(__name__)
    
    from routes.auth import auth_bp

    app.config["JWT_SECRET_KEY"] = os.environ.get("SUPABASE_JWT")
    
    app.register_blueprint(auth_bp, url_prefix='/auth')
    
    return app


if __name__ == '__main__':  

    app = create_app()
    CORS(app)

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

    Base.metadata.create_all(engine)

    #reddit_request("stocks", "Tesla")
    #add_to_database(reddit_request("stocks", "Rivian"), engine, 5)
    daily_reddit_request("stocks", "Tesla", engine, 1)
    #print(news_search("Rivian", engine, 5))
    app.run(debug=True, host='0.0.0.0')


