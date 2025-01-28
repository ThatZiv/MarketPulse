from flask import Flask, session, redirect, url_for, request
from dotenv import load_dotenv
import os
from supabase import create_client, Client
import flask_jwt_extended as jw
from flask_cors import CORS\

from sqlalchemy import create_engine

from flask_jwt_extended import JWTManager
from database.tables import Base, Account, User_Stocks, Stocks, Stock_Info
from sqlalchemy.orm import sessionmaker


load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
#supabase: Client = create_client(url, key)



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
    
    #@app.route('/')
    #def route():
    #    return 'hello'
    
    app.run(debug=True, host='0.0.0.0')