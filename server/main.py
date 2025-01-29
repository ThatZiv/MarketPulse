from flask import Flask, session, redirect, url_for, request, jsonify
from dotenv import load_dotenv
import os
from supabase import create_client, Client
import flask_jwt_extended as jw
from flask_cors import CORS\

from sqlalchemy import create_engine, select

from flask_jwt_extended import JWTManager,jwt_required
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
    
    # Code to insert the five stocks on server start into the stocks table
    # Update the five stocks in the table if they are changed
    Session = sessionmaker(bind=engine)
    session = Session()
    
    newStock = Stocks(stock_id = 1, stock_ticker = "NVDA")
    q1 = select(Stocks).filter_by(stock_id = newStock.stock_id)
    s1 = session.scalars(q1).all()

    if s1 is None:
        session.add(newStock)
    else:
        session.delete(s1[0])
        session.add(newStock)

    newStock = Stocks(stock_id = 2, stock_ticker = "SHOP")
    q1 = select(Stocks).filter_by(stock_id = newStock.stock_id)
    s1 = session.scalars(q1).all()

    if s1 is None:
        session.add(newStock)
    else:
        session.delete(s1[0])
        session.add(newStock)
    
    newStock = Stocks(stock_id = 3, stock_ticker = "GTLB")
    q1 = select(Stocks).filter_by(stock_id = newStock.stock_id)
    s1 = session.scalars(q1).all()

    if s1 is None:
        session.add(newStock)
    else:
        session.delete(s1[0])
        session.add(newStock)
    
    newStock = Stocks(stock_id = 4, stock_ticker = "NET")
    q1 = select(Stocks).filter_by(stock_id = newStock.stock_id)
    s1 = session.scalars(q1).all()

    if s1 is None:
        session.add(newStock)
    else:
        session.delete(s1[0])
        session.add(newStock)
    
    newStock = Stocks(stock_id = 5, stock_ticker = "RDDT")
    q1 = select(Stocks).filter_by(stock_id = newStock.stock_id)
    s1 = session.scalars(q1).all()

    if s1 is None:
        session.add(newStock)
    else:
        session.delete(s1[0])
        session.add(newStock)
    
    session.commit()
    
    return app


if __name__ == '__main__':  


    app = create_app()
    CORS(app)
    jwt = jw.JWTManager()
    jwt.init_app(app) 

    @app.route('/test', methods=['GET', 'POST'])
    @jwt_required()
    def route():
        return jsonify('hello')
    
    app.run(debug=True, host='0.0.0.0')