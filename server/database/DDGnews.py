from selenium import webdriver
import torch
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker
from tables import Stocks, Stock_Info, Base
from datetime import date
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sqlalchemy import create_engine, select
import os
import time
import random
import torch
from sentimentHF import sentiment_model
from duckduckgo_search import DDGS


load_dotenv()
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
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"

def stock_scrap(stock_id, engine):

    Session = sessionmaker(bind=engine)
    session = Session()
    # Search for the stock by using the stock id
    stock_q = select(Stocks).filter(Stocks.stock_id == stock_id)
    stock = session.connection().execute(stock_q).first()
    # Get all the dates out of the table to get news data for
    stock_data_a = select(Stock_Info.time_stamp, Stock_Info.news_data).filter(Stock_Info.stock_id == stock_id)
    stock_data = session.connection().execute(stock_data_a).all()

    # using google search news results to get the news data from a specific date
    # by doing this for all dates in the stock_info table we can fill the table for time series analysis
    
    for day in stock_data:
        # This is to skip days that have already been calculated allowing the program to not have to run all at once
        if day.news_data == None:
            # Going to fast for API this is slow enough to get 500+ values without an API block
            time.sleep(5)          
            try:
                results = DDGS().text(f"{stock.search} news", max_results=5, timelimit=f"{day.time_stamp}..{day.time_stamp}")
                print(day.time_stamp)
                print(results) 

                              
                
                tensors = 0
                tensor = torch.tensor([[0,0,0]])
                
                for r in results:
                    try:
                        # Model goes here
                        article = r['title']+ " " + r['body']
                        if len(article) > 512:
                            logit = sentiment_model(article[:512])
                            tensor = torch.add(tensor, logit)
                            articles.append(article)
                        else:
                            logit = sentiment_model(article)
                            tensor = torch.add(tensor, logit)
                        tensors+=1
                    except Exception as e:
                        print(e)
                        continue
                
                if tensors > 0:
                    # average tesor for the day
                    answer = torch.div(tensor, tensors)
                    update_row = update(Stock_Info).where(Stock_Info.stock_id == stock_id).where(Stock_Info.time_stamp == day.time_stamp).values(news_data = (answer[0][0]*-1+answer[0][2]).item())
                    print(session.connection().execute(update_row))
                
                session.commit()
                session.flush() 

                
            except Exception as e:
                print(e)
                continue
        

    session.close()
    
