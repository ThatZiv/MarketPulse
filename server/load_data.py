# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import threading
from datetime import date, datetime
import requests
from sqlalchemy import select, func, exc
from sqlalchemy.orm import sessionmaker
from engine import get_engine
from database.ddg_news import add_news
from database.reddit import daily_reddit_request
from database.yfinanceapi import add_daily_data
from database.tables import Stocks, Stock_Info

def stock_thread():
    thread = threading.Thread(target=load_stocks)
    thread.start()

def load_stocks():
    print("Starting job")

    #Find the most recent entry for all stocks joined to the stock infromation
    stock_q=select(func.max(Stock_Info.time_stamp), Stock_Info.stock_id,
                        Stocks.stock_ticker, Stocks.search).select_from(Stock_Info).join(Stocks,
                        Stocks.stock_id == Stock_Info.stock_id).group_by(Stock_Info.stock_id,
                        Stocks.stock_ticker, Stocks.search)
    session = sessionmaker(bind=get_engine())
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
