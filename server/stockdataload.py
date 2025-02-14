# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
from sqlalchemy.orm import sessionmaker
from database.yfinanceapi import bulk_stock_data
from database.tables import Stock_Info

def load_data(engine):
    session = sessionmaker(bind=engine)
    session = session()
    stocks = []
    stock1 = bulk_stock_data("TSLA")
    stocks[]
    stocks.append(stock1["Close"].astype(float).tolist())
    stocks.append(stock1["Open"].astype(float).tolist())
    stocks.append(stock1["Low"].astype(float).tolist())
    stocks.append(stock1["High"].astype(float).tolist())
    stocks.append(stock1["Volume"].astype(int).tolist())

    for i in range(len(stock1['Close'])):
        new_row = Stock_Info(stock_id = 1, stock_close = stocks[0][i],
                            stock_volume = stocks[4][i], stock_open=stocks[1][i],
                            stock_high = stocks[3][i], stock_low=stocks[2][i],
                            sentiment_data=0, time_stamp=stock1["Date"][i], news_data = 0)
        session.add(new_row)
    stock2 = bulk_stock_data("F")
    stocks = []
    stocks.append(stock2["Close"].astype(float).tolist())
    stocks.append(stock2["Open"].astype(float).tolist())
    stocks.append(stock2["Low"].astype(float).tolist())
    stocks.append(stock2["High"].astype(float).tolist())
    stocks.append(stock2["Volume"].astype(int).tolist())

    for i in range(len(stock2['Close'])):
        new_row = Stock_Info(stock_id = 2, stock_close = stocks[0][i],
                            stock_volume = stocks[4][i], stock_open=stocks[1][i],
                            stock_high =stocks[3][i], stock_low=stocks[2][i],
                            sentiment_data=0, time_stamp=stock2["Date"][i], news_data = 0)
        session.add(new_row)

    stock3 = bulk_stock_data("GM")
    stocks = []
    stocks[0] = stock3["Close"].astype(float).tolist())
    stocks[1] = stock3["Open"].astype(float).tolist())
    stocks[2] = stock3["Low"].astype(float).tolist())
    stocks[3] = stock3["High"].astype(float).tolist())
    stocks[4] = stock3["Volume"].astype(int).tolist())

    for i in range(len(stock3['Close'])):
        new_row = Stock_Info(stock_id = 3, stock_close = stocks[0][i],
                            stock_volume = stocks[4][i], stock_open=stocks[1][i],
                            stock_high = stocks[3][i], stock_low=stocks[2][i],
                            sentiment_data=0, time_stamp=stock3["Date"][i], news_data = 0)
        session.add(new_row)

    stock4 = bulk_stock_data("TM")
    stocks = []
    stocks.append(stock4["Close"].astype(float).tolist())
    stocks.append(stock4["Open"].astype(float).tolist())
    stocks.append(stock4["Low"].astype(float).tolist())
    stocks.append(stock4["High"].astype(float).tolist())
    stocks.append(stock4["Volume"].astype(int).tolist())

    for i in range(len(stock4['Close'])):
        new_row = Stock_Info(stock_id = 4, stock_close = stocks[0][i],
                            stock_volume = stocks[4][i], stock_open=stocks[1][i],
                            stock_high =stocks[3][i], stock_low=stocks[2][i],
                            sentiment_data=0, time_stamp=stock4["Date"][i], news_data = 0)
        session.add(new_row)

    stock5 = bulk_stock_data("RIVN")
    stocks = []
    stocks.append(stock5["Close"].astype(float).tolist())
    stocks.append(stock5["Open"].astype(float).tolist())
    stocks.append(stock5["Low"].astype(float).tolist())
    stocks.append(stock5["High"].astype(float).tolist())
    stocks.append(stock5["Volume"].astype(int).tolist())

    for i in range(len(stock5['Close'])):
        new_row = Stock_Info(stock_id = 5, stock_close = stocks[0][i],
                            stock_volume = stocks[4][i], stock_open=stocks[1][i],
                            stock_high = stocks[3][i], stock_low=stocks[2][i],
                            sentiment_data=0, time_stamp=stock5["Date"][i], news_data = 0)
        session.add(new_row)
    session.commit()
