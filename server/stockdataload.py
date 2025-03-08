# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
from sqlalchemy.orm import sessionmaker
from database.tables import Stock_Info
import yfinance as yf

def load_data(engine):
    session = sessionmaker(bind=engine)
    session = session()
    '''
    stocks = []
    stock1 = yf.download("TSLA", start='2020-01-29', end='2023-01-29')
    stock1.reset_index(inplace=True)
    stocks.append(stock1["Close"]['TSLA'].astype(float).tolist())
    stocks.append(stock1["Open"]['TSLA'].astype(float).tolist())
    stocks.append(stock1["Low"]['TSLA'].astype(float).tolist())
    stocks.append(stock1["High"]['TSLA'].astype(float).tolist())
    stocks.append(stock1["Volume"]['TSLA'].astype(int).tolist())
    
    for i in range(len(stock1['Close'])):
        new_row = Stock_Info(stock_id = 1, stock_close = stocks[0][i],
                            stock_volume = stocks[4][i], stock_open=stocks[1][i],
                            stock_high = stocks[3][i], stock_low=stocks[2][i],
                            sentiment_data=0, time_stamp=stock1["Date"][i], news_data = 0)
        session.add(new_row)
    stock2 = yf.download("F", start='2020-01-29', end='2023-01-29')
    stocks = []
    stock2.reset_index(inplace=True)
    stocks.append(stock2["Close"]['F'].astype(float).tolist())
    stocks.append(stock2["Open"]['F'].astype(float).tolist())
    stocks.append(stock2["Low"]['F'].astype(float).tolist())
    stocks.append(stock2["High"]['F'].astype(float).tolist())
    stocks.append(stock2["Volume"]['F'].astype(int).tolist())

    for i in range(len(stock2['Close'])):
        new_row = Stock_Info(stock_id = 2, stock_close = stocks[0][i],
                            stock_volume = stocks[4][i], stock_open=stocks[1][i],
                            stock_high =stocks[3][i], stock_low=stocks[2][i],
                            sentiment_data=0, time_stamp=stock2["Date"][i], news_data = 0)
        session.add(new_row)

    stock3 = yf.download("GM", start='2020-01-29', end='2023-01-29')
    stocks = []
    stock3.reset_index(inplace=True)
    stocks.append(stock3["Close"]['GM'].astype(float).tolist())
    stocks.append(stock3["Open"]['GM'].astype(float).tolist())
    stocks.append(stock3["Low"]['GM'].astype(float).tolist())
    stocks.append(stock3["High"]['GM'].astype(float).tolist())
    stocks.append(stock3["Volume"]['GM'].astype(int).tolist())

    for i in range(len(stock3['Close'])):
        new_row = Stock_Info(stock_id = 3, stock_close = stocks[0][i],
                            stock_volume = stocks[4][i], stock_open=stocks[1][i],
                            stock_high = stocks[3][i], stock_low=stocks[2][i],
                            sentiment_data=0, time_stamp=stock3["Date"][i], news_data = 0)
        session.add(new_row)

    stock4 = yf.download("TM", start='2020-01-29', end='2023-01-29')
    stocks = []
    stock4.reset_index(inplace=True)
    stocks.append(stock4["Close"]['TM'].astype(float).tolist())
    stocks.append(stock4["Open"]['TM'].astype(float).tolist())
    stocks.append(stock4["Low"]['TM'].astype(float).tolist())
    stocks.append(stock4["High"]['TM'].astype(float).tolist())
    stocks.append(stock4["Volume"]['TM'].astype(int).tolist())

    for i in range(len(stock4['Close'])):
        new_row = Stock_Info(stock_id = 4, stock_close = stocks[0][i],
                            stock_volume = stocks[4][i], stock_open=stocks[1][i],
                            stock_high =stocks[3][i], stock_low=stocks[2][i],
                            sentiment_data=0, time_stamp=stock4["Date"][i], news_data = 0)
        session.add(new_row)
    '''
    stock5 = yf.download("STLA", start='2020-01-29', end='2025-02-25')
    stocks = []
    stock5.reset_index(inplace=True)
    stocks.append(stock5["Close"]['STLA'].astype(float).tolist())
    stocks.append(stock5["Open"]['STLA'].astype(float).tolist())
    stocks.append(stock5["Low"]['STLA'].astype(float).tolist())
    stocks.append(stock5["High"]['STLA'].astype(float).tolist())
    stocks.append(stock5["Volume"]['STLA'].astype(int).tolist())

    for i in range(len(stock5['Close'])):
        new_row = Stock_Info(stock_id = 5, stock_close = stocks[0][i],
                            stock_volume = stocks[4][i], stock_open=stocks[1][i],
                            stock_high = stocks[3][i], stock_low=stocks[2][i],
                            sentiment_data=0, time_stamp=stock5["Date"][i], news_data = 0)
        session.add(new_row)
    session.commit()
    session.close()
