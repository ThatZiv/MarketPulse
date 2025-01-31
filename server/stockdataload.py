from sqlalchemy.orm import sessionmaker
from database.yfinanceapi import bulk_stock_data
from database.tables import Base, Account, User_Stocks, Stocks, Stock_Info

def loadData(engine):
    Session = sessionmaker(bind=engine)
    session = Session()

    stock1 = bulk_stock_data("TSLA")
    close1 = stock1["Close"].astype(float).tolist()
    open1 = stock1["Open"].astype(float).tolist()
    low1 = stock1["Low"].astype(float).tolist()
    high1 = stock1["High"].astype(float).tolist()
    volume1 = stock1["Volume"].astype(int).tolist()

    for i in range(len(stock1['Close'])):
        newRow = Stock_Info(stock_id = 1, stock_close = close1[i], stock_volume = volume1[i], stock_open=open1[i], stock_high = high1[i], stock_low=low1[i], sentiment_data=0, time_stamp=stock1["Date"][i])
        session.add(newRow)
    
    stock2 = bulk_stock_data("F")
    close2 = stock2["Close"].astype(float).tolist()
    open2 = stock2["Open"].astype(float).tolist()
    low2 = stock2["Low"].astype(float).tolist()
    high2 = stock2["High"].astype(float).tolist()
    volume2 = stock2["Volume"].astype(int).tolist()

    for i in range(len(stock2['Close'])):
        newRow = Stock_Info(stock_id = 2, stock_close = close2[i], stock_volume = volume2[i], stock_open=open2[i], stock_high = high2[i], stock_low=low2[i], sentiment_data=0, time_stamp=stock2["Date"][i])
        session.add(newRow)

    stock3 = bulk_stock_data("GM")
    close3 = stock3["Close"].astype(float).tolist()
    open3 = stock3["Open"].astype(float).tolist()
    low3 = stock3["Low"].astype(float).tolist()
    high3 = stock3["High"].astype(float).tolist()
    volume3 = stock3["Volume"].astype(int).tolist()

    for i in range(len(stock3['Close'])):
        newRow = Stock_Info(stock_id = 3, stock_close = close3[i], stock_volume = volume3[i], stock_open=open3[i], stock_high = high3[i], stock_low=low3[i], sentiment_data=0, time_stamp=stock3["Date"][i])
        session.add(newRow)

    stock4 = bulk_stock_data("TM")
    close4 = stock4["Close"].astype(float).tolist()
    open4 = stock4["Open"].astype(float).tolist()
    low4 = stock4["Low"].astype(float).tolist()
    high4 = stock4["High"].astype(float).tolist()
    volume4 = stock4["Volume"].astype(int).tolist()

    for i in range(len(stock4['Close'])):
        newRow = Stock_Info(stock_id = 4, stock_close = close4[i], stock_volume = volume4[i], stock_open=open4[i], stock_high = high4[i], stock_low=low4[i], sentiment_data=0, time_stamp=stock4["Date"][i])
        session.add(newRow)

    stock5 = bulk_stock_data("RIVN")
    close5 = stock5["Close"].astype(float).tolist()
    open5 = stock5["Open"].astype(float).tolist()
    low5 = stock5["Low"].astype(float).tolist()
    high5 = stock5["High"].astype(float).tolist()
    volume5 = stock5["Volume"].astype(int).tolist()

    for i in range(len(stock5['Close'])):
        newRow = Stock_Info(stock_id = 5, stock_close = close5[i], stock_volume = volume5[i], stock_open=open5[i], stock_high = high5[i], stock_low=low5[i], sentiment_data=0, time_stamp=stock5["Date"][i])
        session.add(newRow)
    
    session.commit()