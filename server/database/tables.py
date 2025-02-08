from sqlalchemy import create_engine, Table, Column, MetaData,  ForeignKey, String, PrimaryKeyConstraint, Float, Date, inspect
from sqlalchemy.dialects.postgresql import JSONB, INTEGER
from sqlalchemy.orm import mapped_column, relationship
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()

def dump_datetime(value):
    """Deserialize datetime object into string form for JSON processing."""
    if value is None:
        return None
    return [value.strftime("%Y-%m-%d"), value.strftime("%H:%M:%S")]

class Account(Base):
    __tablename__ = "Account"

    user_id = Column("user_id", INTEGER, primary_key=True)
    first_name = Column("first_name", String)
    last_name = Column("last_name", String)

    def __init__(self, user_id, first_name, last_name):
        self.user_id = user_id
        self.first_name = first_name
        self.last_name = last_name


class Stocks(Base):
    __tablename__ = "Stocks"

    stock_id = Column("stock_id", INTEGER, primary_key=True)
    stock_ticker = Column("stock_ticker", String)
    stock_name = Column("stock_name", String)
    search = Column("search", String)

    def __init__(self, stock_id, stock_ticker, stock_name,search):
        self.stock_id = stock_id
        self.stock_ticker = stock_ticker
        self.stock_name = stock_name
        self.search = search



class User_Stocks(Base):
    __tablename__ = "User_Stocks"

    user_id = Column("user_id", ForeignKey("Account.user_id"))
    stock_id = Column("stock_id", ForeignKey("Stocks.stock_id"))
    amount_owned = Column("amount_owned", INTEGER)


    __table_args__ = (PrimaryKeyConstraint('user_id', 'stock_id'),)




    def __init__(self, user_id, stock_id, amount_owned):
        self.user_id = user_id
        self.first_name = first_name
        self.last_name = last_name

    


class Stock_Info(Base):
    __tablename__ = "Stock_Info"

    stock_id = Column("stock_id", ForeignKey("Stocks.stock_id"))
    stock_close = Column("stock_close", Float)
    stock_volume = Column("stock_volume", INTEGER)
    stock_open = Column("stock_open", Float)
    stock_high = Column("stock_high", Float)
    stock_low= Column("stock_low", Float)
    sentiment_data = Column("sentiment_data", Float)
    time_stamp = Column("time_stamp", Date)
    news_data = Column("news_data", Float)

    __table_args__ = (PrimaryKeyConstraint('stock_id', 'time_stamp'),)

    def __init__(self, stock_id, stock_close, stock_volume, stock_open, stock_high, stock_low, sentiment_data, time_stamp, news_data):
        self.stock_id = stock_id
        self.stock_close = stock_close
        self.stock_volume = stock_volume
        self.stock_open = stock_open
        self.stock_high = stock_high
        self.stock_low = stock_low
        self.sentiment_data = sentiment_data
        self.time_stamp = time_stamp
        self.news_data = news_data
    
