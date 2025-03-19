# pylint: disable=all
from sqlalchemy import create_engine, Table, Column, MetaData,  ForeignKey, String, PrimaryKeyConstraint, Float, Date, inspect
from sqlalchemy.dialects.postgresql import JSONB, INTEGER
from sqlalchemy.orm import mapped_column, relationship, declarative_base


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
    stock = relationship("Stock_Info", back_populates="stocks")

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
    stocks = relationship("Stocks", back_populates="stock")  

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

class User_Stock_Purchases(Base):
    __tablename__ = "User_Stock_Purchases"

    user_id = Column("user_id", ForeignKey("User_Stocks.user_id"))
    date = Column("date", Date)
    price_purchased = Column("price_purchased", Float)
    stock_id = Column("stock_id", ForeignKey("User_Stocks.stock_id"))
    amount_purchased = Column("amount_purchased", Float)

    __table_args__ = (PrimaryKeyConstraint('stock_id', 'date', 'user_id'),)

    def __init__(self, user_id, date, price_purchased, stock_id, amount_purchased):
        self.user_id = user_id
        self.date = date
        self.price_purchased = price_purchased
        self.stock_id = stock_id
        self.amount_purchased = amount_purchased

class Stock_Predictions(Base):
    __tablename__ = "Stock_Predictions"

    stock_id = Column("stock_id", ForeignKey("Stocks.stock_id"))
    created_at = Column("created_at", Date)
    model_1 = Column("model_1", JSONB)
    model_2 = Column("model_2", JSONB)
    model_3 = Column("model_3", JSONB)
    model_4 = Column("model_4", JSONB)
    model_5 = Column("model_5", JSONB)

    __table_args__ = (PrimaryKeyConstraint('stock_id', 'created_at'),)

    def __init__(self, stock_id, created_at, model_1, model_2, model_3, model_4, model_5):
        self.stock_id = stock_id
        self.created_at = created_at
        self.model_1 = model_1
        self.model_2 = model_2
        self.model_3 = model_3
        self.model_4 = model_4
        self.model_5 = model_5
