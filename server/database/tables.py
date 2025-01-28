from sqlalchemy import create_engine, Table, Column, MetaData,  ForeignKey, String, PrimaryKeyConstraint
from sqlalchemy.dialects.postgresql import JSONB, INTEGER
from sqlalchemy.orm import mapped_column, relationship
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()

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

    def __init__(self, stock_id, stock_ticker):
        self.stock_id = stock_id
        self.stock_ticker = stock_ticker



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

    stock_id = Column("stock_id", ForeignKey("Stocks.stock_id"), primary_key=True)
    price_data = Column("price_data", JSONB)
    sentiment_data = Column("sentiment_data", JSONB)
    time_stamp = Column("time_stamp", INTEGER)

    def __init__(self, stock_id, price_data, sentiment_data, time_stamp):
        self.stock_id = stock_id
        self.price_data = price_data
        self.sentiment_data = sentiment_data
        self.time_stamp = time_stamp


