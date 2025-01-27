from sqlalchemy import create_engine, Table, Column, MetaData, JSON, ForeignKey, Mapped
from sqlalchemy.dialects.postgresql import JSONB, INTEGER
from sqlalchemy.orm import mapped_column, relationship, DeclarativeBase


class Base(DeclarativeBase):
    pass

class Account(Base):
    __tablename__ = "Account"

    user_id = Column(INTEGER, primary_key=True)
    user_name = Column(String)

class User_Stocks(Base):
    __tablename__ = "User_Stocks"

    user_id = Column(ForeignKey("Account.user_id", primary_key=True))
    stock_id = Column(ForeignKey("Stocks.stock_id") primary_key=True)
    amount_owned = Column(INTEGER),

class Stocks(Base):
    __tablename__ = "Stocks"

    stock_id = Column(INTEGER, primary_key=True),
    stock_ticker = Column(String)

class Stock_Info(Base):
    __tablename__ = "Stock_Info"

    stock_id = Column(ForeignKey("Stocks.stock_id"), primary_key=True),
    price_data = Column(, JSONB),
    sentiment_data = Column(JSONB),
    time_stamp = Column(, INTEGER)
