from sqlalchemy import create_engine, Table, Column, Integer, MetaData, JSON
from sqlalchemy.dialects.postgresql import JSONB


def table(engine): 

    metadata=MetaData()
    metadata.reflect(bind=engine)
    users_data = Table('Account', metadata, 
        Column('user_id', Integer, primary_key=True),
        Column('user_name', String), 
        extend_existing = True
        )
    
    user_stocks = Table('User_Stocks', metadata,
        Column('user_id', Integer, primary_key=True),
        Column('stock_id', Integer, primary_key=True),
        Column('amount_owned', Integer),
        extend_existing = True
        )

    stocks = Table('Stocks', metadata,
        Column('stock_id', Integer, primary_key=True),
        extend_existing = True
        )

    stock_info = Table('Stocks', metadata,
        Column('stock_id', Integer, primary_key=True),
        Column('price_data', JSONB),
        Column('sentiment_data', JSONB),
        extend_existing = True
        )