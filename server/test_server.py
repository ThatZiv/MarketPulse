#Created as a test server to allow for testing api's without adding them to the main server
from models.lstm_attention import attention_lstm
from dotenv import load_dotenv
import os

from sqlalchemy import create_engine, select, func

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
attention_lstm('TSLA', engine)



