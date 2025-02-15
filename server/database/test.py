# pylint: disable=all
from sqlalchemy import create_engine,  Column, Integer, String
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
import pytest
import psycopg2



#Base = declarative_base()

#class User(Base):
#    __tablename__ = 'users'
#
#    id = Column(Integer, primary_key = True)
#    firstname = Column(String)
#    lastname = Column(String)


url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

