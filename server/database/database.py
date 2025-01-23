# Possibly using users table as a session 

import sqlalchemy as db
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///example.db')
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer,primary_key=True)
    email = db.Column(db.String(100), unique=True)
    date_joined = db.Column(db.Date, default=datetime.utcnow)
    password = db.Columm(db.String(100))
    isGoogle = db.Column(db.Boolean, default = False)

Base.metadata.create_all(engine)
