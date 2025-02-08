from bs4 import BeautifulSoup
import requests
import torch
from models.sentimentHF import sentiment_model
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker
from database.tables import Stock_Info
import datetime
import torch



# for todays news from google rss feed
def news_search(search, engine, id):
   
    url = requests.get(f"https://news.google.com/rss/search?q={search}")
    soup = BeautifulSoup(url.text, 'xml')
    data = soup.find_all('item')
    content = []
    
    # create an array of titles and links to connect to and gather data on
    tensor = torch.tensor([[0,0,0]])
    for i in range (10):
        title = data[i].title.text
        #content.append(sentiment_model(title))
        tensor = torch.add(tensor, sentiment_model(title))

    # 0 -> Negative; 1 -> Neutral; 2 -> Positive
    
    
    # Add the data to the database

        
    
    tensor = torch.div(tensor, 10)
    
    Session = sessionmaker(bind=engine)
    session = Session()

    date= datetime.date.today()
    update_row = update(Stock_Info).where(id == Stock_Info.stock_id).where(date==Stock_Info.time_stamp).values(news_data = (tensor[0][0]*-1+tensor[0][2]).item())
    print(session.connection().execute(update_row))

    session.commit()
    session.flush()
    session.close() 
          
    return content
    
    
    