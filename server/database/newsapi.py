from bs4 import BeautifulSoup
import requests
import torch
from models.sentimentHF import sentiment_model

search = "TSLA"

# for todays news from google rss feed
def news_search(search, engine):
   
    url = requests.get(f"https://news.google.com/rss/search?q={search}")
    soup = BeautifulSoup(url.text, 'xml')
    data = soup.find_all('item')
    content = []
    
    # create an array of titles and links to connect to and gather data on
    
    for i in range (10):
        title = data[i].title.text
        content.append(sentiment_model(title))

    # 0 -> Negative; 1 -> Neutral; 2 -> Positive
    
    Session = sessionmaker(bind=engine)
    session = Session()
    # Add the data to the database
    
    return content
    
    
    