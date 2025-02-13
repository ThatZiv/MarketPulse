from sqlalchemy import create_engine, select, update
from database.tables import Stocks, Stock_Info, Base
from datetime import date
import time
from models.sentimentHF import sentiment_model
from duckduckgo_search import DDGS
import torch

def stock_scrap(stock_id, engine):

    Session = sessionmaker(bind=engine)
    session = Session()
    # Search for the stock by using the stock id
    stock_q = select(Stocks).filter(Stocks.stock_id == stock_id)
    stock = session.connection().execute(stock_q).first()
    # Get all the dates out of the table to get news data for
    stock_data_a = select(Stock_Info.time_stamp, Stock_Info.news_data).filter(Stock_Info.stock_id == stock_id)
    stock_data = session.connection().execute(stock_data_a).all()
    
    for day in stock_data:
        # This is to skip days that have already been calculated allowing the program to not have to run all at once
        if day.news_data == None:
            # Going to fast for API this is slow enough to get 500+ values without an API block
            time.sleep(5)          
            try:
                results = DDGS().text(f"{stock.search} news", max_results=5, timelimit=f"{day.time_stamp}..{day.time_stamp}")
                print(day.time_stamp)
                print(results) 

                              
                
                tensors = 0
                tensor = torch.tensor([[0,0,0]])
                
                for r in results:
                    try:
                        # Model goes here
                        article = r['title']+ " " + r['body']
                        if len(article) > 512:
                            logit = sentiment_model(article[:512])
                            tensor = torch.add(tensor, logit)
                            articles.append(article)
                        else:
                            logit = sentiment_model(article)
                            tensor = torch.add(tensor, logit)
                        tensors+=1
                    except Exception as e:
                        print(e)
                        continue
                
                if tensors > 0:
                    # average tesor for the day
                    answer = torch.div(tensor, tensors)
                    update_row = update(Stock_Info).where(Stock_Info.stock_id == stock_id).where(Stock_Info.time_stamp == day.time_stamp).values(news_data = (answer[0][0]*-1+answer[0][2]).item())
                    print(session.connection().execute(update_row))
                
                session.commit()
                session.flush() 

                
            except Exception as e:
                print(e)
                continue
        

    session.close()
    
def add_news(dates):
    answers = []
    for day in dates:
        
        time.sleep(5)  
        try:
            results = DDGS().text(f"{day["search"]} news", max_results=5, timelimit=f"{day["time_stamp"].strftime("%Y-%m-%d")}..{day["time_stamp"].strftime("%Y-%m-%d")}")                    
            tensors = 0
            tensor = torch.tensor([[0,0,0]])
                
            for r in results:
                try:
                    # Model goes here
                    article = r['title']+ " " + r['body']
                    if len(article) > 512:
                        logit = sentiment_model(article[:512])
                        tensor = torch.add(tensor, logit)
                        articles.append(article)
                    else:
                        logit = sentiment_model(article)
                        tensor = torch.add(tensor, logit)
                    tensors+=1
                except Exception as e:
                    print(e)
                    continue
                
            if tensors > 0:
                # average tesor for the day
                answer = torch.div(tensor, tensors)
                answers.append({"news": (answer[0][0]*-1+answer[0][2]).item(), "time_stamp": day["time_stamp"]})

                
        except Exception as e:
            print(e)
            continue
    
    return answers            

   

