# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import time
import torch
from sqlalchemy import select, update
from sqlalchemy.orm import sessionmaker
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import (
    ConversationLimitException,
    DuckDuckGoSearchException,
    RatelimitException,
    TimeoutException,
)

from models.sentiment_hf import sentiment_model
from database.tables import Stock_Info, Stocks


def stock_scrap(stock_id, engine):

    session = sessionmaker(bind=engine)
    session = session()
    # Search for the stock by using the stock id
    stock = select(Stocks).filter(Stocks.stock_id == stock_id)
    stock = session.connection().execute(stock).first()
    # Get all the dates out of the table to get news data for
    stock_data = select(Stock_Info.time_stamp,
                    Stock_Info.news_data).filter(Stock_Info.stock_id == stock_id)
    stock_data = session.connection().execute(stock_data).all()
    for day in stock_data:

        # allowing the program to not have to run all at once
        # Going to fast for API this is slow enough to get 500+ values without an API block
        time.sleep(5)
        try:
            results = DDGS().text(f"{stock.search} news",
                        max_results=5,
                        timelimit=f"{day.time_stamp}..{day.time_stamp}")
            print(day.time_stamp)
            print(results)
            articles = []
            tensors = 0
            tensor = torch.tensor([[0,0,0]])
            for r in results:
                # removed try block
                # Model goes here
                article = r['title']+ " " + r['body']
                if len(article) > 512:
                    tensor = torch.add(tensor, sentiment_model(article[:512]))
                    articles.append(article)
                else:
                    tensor = torch.add(tensor, sentiment_model(article))
                tensors+=1
            if tensors > 0:
                # average tesor for the day
                answer = torch.div(tensor, tensors)
                update_row = update(Stock_Info)
                update_row = update_row.where(Stock_Info.stock_id == stock_id)
                update_row = update_row.where(Stock_Info.time_stamp == day.time_stamp)
                update_row = update_row.values(news_data = (answer[0][0]*-1+answer[0][2]).item())
                print(session.connection().execute(update_row))
            session.commit()
            session.flush()

        except ConversationLimitException as e:
            print(e)
            continue
        except RatelimitException as e:
            print(e)
            continue
        except TimeoutException as e:
            print(e)
            continue
        except DuckDuckGoSearchException as e:
            print(e)
            continue


    session.close()

def add_news(dates):
    answers = []
    for day in dates:
        time.sleep(10)
        try:
            day["time_stamp"] = day["time_stamp"].strftime("%Y-%m-%d")
            results = DDGS().text(f"{day['search']} news", max_results=5,
                    timelimit=f"{day['time_stamp']}..{day['time_stamp']}")

            tensors = 0
            tensor = torch.tensor([[0,0,0]])
            for r in results:
                # removed try block
                # Model goes here
                articles = []
                article = r['title']+ " " + r['body']
                if len(article) > 512:
                    logit = sentiment_model(article[:512])
                    tensor = torch.add(tensor, logit)
                    articles.append(article)
                else:
                    logit = sentiment_model(article)
                    tensor = torch.add(tensor, logit)
                tensors+=1
            if tensors > 0:
                # average tesor for the day
                answer = torch.div(tensor, tensors)
                answers.append({"news": (answer[0][0]*-1+answer[0][2]).item(),
                 "time_stamp": day["time_stamp"]})
        except ConversationLimitException as e:
            print(e)
            continue
        except RatelimitException as e:
            time.sleep(10)
            print(e)
            continue
        except TimeoutException as e:
            print(e)
            continue
        except DuckDuckGoSearchException as e:
            print(e)
            continue
    return answers
