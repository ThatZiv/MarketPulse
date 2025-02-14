# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import os
import copy
import time
import datetime
import requests
import torch
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, update, exc
from database.tables import Stock_Info
from models.sentiment_hf import sentiment_model


load_dotenv()

PUBLIC_KEY = os.environ.get("reddit_public_key")
SECRET_KEY = os.environ.get("reddit_secret_key")
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                "AppleWebKit/537.36 (KHTML, like Gecko)" 
                "Chrome/132.0.0.0 Safari/537.36")

AUTH = requests.auth.HTTPBasicAuth(PUBLIC_KEY, SECRET_KEY)

URL = "https://www.reddit.com/api/v1/access_token"

DATA = {
    "grant_type": "client_credentials"
}

HEADERS = {
    "User-Agent": USER_AGENT
}

OUTPUT = requests.post(URL, auth=AUTH, data=DATA, headers=HEADERS , timeout = 5)

if OUTPUT.status_code == 200:
    AUTH_TOKEN = OUTPUT.json()["access_token"]
    print("Success")
else:
    AUTH_TOKEN = 0
#sub_reddits = ["stocks","investing","wallstreetbets","Wallstreetbetsnew","StockMarket"]
#stocks = ["Tesla", "Ford Motor Company", "General Motors", "Toyota Motors", "Rivian"]


# Collects the reddit api for a subreddit and topic at
# this depth all the above subreddits have the same results
def reddit_request(subreddit, topic):
    headers = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "User-Agent": USER_AGENT
    }
    url = f"https://oauth.reddit.com/r/{subreddit}/search"

    params = {
    "t": "all",
    "limit": 100,
    "q": topic,
    "sort": "relevance",
    "type": "link"
    }

    output = requests.get(url, headers=headers, params=params, timeout = 5)
    inputs = []
    if output.status_code == 200:
        data = output.json()["data"]["children"]
        after = output.json()["data"]["after"]
        count = 0

        for listing in data:
            count+=1
            inputs.append({'title': listing["data"]["title"],
                        'self_text': listing["data"]["selftext"],
                        'timestamp': listing["data"]["created_utc"],
                        'name': listing["data"]["name"],
                        'url': listing['data']['subreddit']} )

        # gather the first 5000 results on this subreddit,
        # there are usually less than 400 results so this will always end with the break
        for i  in range (49):
            print(i)
            params = {
                "t": "all",
                "limit": 100,
                "q": topic,
                "after": after
            }
            output = requests.get(url, headers=headers, params=params, timeout = 5)
            if count == 100 and output.status_code == 200:
                data = output.json()["data"]["children"]
                after = output.json()["data"]["after"]
                for listing in data:
                    count+=1
                    inputs.append({'title': listing["data"]["title"],
                                'self_text': listing["data"]["selftext"],
                                'timestamp': listing["data"]["created_utc"],
                                'name': listing["data"]["name"],
                                'url': listing['data']['subreddit']})
        # search for the comments to the posts this will grab the first 25 comments
        # needs to be a deep copy to prevent infinite loop
        inputs_temp = copy.deepcopy(inputs)
        for lists in inputs_temp:
            # slow down to limit the amount of requests per minute
            #reddit will allow 100 requests per minute so this could be lowered
            time.sleep(1)
            try:
                url = f"https://oauth.reddit.com/r/{lists["url"]}/comments/{lists["name"][3:]}"
                headers = {
                "Authorization": f"Bearer {AUTH_TOKEN}",
                "User-Agent": USER_AGENT
                }
                params = {
                "limit": 100,
                "sort": "top",
                }
                output = requests.get(url, headers=headers, params=params , timeout = 5)
                data = output.json()[1]["data"]["children"]
                #print(data)
                for listing in data:
                    try:
                        #not adding name or url because I do not want to recure over the comments
                        inputs.append({'title': "", 'self_text': listing["data"]["body"],
                                    'timestamp': listing["data"]["created_utc"],
                                    'name': "", 'url': ""})
                    except KeyError as e:
                        print(e)
                        # This will error happen every pass because the last
                        # element is a list of comments and does not have a body
                        continue
                # Number of elements currently in the output array
            except requests.exceptions.Timeout as e:
                print(e)
            except requests.exceptions.HTTPError as e:
                print(e)
                time.sleep(60)
                continue
        return inputs
    return 0
#find dates from the database for the chosen stock and add the data to those table rows
def add_to_database(data, engine, stock_id):
    try:
        session = sessionmaker(bind=engine)
        session = session()
        stock_time = select(Stock_Info.time_stamp).where(Stock_Info.stock_id == stock_id)
        output_time = session.connection().execute(stock_time).all()
        for i in output_time:
            tensors = 0
            tensor = torch.tensor([[0,0,0]])
            for j in data:
                if datetime.date.fromtimestamp(j['timestamp']) == i.time_stamp:
                    tensors+=1
                    # If message is to long we will only take
                    # the first 512 characters of the message
                    if len(j['title']+" "+ j['self_text']) > 512:
                        logit = sentiment_model((j['title']+" "+j['self_text'])[:512])
                        tensor = torch.add(tensor, logit)
                    else:
                        logit = sentiment_model(j['title']+" "+j['self_text'])
                        tensor = torch.add(tensor, logit)
            if tensors > 0:
                # average tesor for the day
                answer = torch.div(tensor, tensors)
                update_row=update(Stock_Info)
                update_row=update_row.where(Stock_Info.stock_id==stock_id)
                update_row=update_row.where(Stock_Info.time_stamp==i.time_stamp)
                update_row=update_row.values(sentiment_data=(answer[0][0]*-1+answer[0][2]).item())
                print(session.connection().execute(update_row))
                # update the database for the day

        session.commit()
        session.flush()
        session.close()
    except exc.SQLAlchemyError as e:
        print(e)

# Finds the daily reddit posts for a stock and adds them to the database
def daily_reddit_request(subreddit, dates):
    headers = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "User-Agent": USER_AGENT
    }
    params = {
    "t": "week",
    "limit": 50,
    "q": dates[0]["search"],
    "sort": "relevance",
    "type": "link"
    }

    output = requests.get(f"https://oauth.reddit.com/r/{subreddit}/search",
                        headers=headers, params=params, timeout = 5)
    inputs = []
    if output.status_code == 200:
        data = output.json()["data"]["children"]
        for j in data:
            inputs.append({'title': j["data"]["title"],
            'self_text': j["data"]["selftext"],
            'timestamp': j["data"]["created_utc"],
            'name': j["data"]["name"],
            'url': j['data']['subreddit']} )

        inputs_temp = copy.deepcopy(inputs)
        for j in inputs_temp:
            # slow down to limit the amount of requests per minute reddit
            # will allow 100 requests per minute so this could be lowered
            time.sleep(1)
            try:
                headers = {
                "Authorization": f"Bearer {AUTH_TOKEN}",
                "User-Agent": USER_AGENT
                }
                params = {
                "limit": 25,
                "sort": "top",
                }
                output=requests.get((f"https://oauth.reddit.com/r/{j["url"]}"
                                    f"/comments/{j["name"][3:]}"),
                                    headers=headers, params=params, timeout = 5)
                data = output.json()[1]["data"]["children"]
                #print(data)
                try:
                    for k in data:
                        #not adding name or url because I do not want to recure over the comments
                        inputs.append({'title': "",
                        'self_text': k["data"]["body"],
                        'timestamp': k["data"]["created_utc"],
                        'name': "", 'url': ""})
                except KeyError:
                    continue
            except requests.exceptions.Timeout as e:
                print(e)
            except requests.exceptions.HTTPError as e:
                print(e)
                time.sleep(60)
                continue
        output = []
        for k in dates:
            tensors = 0
            tensor = torch.tensor([[0,0,0]])
            for j in inputs:
                d=[]
                d.append(datetime.date.fromtimestamp(j['timestamp']))
                d.append(datetime.datetime.strptime(k["time_stamp"].strftime("%b %d %H:%M:%S %Y"),
                                                "%b %d %H:%M:%S %Y"))
                d[1] = d[1].date()
                if d[1] == d[0] and len(j['title']+" "+ j['self_text']) > 512:
                    tensors+=1
                    tensor = torch.add(tensor,
                    sentiment_model((j['title']+" "+j['self_text'])[:512]))
                elif d[1] == d[0]:
                    tensors+=1
                    tensor = torch.add(tensor,
                    sentiment_model(j['title']+" "+j['self_text']))
            if tensors > 0:
                # average tesor for the day
                answer = torch.div(tensor, tensors)
                # value added to the database
                output.append({"answer": (answer[0][0]*-1+answer[0][2]).item(),
                                    "time_stamp": k["time_stamp"]})
        return output
    return 0
