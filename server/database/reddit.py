# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-nested-blocks

import os
import copy
import time
import datetime
import requests
import torch
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, update, exc
from models.sentiment_hf import sentiment_model
from database.tables import Stock_Info


def reddit_auth():
    load_dotenv()

    public_key = os.environ.get("reddit_public_key")
    secret_key = os.environ.get("reddit_secret_key")
    user_agent = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                    "AppleWebKit/537.36 (KHTML, like Gecko)" 
                    "Chrome/132.0.0.0 Safari/537.36")

    auth = requests.auth.HTTPBasicAuth(public_key, secret_key)

    url = "https://www.reddit.com/api/v1/access_token"

    data = {
        "grant_type": "client_credentials"
    }

    headers = {
        "User-Agent": user_agent
    }


    output = requests.post(url, auth=auth, data=data, headers=headers , timeout = 5)

    if output.status_code == 200:
        auth_token = output.json()["access_token"]
        print("Success")
    else:
        auth_token = 0
    return auth_token

#sub_reddits = ["stocks","investing","wallstreetbets","Wallstreetbetsnew","StockMarket"]
#stocks = ["Tesla", "Ford Motor Company", "General Motors", "Toyota Motors", "Rivian"]


# Collects the reddit api for a subreddit and topic at
# this depth all the above subreddits have the same results

def base_request(url, params, auth):
    user_agent = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                    "AppleWebKit/537.36 (KHTML, like Gecko)" 
                    "Chrome/132.0.0.0 Safari/537.36")

    request_headers = {
        "Authorization": f"Bearer {auth}",
        "User-Agent": user_agent
        }
    output = -1
    try:
        output = requests.get(url, headers=request_headers, params=params, timeout = 5)
    except requests.exceptions.HTTPError as e:
        print(e)
        time.sleep(30)
    except requests.exceptions.Timeout as e:
        print(e)
        time.sleep(30)
    except requests.ConnectionError as e:
        print(e)
        time.sleep(30)
    except requests.exceptions.RequestException as e:
        print(e)
    return output

def request_search(subreddit, topic, thread_limit, period, auth):
    url = f"https://oauth.reddit.com/r/{subreddit}/search"
    params = {
    "t": period,
    "limit": thread_limit,
    "q": topic,
    "sort": "relevance",
    "type": "link"
    }
    return base_request(url, params, auth)

def request_after(subreddit, topic, thread_limit, after, period, auth):
    url = f"https://oauth.reddit.com/r/{subreddit}/search"
    params = {
                "t": period,
                "limit": thread_limit,
                "q": topic,
                "after": after
            }
    return base_request(url, params, auth)

def request_comment(lists_url, lists_name, comment_limit, auth):
    url = f"https://oauth.reddit.com/r/{lists_url}/comments/{lists_name}"
    params = {
            "limit": comment_limit,
            "sort": "top",
            }
    output = base_request(url, params, auth)
    try:
        if output.status_code == 200:
            return output.json()[1]["data"]["children"]
    except KeyError:
        return -1
    except AttributeError:
        return -1
    return -1
# period can be hour, day, week, month, year, all
def reddit_request(subreddit, topic, thread_limit, comment_limit, period):
    auth = reddit_auth()
    if auth != 0:
        output = request_search(subreddit, topic, thread_limit, period, auth)
        inputs = []

        if output.status_code == 200:
            try:
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
            except KeyError as e:
                print(e)
            # gather the first 5000 results on this subreddit,
            # there are usually less than 400 results so this will always end with the break
            for i  in range (49):
                print(i)
                output = request_after(subreddit, topic, thread_limit, after, period, auth)
                if output.status_code == 200 and count == 100:
                    try:
                        data = output.json()["data"]["children"]
                        after = output.json()["data"]["after"]
                        count = 0
                        for listing in data:
                            count+=1
                            inputs.append({'title': listing["data"]["title"],
                                            'self_text': listing["data"]["selftext"],
                                            'timestamp': listing["data"]["created_utc"],
                                            'name': listing["data"]["name"],
                                            'url': listing['data']['subreddit']})
                    except KeyError as e:
                        print(e)
                        continue
                else:
                    break
            # search for the comments to the posts this will grab the first 25 comments
            # needs to be a deep copy to prevent infinite loop
            inputs_temp = copy.deepcopy(inputs)
            for lists in inputs_temp:
                # slow down to limit the amount of requests per minute
                #reddit will allow 100 requests per minute so this could be lowered
                time.sleep(1)
                data = request_comment(lists['url'], lists['name'][3:], comment_limit, auth)
                    #print(data)
                for listing in data:
                    try:
                        #not adding name or url because I do not want to recure over the comments
                        inputs.append({'title': "", 'self_text': listing["data"]["body"],
                                    'timestamp': listing["data"]["created_utc"],
                                    'name': "", 'url': ""})
                    except KeyError as e:
                        # This will error happen every pass because the last
                        # element is a list of comments and does not have a body
                        continue
                    # Number of elements currently in the output array
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
        session.close()

# Finds the daily reddit posts for a stock and adds them to the database
# dates[0]["search"]
def daily_reddit_request(subreddit, dates):
    inputs = reddit_request(subreddit, dates[0]["search"], 25, 50, "week")
    out = []
    for k in dates:
        tensors = 0
        tensor = torch.tensor([[0,0,0]])
        for j in inputs:
            try:
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
            except KeyError as e:
                print(e)
                continue
        if tensors > 0:
            # average tesor for the day
            answer = torch.div(tensor, tensors)
            # value added to the database
            out.append({"answer": (answer[0][0]*-1+answer[0][2]).item(),
                                    "time_stamp": k["time_stamp"]})
    print(out)
    return out
