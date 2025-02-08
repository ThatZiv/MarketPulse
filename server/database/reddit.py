import requests
from dotenv import load_dotenv
import os
import copy
import time
from database.tables import Stock_Info
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, select, update
import datetime
from models.sentimentHF import sentiment_model
import torch

load_dotenv()

public_key = os.environ.get("reddit_public_key")
secret_key = os.environ.get("reddit_secret_key")
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"

auth = requests.auth.HTTPBasicAuth(public_key, secret_key)

url = "https://www.reddit.com/api/v1/access_token"

data = {
    "grant_type": "client_credentials"
}

headers = {
    "User-Agent": user_agent
}

output = requests.post(url, auth=auth, data=data, headers=headers)

if output.status_code == 200:
    auth_token = output.json()["access_token"]
    #print("Success")

#sub_reddits = ["stocks","investing","wallstreetbets","Wallstreetbetsnew","StockMarket"]
#stocks = ["Tesla", "Ford Motor Company", "General Motors", "Toyota Motors", "Rivian"]


#Collects the reddit api for a subreddit and topic at this depth all the above subreddits have the same results
def reddit_request(subreddit, topic):
    
    headers = {
    "Authorization": f"Bearer {auth_token}",
    "User-Agent": user_agent
    }
    url = f"https://oauth.reddit.com/r/{subreddit}/search"

    params = {
    "t": "all",
    "limit": 100,
    "q": topic,
    "sort": "relevance",
    "type": "link"
    }

    output = requests.get(url, headers=headers, params=params)
    inputs = []
    if output.status_code == 200:
        data = output.json()["data"]["children"]
        after = output.json()["data"]["after"]
        count = 0
        

        for listing in data:
            count+=1
            inputs.append({'title': listing["data"]["title"], 'self_text': listing["data"]["selftext"], 'timestamp': listing["data"]["created_utc"], 'name': listing["data"]["name"], 'url': listing['data']['subreddit']} )
            
       
        
        #gather the first 5000 results on this subreddit, there are usually less than 400 results so this will always end with the break
        for i in range (49):
            params = {
                "t": "all",
                "limit": 100,
                "q": topic,
                "after": after
            }
            output = requests.get(url, headers=headers, params=params)
            if count == 100:
                if output.status_code == 200:
                    data = output.json()["data"]["children"]
                    after = output.json()["data"]["after"]
                    print(after)
                    
                    count = 0
                    for listing in data:
                        count+=1
                        inputs.append({'title': listing["data"]["title"], 'self_text': listing["data"]["selftext"], 'timestamp': listing["data"]["created_utc"], 'name': listing["data"]["name"], 'url': listing['data']['subreddit']})
            else:
                break
        count = len(inputs)
        # search for the comments to the posts this will grab the first 25 comments
        # needs to be a deep copy to prevent infinite loop
        inputs_temp = copy.deepcopy(inputs)
        for listings in inputs_temp:
            # slow down to limit the amount of requests per minute reddit will allow 100 requests per minute so this could be lowered
            time.sleep(1)
            try:
                url = f"https://oauth.reddit.com/r/{listings["url"]}/comments/{listings["name"][3:]}"
                headers = {
                "Authorization": f"Bearer {auth_token}",
                "User-Agent": user_agent
                }
                params = {
                "limit": 100,
                "sort": "top",
                
                }
                output = requests.get(url, headers=headers, params=params)
        
                data = output.json()[1]["data"]["children"]
                #print(data)
                
                for listing in data:
                    try:
                        count+=1
                        #not adding name or url because I do not want to recure over the comments
                        inputs.append({'title': "", 'self_text': listing["data"]["body"], 'timestamp': listing["data"]["created_utc"], 'name': "", 'url': ""})
                    except :
                        # This will error happen every pass because the last element is a list of comments and does not have a body
                        continue
                # Number of elements currently in the output array
                print(count)
                
            except Exception as e:
                print(e)
                # pause to allow the request count to reset incase we went to fast
                time.sleep(60)
                continue
        
        return inputs
        


#find dates from the database for the chosen stock and add the data to those table rows
def add_to_database(data, engine, id):
    try:
        Session = sessionmaker(bind=engine)
        session = Session()
        stock_time = select(Stock_Info.time_stamp).where(Stock_Info.stock_id == id)
        output_time = session.connection().execute(stock_time).all()
        count = 0
        for i in output_time:
            tensors = 0
            tensor = torch.tensor([[0,0,0]])
            for j in data:
                
                date = datetime.date.fromtimestamp(j['timestamp'])
                if date == i.time_stamp:
                    count+=1
                    tensors+=1
                    # If message is to long we will only take the first 512 characters of the message
                    if len(j['title']+" "+ j['self_text']) > 512:
                        logit = sentiment_model((j['title']+" "+j['self_text'])[:512])
                        tensor = torch.add(tensor, logit)
                    else:
                        logit = sentiment_model(j['title']+" "+j['self_text'])
                        tensor = torch.add(tensor, logit)
            if tensors > 0:
                # average tesor for the day
                answer = torch.div(tensor, tensors)
                update_row = update(Stock_Info).where(Stock_Info.stock_id == id).where(Stock_Info.time_stamp == i.time_stamp).values(sentiment_data = (answer[0][0]*-1+answer[0][2]).item())
                print(session.connection().execute(update_row))
                # update the database for the day

        session.commit()
        session.flush() 
        session.close()


        print(count)
                    

    except Exception as e:
        print(e)
        

# Finds the daily reddit posts for a stock and adds them to the database
def daily_reddit_request(subreddit, topic, engine, id):
    headers = {
    "Authorization": f"Bearer {auth_token}",
    "User-Agent": user_agent
    }
    url = f"https://oauth.reddit.com/r/{subreddit}/search"

    params = {
    "t": "day",
    "limit": 20,
    "q": topic,
    "sort": "relevance",
    "type": "link"
    }

    output = requests.get(url, headers=headers, params=params)
    inputs = []
    if output.status_code == 200:
        data = output.json()["data"]["children"]
        after = output.json()["data"]["after"]
        count = 0
        

        for listing in data:
            count+=1
            inputs.append({'title': listing["data"]["title"], 'self_text': listing["data"]["selftext"], 'timestamp': listing["data"]["created_utc"], 'name': listing["data"]["name"], 'url': listing['data']['subreddit']} )
        

        inputs_temp = copy.deepcopy(inputs)
        for listings in inputs_temp:
            # slow down to limit the amount of requests per minute reddit will allow 100 requests per minute so this could be lowered
            time.sleep(1)
            try:
                url = f"https://oauth.reddit.com/r/{listings["url"]}/comments/{listings["name"][3:]}"
                headers = {
                "Authorization": f"Bearer {auth_token}",
                "User-Agent": user_agent
                }
                params = {
                "limit": 20,
                "sort": "top",
                
                }
                output = requests.get(url, headers=headers, params=params)
        
                data = output.json()[1]["data"]["children"]
                #print(data)
                
                for listing in data:
                    try:
                        count+=1
                        #not adding name or url because I do not want to recure over the comments
                        inputs.append({'title': "", 'self_text': listing["data"]["body"], 'timestamp': listing["data"]["created_utc"], 'name': "", 'url': ""})
                    except :
                        # This will error happen every pass because the last element is a list of comments and does not have a body
                        continue
                # Number of elements currently in the output array
                print(count)
                
            except Exception as e:
                print(e)
                # pause to allow the request count to reset incase we went to fast
                time.sleep(60)
                continue
        
        try:
            Session = sessionmaker(bind=engine)
            count = 0
            tensors = 0
            tensor = torch.tensor([[0,0,0]])
            for j in inputs:
                today = datetime.date.today()    
                date = datetime.date.fromtimestamp(j['timestamp'])
                if date == today:
                    count+=1
                    tensors+=1
                    # If message is to long we will only take the first 512 characters of the message
                    if len(j['title']+" "+ j['self_text']) > 512:
                        logit = sentiment_model((j['title']+" "+j['self_text'])[:512])
                        tensor = torch.add(tensor, logit)
                    else:
                        logit = sentiment_model(j['title']+" "+j['self_text'])
                        tensor = torch.add(tensor, logit)
            if tensors > 0:
                # average tesor for the day
                answer = torch.div(tensor, tensors)
                # value added to the database
                return (answer[0][0]*-1+answer[0][2]).item()
            else : 
                return 0

                # update the database for the day                    

        except Exception as e:
            print(e)
            return 0
            
            
            
    
#output = reddit_request("investing", "Tesla")
#morechildren()

#print(len(output))



