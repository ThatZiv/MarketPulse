import requests
from dotenv import load_dotenv
import os

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
    print("Success")

sub_reddits = ["stocks","investing","wallstreetbets","Wallstreetbetsnew","StockMarket"]
stocks = ["Tesla", "Ford Motor Company", "General Motors", "Toyota Motors", "Rivian"]

def reddit_request(subreddit, topic):
    
    headers = {
    "Authorization": f"Bearer {auth_token}",
    "User-Agent": user_agent
    }
    url = f"https://oauth.reddit.com/r/{subreddit}/search"

    params = {
    "t": "all",
    "limit": 100,
    "q": topic
    }

    output = requests.get(url, headers=headers, params=params)
    inputs = []
    if output.status_code == 200:
        data = output.json()["data"]["children"]
        after = output.json()["data"]["after"]
        print(after)
        count = 0
        for listing in data:
            count+=1
            inputs.append({'title': listing["data"]["title"], 'self_text': listing["data"]["selftext"], 'timestamp': listing["data"]["created_utc"], 'name': listing["data"]["name"]})
            
       
        
        #gather the first 5000 results on this subreddit
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
                        inputs.append({'title': listing["data"]["title"], 'self_text': listing["data"]["selftext"], 'timestamp': listing["data"]["created_utc"], 'name': listing["data"]["name"]})
            else:
                break
        # search for the comments to the posts
        for listing in inputs:
            url = f"https://oauth.reddit.com/api/morechildren"
            params = {
                "link_id": listing["name"],
                "limit_children": False
            }
            output = requests.get(url, headers=headers, params=params)
            print(output)



        return inputs 
        

    
def morechildren():
    url = f"https://oauth.reddit.com/r/CoveredCalls/comments/"
    headers = {
    "Authorization": f"Bearer {auth_token}",
    "User-Agent": user_agent
    }
    params = {
        "article": 't3_1hgvjcq',
        "depth": "10",
        }
    output = requests.get(url, headers=headers, params=params)
    data = output.json()["data"]["children"]
    count = 0
    for listing in data:
        count+=1
        print('\n')
        print(listing["data"]["link_title"])
        print(listing["data"]["body"])
        print(listing["data"]["created"])
    print(count)
#output = reddit_request("investing", "Ford Stock")
morechildren()

#print(len(output))



