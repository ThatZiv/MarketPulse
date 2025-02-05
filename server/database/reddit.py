import requests
from dotenv import load_dotenv

load_dotenv()

public_key = "6Fw2rOLKECBtYpOD3Sr0uA"
secret_key = "Jv4tM3v_MFUi-iIZmRDoIbfmkTaYyQ"
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



def redditrequest(subreddit, topic):
    headers = {
    "Authorization": f"Bearer {auth_token}",
    "User-Agent": user_agent
    }
    url = f"https://oauth.reddit.com/r/{subreddit}/search"

    params = {
    "t": "week",
    "limit": 10,
    "q": topic
    }

    output = requests.get(url, headers=headers, params=params)

    if output.status_code == 200:
        data = output.json()["data"]["children"]

        for listing in data:
            print(listing["data"]["title"])
            print(listing["data"]["selftext"])
            print("\n")

redditrequest("stocks", "TSLA")



