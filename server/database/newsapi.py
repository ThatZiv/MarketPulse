from bs4 import BeautifulSoup
import requests
import httpx


search = "TSLA"
url = requests.get(f"https://news.google.com/rss/search?q={search}")

soup = BeautifulSoup(url.content, 'xml')
data = soup.find_all('item')
articles = []
content = []
# create an array of titles and links to connect to and gather data on
for article in data:
    title = article.title.text
    link = article.link.text
    articles.append({'title': title, 'link': link})

#print(articles)
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"}
url = requests.get(articles[10]['link'],headers=headers, allow_redirects=True)

#url =  client.get(articles[10]['link'])
#print(url.headers)
soup = BeautifulSoup(url.text, 'html.parser')

#print(data)
data = soup.find_all('body')
print(data)
#print(soup.prettify())
