from bs4 import BeautifulSoup
import requests

import torch



# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

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
    inputs = tokenizer(title, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    model.config.id2label[predicted_class_id]
    # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
    num_labels = len(model.config.id2label)
    labels = torch.tensor([1])
    loss = model(**inputs, labels=labels).loss
    round(loss.item(), 2)
    content.append(logits)


# 0 -> Negative; 1 -> Neutral; 2 -> Positive I Think
print(content)


print(logits)