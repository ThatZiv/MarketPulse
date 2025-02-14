import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ADD_1 = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
ADD_2 = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(ADD_1)
model = AutoModelForSequenceClassification.from_pretrained(ADD_2)


# calls the model distilroberta-finetuned-financial-news-sentiment-analysis for one text input
def sentiment_model(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    # Pylint wanted next 2 lines removed
    # predicted_class_id = logits.argmax().item()
    # model.config.id2label[predicted_class_id]
    # To train a model on `num_labels` classes, you
    # can pass `num_labels=num_labels` to `.from_pretrained(...)`
    # num_labels = len(model.config.id2label)
    labels = torch.tensor([1])
    loss = model(**inputs, labels=labels).loss
    round(loss.item(), 2)

    # 0 -> Negative; 1 -> Neutral; 2 -> Positive
    return logits
