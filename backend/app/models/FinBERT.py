from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


class FinBERTModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    def predict_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_score = probabilities[:, 2].item() - probabilities[:, 0].item()  # Positive - Negative
        return sentiment_score

    def analyze_news(self, news_data):
        sentiments = []
        for news_item in news_data:
            sentiment = self.predict_sentiment(news_item['title'] + ' ' + news_item['description'])
            sentiments.append(sentiment)
        return np.mean(sentiments)
