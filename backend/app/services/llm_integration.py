from openai import OpenAI
import logging
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logger = logging.getLogger(__name__)

class GPT4Processor:
    def __init__(self):
        logger.info("Initializing GPT4Processor")

    def process_text(self, messages):
        logger.info(f"Processing text with GPT-4: {messages}")
        try:
            response = client.chat.completions.create(model="gpt-4o-mini",
            messages=messages,
            max_tokens=256,
            n=1,
            stop=None,
            temperature=0.7)
            generated_text = response.choices[0].message.content.strip()
            logger.info(f"GPT-4 generated text: {generated_text}")
            return generated_text
        except Exception as e:
            logger.error(f"Error processing text with GPT-4: {str(e)}")
            raise

    def analyze_sentiment(self, news_data):
        logger.info("Analyzing sentiment for news data")
        combined_text = "\n".join([f"Title: {article['title']}\nSummary: {article['summary']}" for article in news_data])
        messages = [
            {"role": "system", "content": "You are a financial analyst. Analyze the sentiment of the following news articles and respond with a single overall sentiment score between -1 (very negative) and 1 (very positive)."},
            {"role": "user", "content": combined_text},
        ]
        response = self.process_text(messages)
        try:
            sentiment_score = float(response.strip())
            logger.info(f"Sentiment analysis result: {sentiment_score}")
            return sentiment_score
        except ValueError:
            logger.error(f"Failed to parse sentiment score: {response}")
            return 0.0  # Return neutral sentiment if parsing fails

    def explain_sentiment(self, news_data, sentiment_score):
        logger.info("Generating explanation for sentiment")
        combined_text = "\n".join([f"Title: {article['title']}\nSummary: {article['summary']}" for article in news_data])
        messages = [
            {"role": "system", "content": "You are a financial analyst. Provide a brief 1-2 line explanation for the given sentiment score based on the news articles."},
            {"role": "user", "content": f"News:\n{combined_text}\n\nSentiment Score: {sentiment_score}"},
        ]
        explanation = self.process_text(messages)
        logger.info(f"Sentiment explanation: {explanation}")
        return explanation

    def final_analysis(self, news_sentiment, sentiment_explanation, lstm_prediction):
        logger.info("Generating final analysis")
        messages = [
            {"role": "system", "content": "You are a financial analyst. Based on the given information, provide a short description of the stock's immediate future and recommend whether a shareholder should buy more or sell. Include key factors influencing your decision."},
            {"role": "user", "content": f"News Sentiment: {news_sentiment}\nSentiment Explanation: {sentiment_explanation}\nLSTM Model Prediction: {lstm_prediction}"},
        ]
        analysis = self.process_text(messages)
        logger.info(f"Final analysis: {analysis}")
        return analysis