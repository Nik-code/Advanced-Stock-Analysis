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

    def analyze_sentiment(self, text):
        logger.info(f"Analyzing sentiment for text: {text[:100]}...")
        messages = [
            {"role": "system", "content": "You are a financial analyst. Analyze the sentiment of the following text and respond with the sentiment (positive, negative, or neutral) followed by a brief explanation. Format your response as 'Sentiment: [sentiment]\nExplanation: [explanation]'"},
            {"role": "user", "content": text},
        ]
        result = self.process_text(messages)
        logger.info(f"Sentiment analysis result: {result}")
        return result

    def generate_summary(self, news_data):
        logger.info("Generating summary for all news articles")
        combined_text = "\n".join([f"Title: {article['title']}\nSummary: {article['summary']}" for article in news_data])
        messages = [
            {"role": "system", "content": "You are a financial analyst. Summarize the following news articles in terms of their potential impact on the stock market. Provide a concise summary that reflects the overall sentiment and key points."},
            {"role": "user", "content": combined_text},
        ]
        summary = self.process_text(messages)
        logger.info(f"Generated summary: {summary}")
        return summary