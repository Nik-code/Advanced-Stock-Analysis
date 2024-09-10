from openai import OpenAI
import logging
import os
from dotenv import load_dotenv
import re
from app.services.technical_indicators import calculate_rsi, calculate_sma, calculate_ema, calculate_macd, calculate_bollinger_bands, calculate_atr

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logger = logging.getLogger(__name__)


class GPT4Processor:
    def __init__(self):
        logger.info("Initializing GPT4Processor")

    def process_text(self, messages):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=700,
                n=1,
                stop=None,
                temperature=0.7
            )
            generated_text = response.choices[0].message.content.strip()
            return generated_text
        except Exception as e:
            logger.error(f"Error processing text with GPT-4: {str(e)}")
            raise

    def analyze_news_sentiment(self, news_data):
        logger.info("Analyzing sentiment and generating explanation for news data")
        combined_text = "\n".join([f"Title: {article['title']}\nSummary: {article['summary']}" for article in news_data])
        messages = [
            {"role": "system", "content": "You are a financial analyst. Analyze the sentiment of the following news articles and provide a comprehensive analysis. Your very first line should just be the sentiment score, a value between -1 (very negative) and 1 (very positive). Then starting from a new line, provide a brief explanation for the sentiment score, followed by a bullet-point summary of key insights from the news."},
            {"role": "user", "content": combined_text},
        ]
        response = self.process_text(messages)
        
        # Parse the response to extract sentiment score and explanation
        lines = response.split('\n')
        sentiment_score = float(re.search(r"-?\d+\.\d+", lines[0].strip()).group())
        explanation = '\n'.join(lines[1:])
        
        logger.info(f"Sentiment analysis result: {sentiment_score}")
        logger.info(f"Sentiment explanation: {explanation}")
        
        return sentiment_score, explanation

    async def run_arima_strategy(self, stock_code, time_frame='1year'):
        from backend.scripts.arima_strategy import run_strategy_test
        arima_results = await run_strategy_test(stock_code, time_frame=time_frame)
        return arima_results

    async def final_analysis(self, stock_code, news_sentiment, sentiment_explanation, lstm_prediction, technical_indicators, historical_data):
        logger.info("Generating final analysis")
        
        # Run ARIMA strategy
        arima_results = await self.run_arima_strategy(stock_code)
        
        messages = [
            {"role": "system", "content": "You are a financial analyst for the Indian Stock Market. Based on the given information, provide a detailed analysis of the stock's performance and future outlook. Include key factors influencing your decision and recommend whether a shareholder should buy, hold, or sell. Structure your response with clear sections and bullet points for easy readability."},
            {"role": "user", "content": f"""
            Stock Code: {stock_code}
            News Sentiment Score: {news_sentiment}
            News Sentiment Analysis: {sentiment_explanation}
            LSTM Model Prediction: {lstm_prediction}
            Historical Data: {historical_data}
            Technical Indicators:
            - SMA 20: {technical_indicators['sma_20']}
            - EMA 50: {technical_indicators['ema_50']}
            - RSI: {technical_indicators['rsi']}
            - MACD: {technical_indicators['macd']}
            - ATR: {technical_indicators['atr']}
            ARIMA Strategy Results:
            - Total Return: {arima_results['total_return']}
            - Annualized Return: {arima_results['annualized_return']}
            - Sharpe Ratio: {arima_results['sharpe_ratio']}
            - Max Drawdown: {arima_results['max_drawdown']}
            - Win Rate: {arima_results['win_rate']}
            """},
        ]
        analysis = self.process_text(messages)
        logger.info(f"Final analysis: {analysis}")
        return analysis

    async def process_news_data(self, news_data, lstm_prediction, symbol, historical_data):
        sentiment_score, explanation = self.analyze_news_sentiment(news_data)
        technical_indicators = await self.fetch_technical_indicators(symbol)
        final_analysis = await self.final_analysis(
            symbol,
            sentiment_score,
            explanation,
            lstm_prediction,
            technical_indicators,
            historical_data
        )
        return {
            'sentiment': sentiment_score,
            'explanation': explanation,
            'analysis': final_analysis
        }

    async def fetch_technical_indicators(self, symbol):
        from app.services.data_collection import fetch_historical_data
        historical_data = await fetch_historical_data(symbol, '1year')
        if historical_data is None:
            logger.error(f"No historical data found for {symbol}")
            return None

        close_prices = historical_data['close']
        high_prices = historical_data['high']
        low_prices = historical_data['low']

        indicators = {
            'sma_20': calculate_sma(close_prices, 20).iloc[-1],
            'ema_50': calculate_ema(close_prices, 50).iloc[-1],
            'rsi': calculate_rsi(close_prices).iloc[-1],
            'macd': calculate_macd(close_prices)[0].iloc[-1],  # MACD line
            'atr': calculate_atr(high_prices, low_prices, close_prices).iloc[-1]
        }

        return indicators
