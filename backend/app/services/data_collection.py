from .zerodha_service import ZerodhaService
import pandas as pd
import os
from datetime import datetime, timedelta
import traceback
import logging
from .technical_indicators import calculate_rsi, calculate_sma, calculate_ema, calculate_macd, calculate_bollinger_bands, calculate_atr
from ..db.influx_writer import write_stock_data_to_influxdb
from ..db.influx_client import get_influxdb_client
from .llm_integration import LLaMAProcessor
import aiohttp
import yfinance as yf
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

zerodha_service = ZerodhaService()
client = get_influxdb_client()

llm_processor = LLaMAProcessor()

async def fetch_historical_data(scrip_code: str, time_frame: str = '1year'):
    try:
        end_date = datetime.now()
        
        if time_frame == '1month':
            start_date = end_date - timedelta(days=30)
        elif time_frame == '3months':
            start_date = end_date - timedelta(days=90)
        elif time_frame == '1year':
            start_date = end_date - timedelta(days=365)
        elif time_frame == '5years':
            start_date = end_date - timedelta(days=1825)
        else:
            raise ValueError(f"Invalid time frame: {time_frame}")
        
        logger.info(f"Fetching instrument token for {scrip_code}")
        instrument_token = zerodha_service.get_instrument_token("BSE", scrip_code)
        
        if not instrument_token:
            logger.error(f"No instrument token found for {scrip_code}")
            return None
        
        logger.info(f"Fetching historical data for {scrip_code} from {start_date} to {end_date}")
        data = zerodha_service.get_historical_data(
            instrument_token,
            start_date,
            end_date,
            "day"
        )
        
        if not data:
            logger.warning(f"No data found for {scrip_code}. It may be delisted.")
            return None

        df = pd.DataFrame(data)
        logger.info(f"Columns in the dataframe: {df.columns}")
        logger.info(f"Successfully fetched {len(df)} data points for {scrip_code}")
        return df
    except Exception as e:
        logger.error(f"Error fetching historical data for {scrip_code}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

async def fetch_process_store_data(scrip_code: str, time_frame: str = '1year'):
    df = await fetch_historical_data(scrip_code, time_frame)
    if df is not None:
        # Calculate technical indicators
        df['RSI'] = calculate_rsi(df['close'])
        df['SMA_20'] = calculate_sma(df['close'], 20)
        df['EMA_50'] = calculate_ema(df['close'], 50)
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['close'])
        df['Bollinger_Upper'], df['Bollinger_Middle'], df['Bollinger_Lower'] = calculate_bollinger_bands(df['close'])
        df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])

        # Fetch and process news data
        news_data = await fetch_news_data(scrip_code)
        news_analysis = await process_news_data(news_data)
        
        # Add news analysis to the dataframe
        df['News_Sentiment'] = news_analysis['sentiment']
        df['News_Topics'] = ','.join(news_analysis['topics'])

        # Write to InfluxDB
        write_stock_data_to_influxdb(df, scrip_code)
        logger.info(f"Data for {scrip_code} processed and stored in InfluxDB.")
    else:
        logger.error(f"Skipping {scrip_code} due to missing data.")

async def fetch_news_data(scrip_code: str):
    try:
        # Run the yfinance operation in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        stock = await loop.run_in_executor(None, yf.Ticker, scrip_code)
        news = await loop.run_in_executor(None, stock.news)
        
        # Process and return the news data
        return [
            {
                'title': item['title'],
                'link': item['link'],
                'publisher': item['publisher'],
                'published_date': item['providerPublishTime'],
                'summary': item.get('summary', '')
            }
            for item in news
        ]
    except Exception as e:
        logger.error(f"Error fetching news data for {scrip_code}: {str(e)}")
        return []

async def process_news_data(news_data):
    sentiments = []
    topics = []
    for article in news_data:
        sentiment = article.get('overall_sentiment_score', 0)
        article_topics = article.get('topics', [])
        sentiments.append(sentiment)
        topics.extend([topic['topic'] for topic in article_topics])
    
    return {
        'sentiment': sum(sentiments) / len(sentiments) if sentiments else 0,
        'topics': list(set(topics))
    }

async def update_influxdb_with_latest_data(scrip_code: str):
    try:
        query_api = client.query_api()
        query = f'''
        from(bucket: "stock_data")
          |> range(start: 0)
          |> filter(fn: (r) => r._measurement == "stock_data" and r.symbol == "{scrip_code}")
          |> keep(columns: ["_time"])
          |> sort(columns: ["_time"], desc: true)
          |> limit(n: 1)
        '''
        result = query_api.query(org="my_org", query=query)
        if not result:
            logger.warning(f"No existing data found for {scrip_code} in InfluxDB.")
            await fetch_process_store_data(scrip_code, '5years')
            return

        last_date = result[0].records[0].get_time()
        start_date = last_date + timedelta(days=1)
        end_date = datetime.now()

        logger.info(f"Fetching latest data for {scrip_code} from {start_date} to {end_date}")
        instrument_token = zerodha_service.get_instrument_token("BSE", scrip_code)
        if not instrument_token:
            logger.error(f"No instrument token found for {scrip_code}")
            return

        data = zerodha_service.get_historical_data(
            instrument_token,
            start_date,
            end_date,
            "day"
        )

        if not data:
            logger.warning(f"No new data found for {scrip_code}.")
            return

        df = pd.DataFrame(data)
        logger.info(f"Columns in the dataframe: {df.columns}")
        logger.info(f"Successfully fetched {len(df)} new data points for {scrip_code}")

        # Calculate technical indicators
        df['RSI'] = calculate_rsi(df['close'])
        df['SMA_20'] = calculate_sma(df['close'], 20)
        df['EMA_50'] = calculate_ema(df['close'], 50)
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['close'])
        df['Bollinger_Upper'], df['Bollinger_Middle'], df['Bollinger_Lower'] = calculate_bollinger_bands(df['close'])
        df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])

        # Write to InfluxDB
        write_stock_data_to_influxdb(df, scrip_code)
        logger.info(f"New data for {scrip_code} processed and stored in InfluxDB.")
    except Exception as e:
        logger.error(f"Error updating data for {scrip_code}: {str(e)}")
        logger.error(traceback.format_exc())