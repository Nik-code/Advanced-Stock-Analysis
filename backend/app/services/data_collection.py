from .zerodha_service import ZerodhaService
import pandas as pd
import os
from datetime import datetime, timedelta
import traceback
import logging
from .technical_indicators import calculate_rsi, calculate_sma, calculate_ema, calculate_macd, calculate_bollinger_bands, calculate_atr
from ..db.influx_writer import write_stock_data_to_influxdb
from ..db.influx_client import get_influxdb_client
from .llm_integration import GPT4Processor
import aiohttp
from bs4 import BeautifulSoup
import yfinance as yf
import asyncio
import urllib.parse
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

zerodha_service = ZerodhaService()
client = get_influxdb_client()

llm_processor = GPT4Processor()

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
        news_analysis = await process_news_data(news_data, df['close'].tolist())
        
        # Add news analysis to the dataframe
        df['News_Sentiment'] = news_analysis['sentiment']
        df['News_Explanation'] = news_analysis['explanation']
        df['News_Summary'] = news_analysis['analysis']

        # Write to InfluxDB
        write_stock_data_to_influxdb(df, scrip_code)
        logger.info(f"Data for {scrip_code} processed and stored in InfluxDB.")
    else:
        logger.error(f"Skipping {scrip_code} due to missing data.")


async def fetch_news_data(scrip_code: str, limit: int = 5):
    try:
        base_url = "https://news.google.com/rss/search"
        query = urllib.parse.urlencode({"q": f"{scrip_code} stock"})
        search_url = f"{base_url}?{query}&hl=en-IN&gl=IN&ceid=IN:en"

        async with aiohttp.ClientSession() as session:
            async with session.get(search_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch news for {scrip_code}. Status code: {response.status}")
                    return []
                
                xml_content = await response.text()
                root = ET.fromstring(xml_content)

                articles = []
                for item in root.findall('.//item')[:limit]:
                    title = item.find('title').text
                    link = item.find('link').text
                    pub_date = item.find('pubDate').text
                    description = item.find('description').text

                    articles.append({
                        'title': title,
                        'link': link,
                        'published_time': pub_date,
                        'summary': description[:200] + '...' if len(description) > 200 else description
                    })
                
                return articles
    except Exception as e:
        logger.error(f"Error fetching news for {scrip_code}: {str(e)}")
        return []


async def process_news_data(news_data, lstm_prediction, symbol, historical_data):
    if not news_data or len(news_data) == 0:
        logger.warning("No news data to process, defaulting sentiment to 0.")

        # Default values if no news data is present
        sentiment_score = 0
        explanation = 'No News Data found'

        # Fetch technical indicators even without news data
        technical_indicators = await fetch_technical_indicators(symbol)

        # Perform final analysis based on other information
        final_analysis = llm_processor.final_analysis(sentiment_score, explanation, lstm_prediction,
                                                      technical_indicators, historical_data)

        result = {
            'sentiment': sentiment_score,
            'explanation': explanation,
            'analysis': final_analysis
        }
        logger.info(f"Final sentiment analysis result without news data: {result}")
        return result

    logger.info(f"Processing {len(news_data)} news articles")

    # Analyze sentiment using the news data
    sentiment_score, explanation = llm_processor.analyze_news_sentiment(news_data)

    # Fetch technical indicators using the symbol from the news data
    technical_indicators = await fetch_technical_indicators(symbol)

    # Perform final analysis combining sentiment score, explanation, LSTM prediction, and technical indicators
    final_analysis = llm_processor.final_analysis(sentiment_score, explanation, lstm_prediction, technical_indicators, historical_data)

    result = {
        'sentiment': sentiment_score,
        'explanation': explanation,
        'analysis': final_analysis
    }

    logger.info(f"Final sentiment analysis result: {result}")
    return result


async def fetch_technical_indicators(symbol):
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