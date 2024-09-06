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


async def fetch_news_data(scrip_code: str):
    try:
        logger.info(f"Fetching news data for {scrip_code} from Yahoo Finance")
        
        url = f"https://finance.yahoo.com/quote/{scrip_code}.NS/"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch Yahoo Finance page for {scrip_code}. Status code: {response.status}")
                    return []
                
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Find the news section on the Yahoo Finance page
                news_section = soup.find('section', {'id': 'quoteNewsStream-0-Stream'})
                if not news_section:
                    logger.warning(f"No news section found for {scrip_code} on Yahoo Finance.")
                    return []
                
                news_items = news_section.find_all('li', {'class': 'js-stream-content'})
                if not news_items:
                    logger.warning(f"No news items found for {scrip_code} on Yahoo Finance.")
                    return []

                # Process each news item
                processed_news = []
                for item in news_items:
                    title_element = item.find('h3')
                    if not title_element:
                        continue

                    title = title_element.get_text(strip=True)
                    link = title_element.find('a')['href']
                    link = f"https://finance.yahoo.com{link}" if link.startswith('/') else link
                    publisher_element = item.find('span', {'class': 'C(#959595)'})
                    publisher = publisher_element.get_text(strip=True) if publisher_element else "Unknown Publisher"
                    published_date_element = item.find('time')
                    published_date = published_date_element['datetime'] if published_date_element else "Unknown Date"

                    processed_news.append({
                        'symbol': f"{scrip_code}.NS",
                        'title': title,
                        'link': link,
                        'publisher': publisher,
                        'published_date': published_date,
                        'summary': ''  # Yahoo Finance doesn't always provide a summary, but you can extract it if available
                    })
        
        if not processed_news:
            logger.warning(f"No relevant Indian news found for {scrip_code}")
        
        logger.info(f"Fetched {len(processed_news)} news articles for {scrip_code}")
        return processed_news
    except Exception as e:
        logger.error(f"Error fetching news data for {scrip_code}: {str(e)}")
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
    technical_indicators = await fetch_technical_indicators(news_data[0]['symbol'])

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