import os
import pandas as pd
from datetime import datetime, timedelta
import logging
import asyncio
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.zerodha_service import ZerodhaService
from app.db.influx_writer import write_stock_data_to_influxdb
from app.db.influx_client import get_influxdb_client
from app.services.technical_indicators import calculate_rsi, calculate_sma, calculate_ema, calculate_macd, calculate_bollinger_bands, calculate_atr

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

zerodha_service = ZerodhaService()
client = get_influxdb_client()

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
        return None

async def add_data_to_influxdb(scrip_code: str, time_frame: str = '1year'):
    df = await fetch_historical_data(scrip_code, time_frame)
    if df is not None:
        # Calculate technical indicators
        df['RSI'] = calculate_rsi(df['close'])
        df['SMA_20'] = calculate_sma(df['close'], 20)
        df['EMA_50'] = calculate_ema(df['close'], 50)
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['close'])
        df['Bollinger_Upper'], df['Bollinger_Middle'], df['Bollinger_Lower'] = calculate_bollinger_bands(df['close'])
        df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])

        # Write to InfluxDB
        write_stock_data_to_influxdb(df, scrip_code)
        logger.info(f"Data for {scrip_code} processed and stored in InfluxDB.")
    else:
        logger.error(f"Skipping {scrip_code} due to missing data.")

if __name__ == "__main__":
    scrip_code = "RELIANCE"  # Change this to the stock code you want to add
    asyncio.run(add_data_to_influxdb(scrip_code))