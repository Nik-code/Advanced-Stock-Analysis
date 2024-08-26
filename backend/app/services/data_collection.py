from .zerodha_service import ZerodhaService
import pandas as pd
import os
from datetime import datetime, timedelta
import traceback
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

zerodha_service = ZerodhaService()


async def fetch_historical_data(scrip_code: str, days: int = 365):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # First, get the instrument token
        instruments = zerodha_service.kite.instruments("BSE")
        instrument = next((i for i in instruments if i['tradingsymbol'] == scrip_code), None)
        
        if not instrument:
            raise ValueError(f"No instrument found for {scrip_code}")
        
        instrument_token = instrument['instrument_token']
        
        # Now fetch the historical data
        data = zerodha_service.kite.historical_data(
            instrument_token,
            start_date,
            end_date,
            "day"
        )
        
        if not data:
            raise ValueError(f"No data found for {scrip_code}. It may be delisted.")

        df = pd.DataFrame(data)
        return df
    except Exception as e:
        logger.error(f"Error fetching historical data for {scrip_code}: {str(e)}")
        logger.error(traceback.format_exc())
        return None


async def collect_and_store_data(scrip_codes: list, data_folder: str = 'data'):
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    for scrip_code in scrip_codes:
        df = await fetch_historical_data(scrip_code)
        if df is not None:
            file_path = os.path.join(data_folder, f"{scrip_code}.csv")
            df.to_csv(file_path)
            print(f"Data for {scrip_code} saved to {file_path}")
        else:
            print(f"Skipping {scrip_code} due to missing data.")