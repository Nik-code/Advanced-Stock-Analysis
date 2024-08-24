import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta


async def fetch_historical_data(scrip_code: str, days: int = 365):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        stock = yf.Ticker(scrip_code)
        df = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

        # If no data is returned, raise an error
        if df.empty:
            raise ValueError(f"No data found for {scrip_code}. It may be delisted.")

        # Reset index
        df.reset_index(inplace=True)
        return df
    except ValueError as ve:
        print(str(ve))
        return None
    except Exception as e:
        print(f"Error fetching historical data for {scrip_code}: {e}")
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
