import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

async def fetch_historical_data(scrip_code: str, days: int = 365):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        stock = yf.Ticker(scrip_code)
        df = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching historical data for {scrip_code}: {e}")
        return None
