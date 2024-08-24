import os
import pandas as pd
from datetime import datetime, timedelta
from bsedata.bse import BSE
import asyncio

b = BSE()


async def fetch_historical_data(scrip_code: str, days: int = 365):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # This is a placeholder. The actual implementation depends on the BSE API's capabilities.
        # You might need to use a different API or web scraping if BSE doesn't provide historical data easily.
        data = b.getPeriodTrend(scrip_code, start_date.strftime('%d%m%Y'), end_date.strftime('%d%m%Y'))

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
        df.set_index('date', inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching historical data for scrip code {scrip_code}: {str(e)}")
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


if __name__ == "__main__":
    scrip_codes = ["500325", "532540", "500180"]  # Example: Reliance, TCS, HDFC Bank
    asyncio.run(collect_and_store_data(scrip_codes))