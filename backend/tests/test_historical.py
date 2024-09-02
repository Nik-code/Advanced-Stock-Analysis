import os
from datetime import datetime, timedelta
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.zerodha_service import ZerodhaService
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

zerodha_service = ZerodhaService()

def test_fetch_historical_data(scrip_code: str):
    try:
        instrument_token = zerodha_service.get_instrument_token("BSE", scrip_code)
        if not instrument_token:
            logger.error(f"No instrument token found for {scrip_code}")
            return

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 10)  # Attempt to fetch 10 years of data

        logger.info(f"Fetching historical data for {scrip_code} from {start_date} to {end_date}")
        data = zerodha_service.get_historical_data(
            instrument_token,
            start_date,
            end_date,
            "day"
        )

        if not data:
            logger.warning(f"No data found for {scrip_code}. It may be delisted or data may not be available for the entire period.")
            return

        logger.info(f"Successfully fetched {len(data)} data points for {scrip_code}")
    except Exception as e:
        logger.error(f"Error fetching historical data for {scrip_code}: {str(e)}")

if __name__ == "__main__":
    scrip_code = "RELIANCE"  # Change this to the stock code you want to test
    test_fetch_historical_data(scrip_code)