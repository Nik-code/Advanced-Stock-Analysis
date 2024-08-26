import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
import sys
import os
from symbol_lookup import SymbolLookup

# Add the parent directory of 'tests' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.zerodha_service import ZerodhaService, RateLimiter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Zerodha service and symbol lookup
zerodha_service = ZerodhaService()
symbol_lookup = SymbolLookup()

def fetch_instrument_token(tradingsymbol):
    """
    Fetch the instrument token for the given tradingsymbol using the Zerodha API.
    """
    try:
        instruments = zerodha_service.kite.instruments("BSE")
        for instrument in instruments:
            if instrument["tradingsymbol"] == tradingsymbol:
                return instrument["instrument_token"]
        logger.error(f"No instrument token found for tradingsymbol: {tradingsymbol}")
        return None
    except Exception as e:
        logger.error(f"Error fetching instrument token for {tradingsymbol}: {str(e)}")
        return None

def fetch_missing_data(security_code: str, start_date: datetime, end_date: datetime):
    try:
        logger.info(f"Fetching data for {security_code} from {start_date} to {end_date}")

        # Fetch the tradingsymbol using the SymbolLookup class
        tradingsymbol = symbol_lookup.get_tradingsymbol(int(security_code))
        print(f"Tradingsymbol: {tradingsymbol}")
        if not tradingsymbol:
            logger.error(f"Tradingsymbol not found for security code {security_code}")
            return None

        # Fetch the instrument token using the tradingsymbol
        instrument_token = fetch_instrument_token(tradingsymbol)
        if not instrument_token:
            logger.error(f"Instrument token not found for {tradingsymbol}")
            return None

        # Fetch historical data using Zerodha API
        data = zerodha_service.get_historical_data(
            instrument_token,
            start_date.strftime('%Y-%m-%d %H:%M:%S'),
            end_date.strftime('%Y-%m-%d %H:%M:%S'),
            "day"
        )

        if not data:
            logger.warning(f"No data found for {security_code}")
            return None

        logger.info(f"Fetched {len(data)} data points for {security_code}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {security_code}: {str(e)}")
        return None


if __name__ == "__main__":
    # Define stock security code and date range for testing
    stock_security_code = "500325"  # Example: Reliance Industries
    start_date = datetime(2023, 8, 1)
    end_date = datetime(2023, 8, 25)

    # Fetch missing data
    fetched_data = fetch_missing_data(stock_security_code, start_date, end_date)

    # Log the fetched data for review
    if fetched_data:
        for data_point in fetched_data:
            print(data_point)
