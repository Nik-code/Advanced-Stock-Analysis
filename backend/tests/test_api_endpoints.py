import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.zerodha_service import ZerodhaService
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_zerodha_connection():
    zerodha_service = ZerodhaService()
    
    # Test login URL
    login_url = zerodha_service.get_login_url()
    logger.info(f"Login URL: {login_url}")
    
    # Test fetching a quote
    symbol = "RELIANCE"  # You can change this to any stock symbol
    quote = zerodha_service.get_quote(f"BSE:{symbol}")
    if quote:
        logger.info(f"Quote for {symbol}: {quote}")
    else:
        logger.error(f"Failed to fetch quote for {symbol}")
    
    # Test fetching instrument token
    instrument_token = zerodha_service.get_instrument_token("BSE", symbol)
    if instrument_token:
        logger.info(f"Instrument token for {symbol}: {instrument_token}")
    else:
        logger.error(f"Failed to fetch instrument token for {symbol}")

if __name__ == "__main__":
    test_zerodha_connection()