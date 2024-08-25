from kiteconnect import KiteConnect
import os
from dotenv import load_dotenv
import logging
import time

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []

    def __call__(self):
        now = time.time()
        self.calls = [call for call in self.calls if call > now - self.period]
        if len(self.calls) >= self.max_calls:
            sleep_time = self.calls[0] - (now - self.period)
            time.sleep(max(0, sleep_time))
        self.calls.append(time.time())


rate_limiter = RateLimiter(max_calls=5, period=1)  # 5 calls per second


class ZerodhaService:
    def __init__(self):
        self.api_key = os.getenv("ZERODHA_API_KEY")
        self.api_secret = os.getenv("ZERODHA_API_SECRET")
        self.kite = KiteConnect(api_key=self.api_key)
        self.access_token = None

    def get_login_url(self):
        return self.kite.login_url()

    def generate_session(self, request_token):
        data = self.kite.generate_session(request_token, api_secret=self.api_secret)
        self.access_token = data["access_token"]
        self.kite.set_access_token(self.access_token)
        return self.access_token

    def set_access_token(self, access_token):
        self.access_token = access_token
        self.kite.set_access_token(self.access_token)

    @rate_limiter
    def get_quote(self, instruments):
        try:
            return self.kite.quote(instruments)
        except Exception as e:
            logger.error(f"Error fetching quotes: {str(e)}")
            return None

    @rate_limiter
    def get_historical_data(self, instrument_token, from_date, to_date, interval):
        try:
            return self.kite.historical_data(instrument_token, from_date, to_date, interval)
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return None

    @rate_limiter
    def get_holdings(self):
        try:
            return self.kite.holdings()
        except Exception as e:
            logger.error(f"Error fetching holdings: {str(e)}")
            return None

    @rate_limiter
    def get_positions(self):
        try:
            return self.kite.positions()
        except Exception as e:
            logger.error(f"Error fetching positions: {str(e)}")
            return None

    @rate_limiter
    def place_order(self, exchange, tradingsymbol, transaction_type, quantity, price=None, product=None, order_type=None):
        try:
            return self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=transaction_type,
                quantity=quantity,
                price=price,
                product=product or self.kite.PRODUCT_CNC,
                order_type=order_type or self.kite.ORDER_TYPE_MARKET
            )
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None

    @rate_limiter
    def get_orders(self):
        try:
            return self.kite.orders()
        except Exception as e:
            logger.error(f"Error fetching orders: {str(e)}")
            return None
