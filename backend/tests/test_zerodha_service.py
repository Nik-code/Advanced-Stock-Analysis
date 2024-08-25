import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import time

# Add the parent directory of 'tests' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.zerodha_service import ZerodhaService, RateLimiter


class TestZerodhaService(unittest.TestCase):

    @patch('app.services.zerodha_service.os.getenv')
    @patch('app.services.zerodha_service.KiteConnect')
    def setUp(self, mock_kite, mock_getenv):
        # Mock environment variables
        mock_getenv.side_effect = lambda key: {'ZERODHA_API_KEY': 'mock_api_key',
                                               'ZERODHA_API_SECRET': 'mock_api_secret'}.get(key)

        self.mock_kite = mock_kite
        self.zerodha_service = ZerodhaService()
        self.zerodha_service.kite = self.mock_kite.return_value

    def test_get_quote(self):
        self.mock_kite.return_value.quote.return_value = {"BSE:500325": {"last_price": 100}}
        result = self.zerodha_service.get_quote(["BSE:500325"])
        self.assertEqual(result, {"BSE:500325": {"last_price": 100}})

    def test_get_quote_error(self):
        self.mock_kite.return_value.quote.side_effect = Exception("API Error")
        result = self.zerodha_service.get_quote(["BSE:500325"])
        self.assertIsNone(result)

    def test_rate_limiter(self):
        @RateLimiter(max_calls=2, period=1)
        def test_function():
            pass

        start_time = time.time()
        test_function()
        test_function()
        test_function()
        end_time = time.time()
        self.assertGreaterEqual(end_time - start_time, 1)


if __name__ == '__main__':
    unittest.main()