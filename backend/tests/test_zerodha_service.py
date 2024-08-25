import unittest
from unittest.mock import patch, MagicMock
from app.services.zerodha_service import ZerodhaService, RateLimiter
import time


class TestZerodhaService(unittest.TestCase):

    def setUp(self):
        self.zerodha_service = ZerodhaService()

    @patch('app.services.zerodha_service.KiteConnect')
    def test_get_quote(self, mock_kite):
        mock_kite.return_value.quote.return_value = {"BSE:500325": {"last_price": 100}}
        result = self.zerodha_service.get_quote(["BSE:500325"])
        self.assertEqual(result, {"BSE:500325": {"last_price": 100}})

    @patch('app.services.zerodha_service.KiteConnect')
    def test_get_quote_error(self, mock_kite):
        mock_kite.return_value.quote.side_effect = Exception("API Error")
        result = self.zerodha_service.get_quote(["BSE:500325"])
        self.assertIsNone(result)

    def test_rate_limiter(self):
        limiter = RateLimiter(max_calls=2, period=1)
        start_time = time.time()
        limiter()
        limiter()
        limiter()
        end_time = time.time()
        self.assertGreaterEqual(end_time - start_time, 1)


if __name__ == '__main__':
    unittest.main()
