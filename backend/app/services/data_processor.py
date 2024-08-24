from typing import List, Dict, Any


def process_bse_data(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # This is a placeholder function. Implement actual data processing logic here
    return [
        {
            "symbol": stock["symbol"],
            "name": stock["companyName"],
            "price": stock["currentPrice"],
            "change": stock["change"],
            "volume": stock["volume"]
        }
        for stock in raw_data
    ]