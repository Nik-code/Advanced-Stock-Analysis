import pandas as pd
import numpy as np
from typing import Dict, Any, List
import math


def is_json_serializable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

def sanitize_value(value):
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return None
    return value if is_json_serializable(value) else str(value)


def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def process_bse_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "symbol": raw_data.get("scripCode"),
        "name": raw_data.get("companyName"),
        "price": raw_data.get("currentValue"),
        "change": raw_data.get("change"),
        "percent_change": raw_data.get("pChange"),
        "volume": raw_data.get("totalTradedQuantity"),
        "value": raw_data.get("totalTradedValue"),
        "high": raw_data.get("dayHigh"),
        "low": raw_data.get("dayLow"),
        "updated_on": raw_data.get("updatedOn")
    }


def process_multiple_bse_stocks(raw_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed_data = []
    for scrip_code, data in raw_data.items():
        df = pd.DataFrame([data])

        # Convert relevant columns to numeric
        numeric_columns = ['currentValue', 'previousClose', 'dayHigh', 'dayLow', 'totalTradedQuantity']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col].replace(',', ''), errors='coerce')

        # Calculate additional metrics
        df['daily_return'] = (df['currentValue'] - df['previousClose']) / df['previousClose']
        df['volatility'] = (df['dayHigh'] - df['dayLow']) / df['previousClose']
        df['rsi'] = calculate_rsi(df['currentValue'])

        processed_stock = df.to_dict(orient='records')[0]
        processed_stock['scrip_code'] = scrip_code

        # Sanitize values
        processed_stock = {k: sanitize_value(v) for k, v in processed_stock.items()}

        processed_data.append(processed_stock)

    return processed_data


def process_top_gainers_losers(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "symbol": item.get("scripCode"),
            "name": item.get("securityID"),
            "price": item.get("LTP"),
            "change": item.get("change"),
            "percent_change": item.get("pChange")
        }
        for item in raw_data
    ]

def process_indices(raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    indices = raw_data.get("indices", [])
    return [
        {
            "name": index.get("name"),
            "value": index.get("currentValue"),
            "change": index.get("change"),
            "percent_change": index.get("pChange")
        }
        for index in indices
    ]