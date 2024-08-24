from typing import List, Dict, Any

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
    return [process_bse_data(data) for data in raw_data.values()]

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