from typing import List, Dict, Any

def process_bse_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "symbol": raw_data.get("securityID"),
        "name": raw_data.get("securityName"),
        "price": raw_data.get("currentValue"),
        "change": raw_data.get("pChange"),
        "volume": raw_data.get("totalTradedVolume")
    }

def process_multiple_bse_stocks(raw_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [process_bse_data(data) for data in raw_data.values()]