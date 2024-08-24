from bsedata.bse import BSE
from typing import Dict, Any

b = BSE()

async def fetch_bse_data() -> Dict[str, Any]:
    try:
        # Fetch data for a specific stock (e.g., Reliance Industries)
        data = b.getQuote('500325')  # 500325 is the BSE code for Reliance Industries
        return data
    except Exception as e:
        print(f"Error fetching BSE data: {e}")
        return None

async def fetch_multiple_bse_stocks(symbols: list) -> Dict[str, Any]:
    results = {}
    for symbol in symbols:
        try:
            data = b.getQuote(symbol)
            results[symbol] = data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    return results