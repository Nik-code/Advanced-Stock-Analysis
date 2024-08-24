from bsedata.bse import BSE
from typing import Dict, Any, List

b = BSE()

async def fetch_bse_data(scrip_code: str) -> Dict[str, Any]:
    try:
        data = b.getQuote(scrip_code)
        return data
    except Exception as e:
        print(f"Error fetching BSE data for scrip code {scrip_code}: {e}")
        return None

async def fetch_multiple_bse_stocks(scrip_codes: List[str]) -> Dict[str, Any]:
    results = {}
    for scrip_code in scrip_codes:
        data = await fetch_bse_data(scrip_code)
        if data:
            results[scrip_code] = data
    return results

async def fetch_top_gainers() -> List[Dict[str, Any]]:
    try:
        return b.topGainers()
    except Exception as e:
        print(f"Error fetching top gainers: {e}")
        return []

async def fetch_top_losers() -> List[Dict[str, Any]]:
    try:
        return b.topLosers()
    except Exception as e:
        print(f"Error fetching top losers: {e}")
        return []

async def fetch_indices(category: str = 'market_cap/broad') -> Dict[str, Any]:
    try:
        return b.getIndices(category=category)
    except Exception as e:
        print(f"Error fetching indices for category {category}: {e}")
        return {}