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


async def fetch_top_bse_stocks(limit: int = 100) -> List[str]:
    try:
        # Assuming the BSE library has a method to fetch top stocks
        # You might need to implement this method or use an alternative API
        top_stocks = b.getTopStocks(limit=limit)
        return [stock['scripCode'] for stock in top_stocks]
    except Exception as e:
        print(f"Error fetching top BSE stocks: {e}")
        return []


async def fetch_multiple_bse_stocks(scrip_codes: List[str] = None) -> Dict[str, Any]:
    if not scrip_codes:
        # Fetch top 20 companies by market cap
        top_companies = b.topGainers()  # This actually returns top companies, not just gainers
        scrip_codes = [company['scripCode'] for company in top_companies[:20]]

    results = {}
    for scrip_code in scrip_codes:
        try:
            data = b.getQuote(scrip_code)
            results[scrip_code] = data
        except Exception as e:
            print(f"Error fetching data for scrip code {scrip_code}: {e}")
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