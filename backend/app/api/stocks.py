from fastapi import APIRouter
from ..services.data_fetcher import fetch_multiple_bse_stocks
from ..services.data_processor import process_multiple_bse_stocks

router = APIRouter()

@router.get("/")
async def get_stocks():
    scrip_codes = ["500325", "532540", "500180"]  # Example: Reliance, TCS, HDFC Bank
    raw_data = await fetch_multiple_bse_stocks(scrip_codes)
    processed_data = process_multiple_bse_stocks(raw_data)
    return processed_data