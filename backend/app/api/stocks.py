from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def get_stocks():
    # This is a placeholder. Replace with actual data fetching from your database
    return [
        {"symbol": "RELIANCE", "name": "Reliance Industries", "price": 2100, "change": 15, "volume": 1000000},
        {"symbol": "TCS", "name": "Tata Consultancy Services", "price": 3200, "change": -5, "volume": 500000}
    ]