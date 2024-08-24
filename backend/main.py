from fastapi import FastAPI, HTTPException
from app.services.data_collection import fetch_historical_data
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
import logging
import os

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
scheduler = AsyncIOScheduler()

@app.get("/")
async def root():
    return {"message": "BSE Stock Analysis API is running"}


@app.get("/api/historical/{code}")
async def get_historical_data(code: str, days: int = 365):
    """
    Fetch historical data for a given stock code.
    :param code: Stock code (ticker symbol)
    :param days: Number of days for which to fetch historical data
    :return: Historical stock data
    """
    try:
        data = await fetch_historical_data(code, days)
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for stock code {code}")
        return data.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Error fetching historical data for {code}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
