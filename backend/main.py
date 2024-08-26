from fastapi import FastAPI, HTTPException, Request
from app.services.data_collection import fetch_historical_data
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv, find_dotenv, set_key
import logging
from app.services.zerodha_service import ZerodhaService
from app.api import stocks
import os
import pandas as pd
import numpy as np

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()  # This line is crucial
scheduler = AsyncIOScheduler()
zerodha_service = ZerodhaService()

app.include_router(stocks.router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "BSE Stock Analysis API is running"}

@app.get("/api/quote")
async def get_quote(instruments: str):
    try:
        quote_data = zerodha_service.get_quote(instruments)
        if quote_data is None:
            raise HTTPException(status_code=404, detail="Quote not found")
        return quote_data
    except Exception as e:
        logger.error(f"Error fetching quote for {instruments}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching quote")

@app.get("/api/historical/{code}")
async def get_historical_data(code: str, days: int = 365):
    try:
        data = await fetch_historical_data(code, days)
        if data is None:
            raise HTTPException(status_code=404, detail=f"No data found for stock code {code}")
        return data.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Error fetching historical data for {code}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/login")
async def login():
    login_url = zerodha_service.get_login_url()
    return {"login_url": login_url}

@app.get("/api/callback")
async def callback(request: Request):
    params = dict(request.query_params)
    request_token = params.get("request_token")
    if not request_token:
        raise HTTPException(status_code=400, detail="No request token provided")
    try:
        access_token = zerodha_service.generate_session(request_token)
        zerodha_service.set_access_token(access_token)
        # Save the access token to .env file
        dotenv_file = find_dotenv()
        set_key(dotenv_file, "ZERODHA_ACCESS_TOKEN", access_token)
        return {"access_token": access_token}
    except Exception as e:
        logger.error(f"Error in callback: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating session")

@app.get("/api/stocks/{symbol}/realtime")
async def get_realtime_data(symbol: str):
    try:
        realtime_data = zerodha_service.get_quote(f"BSE:{symbol}")
        if realtime_data is None or f"BSE:{symbol}" not in realtime_data:
            raise HTTPException(status_code=404, detail=f"No real-time data found for stock symbol {symbol}")
        return realtime_data[f"BSE:{symbol}"]
    except Exception as e:
        logger.error(f"Error fetching real-time data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    return data.rolling(window=window).mean()

def calculate_ema(data: pd.Series, window: int) -> pd.Series:
    return data.ewm(span=window, adjust=False).mean()

def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
    ema_fast = calculate_ema(data, fast_period)
    ema_slow = calculate_ema(data, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    return macd_line, signal_line

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)