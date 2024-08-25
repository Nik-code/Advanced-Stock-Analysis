from fastapi import FastAPI, HTTPException, Request
from app.services.data_collection import fetch_historical_data
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv, find_dotenv, set_key
import logging
from app.services.zerodha_service import ZerodhaService
import os

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()  # This line is crucial
scheduler = AsyncIOScheduler()
zerodha_service = ZerodhaService()

@app.get("/")
async def root():
    return {"message": "BSE Stock Analysis API is running"}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)