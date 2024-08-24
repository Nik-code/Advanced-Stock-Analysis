from fastapi import FastAPI, HTTPException
from app.services.data_fetcher import fetch_multiple_bse_stocks
from app.services.data_processor import process_multiple_bse_stocks
from app.services.ml_predictions import StockPredictor
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
import logging

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
scheduler = AsyncIOScheduler()
predictor = StockPredictor()


async def fetch_and_process_data():
    try:
        logger.info("Fetching BSE data...")
        raw_data = await fetch_multiple_bse_stocks()
        processed_data = process_multiple_bse_stocks(raw_data)

        if len(processed_data) < 2:
            logger.warning("Not enough data to train the model or make predictions.")
            return processed_data

        if not predictor.is_trained:
            logger.info("Training the model...")
            predictor.train(processed_data)

        for stock in processed_data:
            stock['predicted_price'] = predictor.predict(stock)

        logger.info("Data processed and predictions made successfully")
        return processed_data
    except Exception as e:
        logger.error(f"Error in fetch_and_process_data: {str(e)}")
        raise


scheduler.add_job(fetch_and_process_data, 'interval', minutes=5)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the application...")
    scheduler.start()


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down the application...")
    scheduler.shutdown()


@app.get("/")
async def root():
    return {"message": "BSE Stock Analysis API is running"}


@app.get("/api/stocks")
async def get_stocks():
    try:
        data = await fetch_and_process_data()
        if not data:
            raise HTTPException(status_code=404, detail="No stock data available")
        return data
    except Exception as e:
        logger.error(f"Error in get_stocks endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)