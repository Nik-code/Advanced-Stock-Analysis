from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.services.data_fetcher import fetch_multiple_bse_stocks, fetch_top_gainers, fetch_top_losers, fetch_indices
from app.services.data_processor import process_multiple_bse_stocks, process_top_gainers_losers, process_indices
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv

load_dotenv()

scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load any global resources here
    print("Starting up...")
    scheduler.start()
    yield
    # Shutdown: Clean up any global resources here
    print("Shutting down...")
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)

# In your fetch_and_process_data function
async def fetch_and_process_data():
    print("Fetching BSE data...")

    # Fetch and process stock data
    scrip_codes = ["500325", "532540", "500180"]  # Example: Reliance, TCS, HDFC Bank
    raw_stock_data = await fetch_multiple_bse_stocks(scrip_codes)
    processed_stock_data = process_multiple_bse_stocks(raw_stock_data)

    # Fetch and process top gainers and losers
    raw_gainers = await fetch_top_gainers()
    raw_losers = await fetch_top_losers()
    processed_gainers = process_top_gainers_losers(raw_gainers)
    processed_losers = process_top_gainers_losers(raw_losers)

    # Fetch and process indices
    raw_indices = await fetch_indices()
    processed_indices = process_indices(raw_indices)

    print("Data processed successfully")

    # TODO: Save processed_data to database or return it as needed
    return {
        "stocks": processed_stock_data,
        "top_gainers": processed_gainers,
        "top_losers": processed_losers,
        "indices": processed_indices
    }

# Schedule the job
scheduler.add_job(fetch_and_process_data, 'interval', minutes=5)

@app.get("/")
async def root():
    return {"message": "BSE Stock Analysis API is running"}

# Include your routes here
from app.api import stocks
app.include_router(stocks.router, prefix="/api/stocks", tags=["stocks"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)