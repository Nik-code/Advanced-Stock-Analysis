from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.services.data_fetcher import fetch_bse_data, fetch_multiple_bse_stocks
from app.services.data_processor import process_bse_data, process_multiple_bse_stocks
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
    symbols = ["500325", "532540", "500180"]  # Example: Reliance, TCS, HDFC Bank
    raw_data = await fetch_multiple_bse_stocks(symbols)
    if raw_data:
        processed_data = process_multiple_bse_stocks(raw_data)
        print("Data processed successfully")
        # TODO: Save processed_data to database

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